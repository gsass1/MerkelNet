import argparse
import logging
import random
import sys
import face_detection
import numpy as np
import os
import os.path
from moviepy.editor import VideoFileClip
import librosa
import cv2
from threading import Lock, Thread
import time
import tempfile

from tqdm import tqdm

from hparams import HParams, do_arg_parse_with_hparams

def convert_clip_part_to_training_example(hparams: HParams, detector, clip: VideoFileClip, start_frame):
    frames = []

    # Cut clip to fit training example
    start, end = start_frame / hparams.fps, (start_frame + hparams.temporal_dim) / hparams.fps
    #print('Subclipping', start, end)
    clip = clip.subclip(start, end)
    clip = clip.set_duration(hparams.temporal_dim / hparams.fps)
    clip = clip.set_fps(hparams.fps)
    audio = clip.audio

    if audio is None:
        logging.warning('No audio for this clip part')
        return [], []

    S = None

    # Create a temporary audio file
    with tempfile.NamedTemporaryFile(suffix='.wav') as temp_audio_file:
        audio = audio.subclip(0, hparams.temporal_dim / hparams.fps)
        audio = audio.set_duration(hparams.temporal_dim / hparams.fps)
        audio.write_audiofile(temp_audio_file.name, codec='pcm_s16le', verbose=False, write_logfile=False, logger=None)
    
        # Load the audio with librosa
        y, _ = librosa.load(temp_audio_file.name, sr=hparams.sr)
        S = librosa.feature.melspectrogram(
                y=y,
                sr=hparams.sr,
                n_mels=hparams.n_mels,
                fmax=hparams.f_max,
                n_fft=hparams.n_fft,
                hop_length=hparams.hop_length)

    # note: S should have shape (n_mels, temporal_dim+1)
    assert S.shape[0] == hparams.n_mels
    assert S.shape[1] == hparams.temporal_dim+1

    S = S[:, :-1]
    S = S.transpose(1, 0)

    for i in range(hparams.temporal_dim):
        frame_time = (i + start_frame) / hparams.fps

        frame = clip.get_frame(frame_time)
        frames.append(frame)

        # if spectrogram_column < len(S[0]):
        #     mel_vector = S[:, spectrogram_column]
        #     spectrograms.append(mel_vector)
        #     # mel_vector is the Mel spectrogram vector corresponding to the current frame
        # else:
        #     # Handle cases where the frame time exceeds the audio duration
        #     print('Frame time exceeds audio duration? This should not happen.')
        #     exit(1)

    frame_np = np.array(frames).astype(np.uint8)
    #print('Detecting faces...')
    detections = detector.batched_detect(frame_np)
    #print('Done Detecting faces...')

    # sanity check
    if len(detections) != hparams.temporal_dim:
        logging.debug('Number of detections does not match number of frames')
        exit(1)

    bounding_boxes = []
    for i in range(hparams.temporal_dim):
        face = detections[i]
        if len(face) == 0:
            logging.debug('No face detected here, skipping')
            return [], []
        if len(face) > 1:
            logging.debug('Detected more than one face, skipping')
            return [], []
        bb = face[0]
        bounding_boxes.append(bb)

    # smoothen face frames since they are quite jiggly
    # first, lets hard-code the width and height, since those barely change anyway
    # xmin, ymin, xmax, ymax
    bounding_boxes = np.array(bounding_boxes)
    bounding_boxes = bounding_boxes[:, :-1]

    xmin = bounding_boxes[:, 0]
    ymin = bounding_boxes[:, 1]
    xmax = bounding_boxes[:, 2]
    ymax = bounding_boxes[:, 3]

    w, h = xmax[0] - xmin[0], ymax[0] - ymin[0]

    # now lets smoothe out xmin and ymin throughout the entire clip
    smooth_x, smooth_y = np.zeros(hparams.temporal_dim), np.zeros(hparams.temporal_dim)
    for t in range(1, hparams.temporal_dim-1):
        smooth_x[t] = (xmin[t+1]+xmin[t-1])/2
        smooth_y[t] = (ymin[t+1]+ymin[t-1])/2

    smooth_x[0] = xmin[0]
    smooth_y[0] = ymin[0]

    smooth_x[hparams.temporal_dim-1] = xmin[hparams.temporal_dim-1]
    smooth_y[hparams.temporal_dim-1] = ymin[hparams.temporal_dim-1]

    # bounding_boxes[:, 0] = smooth_x
    # bounding_boxes[:, 1] = smooth_y
    bounding_boxes[:, 0] = np.repeat(xmin[0], hparams.temporal_dim)
    bounding_boxes[:, 1] = np.repeat(ymin[0], hparams.temporal_dim)
    bounding_boxes[:, 2] = bounding_boxes[:, 0] + w
    bounding_boxes[:, 3] = bounding_boxes[:, 1] + h
    
    cropped_frames = []
    for i in range(hparams.temporal_dim):
        bb = bounding_boxes[i].astype(int)
        cropped_frame = frames[i][bb[1]:bb[3], bb[0]:bb[2]]
        cropped_frame = cv2.resize(cropped_frame, (hparams.h, hparams.w))
        cropped_frames.append(cropped_frame)

    # cropped_frames + spectograms is the next training example
    cropped_frames = np.array(cropped_frames)
    assert cropped_frames.shape[0] == S.shape[0]
    return cropped_frames, S

class ThreadSafeCounter():
    def __init__(self):
        self.lock = Lock()
        self.counter=0

    def increment(self):
        with self.lock:
            self.counter+=1


    def decrement(self):
        with self.lock:
            self.counter-=1

class PreprocessWorker(Thread):
    hparams: HParams

    def __init__(self, num, corpus_path, hparams, timings, counter):
        Thread.__init__(self)
        self.num = num
        self.timings = timings
        self.corpus_path = corpus_path
        self.hparams = hparams
        self.counter = counter

        self.detector = face_detection.build_detector(
          "RetinaNetResNet50", confidence_threshold=.5, nms_iou_threshold=.3)


    def run(self):
        last_clip_date = None
        current_video = None
        X, Y = [], []
        current_batch = 0
        with tqdm(enumerate(self.timings), unit="timings", total=len(self.timings)) as ttimings:
            for idx, timing in ttimings:
                #logging.info(timing)

                self.counter.increment()

                clip_data = timing.split("|")
                clip_date, clip_start, clip_end, _, _ = clip_data
                clip_start, clip_end = float(clip_start), float(clip_end)
                clip_duration = clip_end - clip_start

                if clip_date != last_clip_date:
                    last_clip_date = clip_date
                    clip_path = os.path.join(self.corpus_path, 'corpus', clip_date, 'video.mp4')
                    if not os.path.isfile(clip_path):
                        logging.warning('Video file does not exist:', clip_path)
                        continue
                    try:
                        current_video = VideoFileClip(clip_path)
                    except:
                        logging.error('Failed to load video, skipping')
                        continue

                if current_video is None:
                    logging.error('No video loaded')
                    exit(1)

                clip = current_video.subclip(clip_start, clip_end)
                clip = clip.set_duration(clip_duration)

                total_frames = int(clip_duration * self.hparams.fps)
                if total_frames < self.hparams.temporal_dim:
                    logging.info('Clip is too short, skipping')
                    continue

                clip_times = []
                remaining_frames = total_frames
                base_idx = 0

                while remaining_frames > self.hparams.temporal_dim:
                    clip_times.append(base_idx)
                    base_idx += self.hparams.temporal_dim
                    remaining_frames -= self.hparams.temporal_dim

                if remaining_frames > 0:
                    clip_times.append((total_frames - self.hparams.temporal_dim))

                #print(clip_times)

                with tqdm(clip_times, unit="parts") as tparts:
                    for clip_start in tparts:
                        #print('Extracting clip parts at', base_idx, 'remaining', remaining_frames, 'total', total_frames)
                        try:
                            cropped_frames, spectrograms = convert_clip_part_to_training_example(self.hparams, self.detector, clip, clip_start)
                            if len(cropped_frames) != 0 and len(spectrograms) != 0:
                                X.append(cropped_frames)
                                Y.append(spectrograms)
                        except Exception as e:
                            logging.error(e)

                while len(X) >= self.hparams.dataset_batch_size:
                    data_path = os.path.join(self.hparams.data_dir, f"batch_{self.num}_{current_batch}.npz")
                    logging.debug('Saving batch to', data_path)
                    np.savez_compressed(data_path, X=X[:self.hparams.dataset_batch_size], Y=Y[:self.hparams.dataset_batch_size])
                    current_batch += 1
                    X = X[self.hparams.dataset_batch_size:]
                    Y = Y[self.hparams.dataset_batch_size:]

def main():
    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filemode='w',
                    stream=sys.stdout)

    parser = argparse.ArgumentParser(
                    prog='make_dataset',
                    description='Generates a dataset from the Merkel corpus')
    parser.add_argument("--workers", default=1, required=False, type=int)
    parser.add_argument('corpus_path')
    args, hparams = do_arg_parse_with_hparams(parser)

    if not os.path.exists(args.corpus_path):
        logging.error('Corpus path does not exist')
        exit(1)

    if os.path.isfile(args.corpus_path):
        logging.error('Corpus path must be a directory')
        exit(1)

    os.makedirs(hparams.data_dir, exist_ok=True)
    logging.info(f'Saving dataset to {hparams.data_dir}')

    # Extract timing data
    TIMINGS_PATH = os.path.join(args.corpus_path, 'timings.txt')
    timings = open(TIMINGS_PATH, 'r').readlines()

    num_workers = args.workers
    logging.info(f'Starting {num_workers} workers')
    workers = []
    timings_per_worker = len(timings) // num_workers

    counter = ThreadSafeCounter()

    for i in range(num_workers):
        start = i * timings_per_worker
        end = start + timings_per_worker
        if i == num_workers - 1:
            end = len(timings)
        worker = PreprocessWorker(i, args.corpus_path, hparams, timings[start:end], counter)
        worker.start()
        workers.append(worker)

    while True:
        for worker in workers:
            if not worker.is_alive():
                worker.join()

        time.sleep(5)
        # progress = counter.counter / len(timings) * 100
        # print('Progress:', progress, '%')

if __name__ == '__main__':
    main()

