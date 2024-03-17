import argparse
import logging
import random
import sys
import numpy as np
import os
import os.path
from moviepy.editor import VideoFileClip
import librosa
import cv2
from threading import Lock, Thread
import time
import tempfile
import noisereduce as nr
from dataset import normalize
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_LIPS
from mediapipe import solutions

model_path = './models/face_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE)

from tqdm import tqdm

from hparams import HParams, do_arg_parse_with_hparams

def convert_clip_part_to_training_example(hparams: HParams, landmarker, clip: VideoFileClip, start_frame):
    frames = []

    # Cut clip to fit training example
    start, end = start_frame / hparams.fps, (start_frame + hparams.temporal_dim) / hparams.fps
    #print('Subclipping', start, end)
    clip = clip.set_fps(hparams.fps)
    clip = clip.subclip(start, end)
    #clip = clip.set_duration(hparams.temporal_dim / hparams.fps)
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

        # Reduce noise
        y_reduced = nr.reduce_noise(y, sr=hparams.sr)

        S = librosa.feature.melspectrogram(
                y=y_reduced,
                sr=hparams.sr,
                n_mels=hparams.n_mels,
                fmax=hparams.f_max,
                n_fft=hparams.n_fft,
                hop_length=hparams.hop_length)

    # note: S should have shape (n_mels, temporal_dim+1)
    assert S.shape[0] == hparams.n_mels
    assert S.shape[1] == hparams.temporal_dim+1

    S = normalize(S, hparams)

    S = S[:, :-1]
    S = S.transpose(1, 0)

    landmarks = []

    frames = []
    for i in range(hparams.temporal_dim):
        frame_time = i / hparams.fps
        frame = clip.get_frame(frame_time)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        result = landmarker.detect(mp_image)

        # no face or more than once face
        if len(result.face_landmarks) != 1:
            return [], []
        frames.append(frame)
        landmarks.append(result.face_landmarks[0])

    # smoothe mean of the mouth position
    mxx = []
    myy = []
    for i in range(hparams.temporal_dim):
        all_points = []

        for j in FACEMESH_LIPS:
            all_points.append(landmarks[i][j[0]])
            all_points.append(landmarks[i][j[1]])
        mx = np.mean([point.x for point in all_points])
        my = np.mean([point.y for point in all_points])
        mxx.append(mx)
        myy.append(my)

    def smooth_positions(positions, window_size=5):
        smoothed = []
        for i in range(len(positions)):
            # Determine the start and end of the window for averaging
            start = max(0, i - window_size + 1)
            end = i + 1
            # Calculate the average within this window
            window_average = np.mean(positions[start:end])
            smoothed.append(window_average)
        return smoothed

    # Assuming mxx and myy are filled with the mean x and y positions of the mouth
    window_size = 5  # Adjust this value as needed for your smoothing
    smoothed_mxx = smooth_positions(mxx, window_size=window_size)
    smoothed_myy = smooth_positions(myy, window_size=window_size)

    cropped_frames = []
    reference_distance = 100

    for i in range(hparams.temporal_dim):
        frame = frames[i]
        frame = cv2.resize(frame, (1280, 720))

        all_points = []

        for j in FACEMESH_LIPS:
            all_points.append(landmarks[i][j[0]])
            all_points.append(landmarks[i][j[1]])

        mx = smoothed_mxx[i]
        my = smoothed_myy[i]

        left_corner_index = 207
        right_corner_index = 427

        lip_corner_left = landmarks[i][left_corner_index]  # You need to define left_corner_index
        lip_corner_right = landmarks[i][right_corner_index]  # You need to define right_corner_index
        actual_distance = np.sqrt((lip_corner_right.x - lip_corner_left.x)**2 + (lip_corner_right.y - lip_corner_left.y)**2)
        actual_distance *= frame.shape[1]  # Adjust for frame size

        # angle = calculate_angle(lip_corner_left, lip_corner_right)
        # center_of_rotation = ((lip_corner_left[0] + lip_corner_right[0]) / 2 * frame.shape[1], (lip_corner_left[1] + lip_corner_right[1]) / 2 * frame.shape[0])
        # rotated_frame = rotate_image(frame, -angle, center=center_of_rotation)

        scale_factor = actual_distance / reference_distance

        x = int(mx * frame.shape[1])
        y = int(my * frame.shape[0])
        w = int(hparams.w//2*scale_factor)
        h = int(hparams.h//2*scale_factor)

        cropped_frame = cv2.resize(frame[y-h:y+h, x-w:x+w], (hparams.w,hparams.h))
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
        self.landmarker = FaceLandmarker.create_from_options(options)

    def run(self):
        last_clip_date = None
        current_video = None
        X, Y = [], []
        current_batch = 0
        skip = 0
        with tqdm(enumerate(self.timings), unit="timings", total=len(self.timings)) as ttimings:
            for idx, timing in ttimings:
                if idx == skip:
                    break

            for idx, timing in ttimings:
                try:
                    if self.counter is not None: self.counter.increment()

                    clip_data = timing.split("|")
                    clip_date, clip_start, clip_end, clip_text, _ = clip_data
                    # skip interviewer questions, presumably
                    if "?" in clip_text: continue
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
                        except Exception as e:
                            logging.error('Failed to load video, skipping', e)
                            continue

                    if current_video is None:
                        logging.error('No video loaded')
                        exit(1)

                    clip = current_video.subclip(clip_start, clip_end)
                    #clip = clip.set_duration(clip_duration)

                    total_frames = int(clip_duration * self.hparams.fps)
                    if total_frames < self.hparams.temporal_dim:
                        #logging.info('Clip is too short, skipping')
                        continue

                    # Calculate clip times by segmenting the clip into parts of temporal_dim frames length
                    # but still considering a little bit of overlap as determined by hparams.frame_overlap
                    clip_times = []
                    remaining_frames = total_frames
                    skip_length = self.hparams.temporal_dim - self.hparams.frame_overlap
                    base_idx = 0

                    while remaining_frames > self.hparams.temporal_dim:
                        clip_times.append(base_idx)
                        base_idx += skip_length
                        remaining_frames -= skip_length

                    if remaining_frames > 0:
                        clip_times.append((total_frames - self.hparams.temporal_dim))

                    #print(clip_times)

                    with tqdm(clip_times, unit="parts") as tparts:
                        for clip_start in tparts:
                            #print('Extracting clip parts at', base_idx, 'remaining', remaining_frames, 'total', total_frames)
                            try:
                                cropped_frames, spectrograms = convert_clip_part_to_training_example(self.hparams, self.landmarker, clip, clip_start)
                                if len(cropped_frames) != 0 and len(spectrograms) != 0:
                                    X.append(cropped_frames)
                                    Y.append(spectrograms)
                            except Exception as e:
                                #print('Error in convert_clip_part_to_training_example', e)
                                logging.error(e)
                                #raise e

                    while len(X) >= self.hparams.dataset_batch_size:
                        data_path = os.path.join(self.hparams.data_dir, f"batch_{self.num}_{current_batch}.npz")
                        logging.debug('Saving batch to ' + data_path)

                        batch_x = np.array(X[:self.hparams.dataset_batch_size])
                        batch_y = np.array(Y[:self.hparams.dataset_batch_size])

                        # B, T, H, W, C -> B, C, T, H ,W
                        batch_x = batch_x.astype(np.float32).transpose(0, 4, 1, 2, 3)

                        # # normalize pixels
                        batch_x[:, :, :, :] /= 255.0

                        np.savez_compressed(data_path, X=batch_x, Y=batch_y)
                        current_batch += 1

                        X = X[self.hparams.dataset_batch_size:]
                        Y = Y[self.hparams.dataset_batch_size:]

                except Exception as e:
                    print('Error in timing loop', e)
                    #raise e

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
        progress = counter.counter / len(timings) * 100
        #print('Progress:', progress, '%')

if __name__ == '__main__':
    main()

