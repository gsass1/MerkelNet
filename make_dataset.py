import argparse
import face_detection
import numpy as np
import os
import os.path
from moviepy.editor import VideoFileClip
import librosa
import cv2
from threading import Lock, Thread
import torch
import time

print('CUDA is available:', torch.cuda.is_available())

T = 50
SR = 16000
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
FPS = 25
F_MAX = 8000
W, H = 48, 48
BATCH_SIZE = 32

WORKERS = 2

DATA_DIR = 'data'
os.makedirs(DATA_DIR, exist_ok=True)

parser = argparse.ArgumentParser(
                    prog='make_dataset',
                    description='Generates a dataset from the Merkel corpus')

parser.add_argument('corpus_path')

args = parser.parse_args()
if not os.path.exists(args.corpus_path):
    print('Corpus path does not exist')
    exit(1)

if os.path.isfile(args.corpus_path):
    print('Corpus path must be a directory')
    exit(1)

corpus_path = args.corpus_path

# detector = face_detection.build_detector(
#   "RetinaNetResNet50", confidence_threshold=.5, nms_iou_threshold=.3)

# Extract video clips
TIMINGS_PATH = os.path.join(corpus_path, 'timings.txt')
timings = open(TIMINGS_PATH, 'r').readlines()

last_clip_date = None
current_video: VideoFileClip|None = None

X = []
Y = []

def convert_clip_part_to_training_example(detector, clip, S, start_frame):
    frames = []
    spectrograms = []

    for i in range(T):
        frame_time = i + start_frame

        frame = clip.get_frame(frame_time)
        frames.append(frame)

        spectrogram_column = int(frame_time / (clip.duration*FPS) * S.shape[1])

        if spectrogram_column < len(S[0]):
            mel_vector = S[:, spectrogram_column]
            spectrograms.append(mel_vector)
            # mel_vector is the Mel spectrogram vector corresponding to the current frame
        else:
            # Handle cases where the frame time exceeds the audio duration
            print('Frame time exceeds audio duration? This should not happen.')
            exit(1)

    frame_np = np.array(frames).astype(np.uint8)
    #print('Detecting faces...')
    detections = detector.batched_detect(frame_np)
    #print('Done Detecting faces...')

    # sanity check
    if len(detections) != T:
        print('Number of detections does not match number of frames')
        exit(1)

    cropped_frames = []
    for i in range(T):
        face = detections[i]
        if len(face) == 0:
            #print('No face detected here, skipping')
            return [], []
        if len(face) > 1:
            #print('Detected more than one face, skipping')
            return [], []
        bb = face[0].astype(int)
        cropped_frame = frames[i][bb[1]:bb[3], bb[0]:bb[2]]
        cropped_frame = cv2.resize(cropped_frame, (W, H))
        cropped_frames.append(cropped_frame)

    # cropped_frames + spectograms is the next training example
    assert len(cropped_frames) == len(spectrograms)
    return cropped_frames, spectrograms

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
    def __init__(self, num, timings, counter):
        Thread.__init__(self)
        self.num = num
        self.timings = timings
        self.counter = counter

        self.detector = face_detection.build_detector(
          "RetinaNetResNet50", confidence_threshold=.5, nms_iou_threshold=.3)


    def run(self):
        last_clip_date = None
        current_video = None
        X, Y = [], []
        current_batch = 0
        for timing in self.timings:
            self.counter.increment()

            clip_data = timing.split("|")
            clip_date, clip_start, clip_end, clip_text, _ = clip_data
            clip_start, clip_end = float(clip_start), float(clip_end)

            #print(f'[{self.num}, {timing_i}, {percent}%] Processing clip {clip_date} "{clip_text}"')
            if clip_date != last_clip_date:
                last_clip_date = clip_date
                clip_path = os.path.join(corpus_path, 'corpus', clip_date, 'video.mp4')
                if not os.path.isfile(clip_path):
                    print('Video file does not exist:', clip_path)
                    continue
                current_video = VideoFileClip(clip_path)

            if current_video is None:
                print('No video loaded')
                exit(1)

            clip = current_video.subclip(clip_start, clip_end)
            clip = clip.set_fps(FPS)

            remaining_frames = int(clip.duration * FPS)
            if remaining_frames < T:
                print('Clip is too short, skipping')
                continue

            # Load the audio with librosa
            audio_path = os.path.join(corpus_path, 'corpus', clip_date, 'audio.wav')
            y, sr = librosa.load(audio_path, sr=SR)

            # Compute the mel spectrogram
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, fmax=F_MAX, n_fft=N_FFT, hop_length=HOP_LENGTH)

            base_idx  = 0
            while remaining_frames > T:
                try:
                    cropped_frames, spectrograms = convert_clip_part_to_training_example(self.detector, clip, S, base_idx)
                    if cropped_frames != [] and spectrograms != []:
                        # cropped_frames + spectograms is the next training example
                        assert len(cropped_frames) == len(spectrograms)
                        X.append(cropped_frames)
                        Y.append(spectrograms)
                        #print(np.array(X).shape, np.array(Y).shape)
                    #else:
                        #print(clip_date, 'from', clip_start, 'to', clip_end, 'did not work out!!')
                except Exception as e:
                    print('Exception:', e)
                    print('Skipping this clip part')

                base_idx += T
                remaining_frames -= T

            while len(X) >= BATCH_SIZE:
                data_path = os.path.join(DATA_DIR, f"batch_{self.num}_{current_batch}.npz")
                print('Saving batch to', data_path)
                np.savez_compressed(data_path, X=X[:BATCH_SIZE], Y=Y[:BATCH_SIZE])
                current_batch += 1
                X = X[BATCH_SIZE:]
                Y = Y[BATCH_SIZE:]

counter = ThreadSafeCounter()

workers = []
print('Starting', WORKERS, 'workers')
timings_per_worker = len(timings) // WORKERS
for i in range(WORKERS):
    start = i * timings_per_worker
    end = start + timings_per_worker
    if i == WORKERS - 1:
        end = len(timings)
    worker = PreprocessWorker(i, timings[start:end], counter)
    worker.start()
    workers.append(worker)

while True:
    for worker in workers:
        if not worker.is_alive():
            worker.join()

    time.sleep(5)
    progress = counter.counter / len(timings) * 100
    print('Progress:', progress, '%')

#timing_i = 0
#current_batch = 0

#for timing in timings:
#    timing_i += 1

#    clip_data = timing.split("|")
#    clip_date, clip_start, clip_end, clip_text, _ = clip_data
#    clip_start, clip_end = float(clip_start), float(clip_end)

#    percent = timing_i / len(timings) * 100
#    percent = round(percent, 2)

#    print(f'[{timing_i}, {percent}%] Processing clip {clip_date} "{clip_text}"')
#    if clip_date != last_clip_date:
#        last_clip_date = clip_date
#        clip_path = os.path.join(corpus_path, 'corpus', clip_date, 'video.mp4')
#        current_video = VideoFileClip(clip_path)

#    if current_video is None:
#        print('No video loaded')
#        exit(1)

#    clip = current_video.subclip(clip_start, clip_end)
#    clip = clip.set_fps(FPS)
#    audio = clip.audio

#    remaining_frames = int(clip.duration * FPS)
#    if remaining_frames < T:
#        print('Clip is too short, skipping')
#        continue

#    # Load the audio with librosa
#    audio_path = os.path.join(corpus_path, 'corpus', clip_date, 'audio.wav')
#    y, sr = librosa.load(audio_path, sr=SR)

#    # Compute the mel spectrogram
#    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, fmax=F_MAX, n_fft=N_FFT, hop_length=HOP_LENGTH)

#    base_idx  = 0
#    while remaining_frames > T:
#        try:
#            cropped_frames, spectrograms = convert_clip_part_to_training_example(clip, S, base_idx)
#            if cropped_frames != [] and spectrograms != []:
#                # cropped_frames + spectograms is the next training example
#                assert len(cropped_frames) == len(spectrograms)
#                X.append(cropped_frames)
#                Y.append(spectrograms)
#                #print(np.array(X).shape, np.array(Y).shape)
#        except Exception as e:
#            print('Exception:', e)
#            print('Skipping this clip part')

#        base_idx += T
#        remaining_frames -= T

#    # TODO: if remaining_frames > 0, add the last frames to another training example

#    while len(X) >= BATCH_SIZE:
#        data_path = os.path.join(DATA_DIR, f"batch_{current_batch}.npz")
#        print('Saving batch to', data_path)
#        np.savez_compressed(data_path, X=X[:BATCH_SIZE], Y=Y[:BATCH_SIZE])
#        current_batch += 1
#        X = X[BATCH_SIZE:]
#        Y = Y[BATCH_SIZE:]
