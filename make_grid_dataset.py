import argparse
import logging
import random
import sys
from ultralytics import YOLO
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
import face_detection

from tqdm import tqdm

from hparams import HParams, do_arg_parse_with_hparams

def process_video(hparams, video_path, detector):
    # Load video
    clip = VideoFileClip(video_path)
    audio = clip.audio.to_soundarray(fps=hparams.sr)
    audio = nr.reduce_noise(audio, sr=hparams.sr)
    
    # Extract mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=audio[:, 0], sr=hparams.sr, n_mels=hparams.n_mels)
    
    frames = []
    for frame in clip.iter_frames():
        # Detect face
        results = detector(frame)
        
        # If face is detected
        if results.xyxy[0].shape[0] > 0:
            # Extract the first detected face
            x1, y1, x2, y2, _, _ = results.xyxy[0][0].cpu().numpy().astype(int)
            face = frame[y1:y2, x1:x2]
            
            # Resize face to 96x96
            face_resized = cv2.resize(face, (96, 96))
            frames.append(face_resized)
    
    # Normalize frames
    S_normalized = normalize(mel_spectrogram)

    frames_normalized = np.array(frames)
    frames_normalized /= 255.0

    return frames, S_normalized

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
    parser.add_argument('speaker_path')
    args, hparams = do_arg_parse_with_hparams(parser)

    if not os.path.exists(args.speaker_path):
        logging.error('Corpus path does not exist')
        exit(1)

    if os.path.isfile(args.speaker_path):
        logging.error('Corpus path must be a directory')
        exit(1)

    os.makedirs(hparams.data_dir, exist_ok=True)
    logging.info(f'Saving dataset to {hparams.data_dir}')

    detector = YOLO('./models/face_yolov8m.pt')

    files = os.listdir(args.speaker_path)

    for file in tqdm(files):
        video_path = os.path.join(args.speaker_path, file)
        if not video_path.endswith('.mpg'):
            continue
        
        frames, mel = process_video(hparams, video_path, detector)

        print(frames.shape)
        print(mel.shape)
        # Save as numpy dictionary
        #np.savez(os.path.join(hparams.data_dir, f'{os.path.splitext(file)[0]}.npz'), X=frames, Y=mel)

if __name__ == '__main__':
    main()