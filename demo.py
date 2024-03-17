import argparse
import logging
import gradio as gr
import librosa
from moviepy.editor import *
import os
import sys
import numpy as np
import cv2
import torch
import soundfile as sf

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_LIPS
from mediapipe import solutions

from dataset import denormalize
from hparams import HParams, do_arg_parse_with_hparams
from model import MerkelNet

model_path = './models/face_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE)

def convert_clip_part_to_training_example(hparams: HParams, landmarker, clip: VideoFileClip, start_frame):
    frames = []

    # Cut clip to fit training example
    start, end = start_frame / hparams.fps, (start_frame + hparams.temporal_dim) / hparams.fps
    clip = clip.set_fps(hparams.fps)
    clip = clip.subclip(start, end)

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

    # cropped_frames
    cropped_frames = np.array(cropped_frames)

    return cropped_frames

def process_video(in_video, progress=gr.Progress()):
    landmarker = FaceLandmarker.create_from_options(options)
    hparams = HParams()
    clip = VideoFileClip(in_video)
    clip.fps = hparams.fps

    total_frames = int(clip.duration * hparams.fps)
    if total_frames < hparams.temporal_dim:
        return None

    # Calculate clip times by segmenting the clip into parts of temporal_dim frames length
    # but still considering a little bit of overlap as determined by hparams.frame_overlap
    clip_times = []
    remaining_frames = total_frames
    skip_length = hparams.temporal_dim
    base_idx = 0

    while remaining_frames > hparams.temporal_dim:
        clip_times.append(base_idx)
        base_idx += skip_length
        remaining_frames -= skip_length
    
    frames = []
    for clip_start in progress.tqdm(clip_times, unit="parts", desc="Extracting clip parts"):
        cropped_frames = convert_clip_part_to_training_example(hparams, landmarker, clip, clip_start)
        if len(cropped_frames):
            frames.append(cropped_frames)

    X = np.array(frames)

    # B, T, H, W, C -> B, C, T, H ,W
    X = X.astype(np.float32).transpose(0, 4, 1, 2, 3)

    # # normalize pixels
    X[:, :, :, :] /= 255.0

    X = torch.tensor(X, device=device)
    mels = []
    with torch.no_grad():
        for idx in progress.tqdm(range(X.shape[0]), unit="batch", desc="Inferencing"):
            out, _ = model.inference(X[idx].unsqueeze(0))
            mels.append(denormalize(out.squeeze(0).cpu().numpy(), hparams))

    S = np.array(mels)
    # out is (B, T, C)
    S = S.reshape((S.shape[0]*S.shape[1], -1))

    linear_segment = librosa.feature.inverse.mel_to_stft(S.T, sr=hparams.sr, n_fft=hparams.n_fft)
    audio_griffin = librosa.griffinlim(linear_segment, n_iter=32, hop_length=hparams.hop_length)  # Adjust n_iter as needed
    sf.write(f'/tmp/tmp_out.wav', audio_griffin, hparams.sr)

    frames = np.array(frames)
    frames = frames.reshape((frames.shape[0]*frames.shape[1], frames.shape[2], frames.shape[3], frames.shape[4]))

    #video = concatenate([ImageClip(f).set_duration(1/25.0) for f in frames])
    video = clip.copy()
    video.audio = AudioFileClip(f'/tmp/tmp_out.wav')
    video.write_videofile("/tmp/tmp_out.mp4", fps=25)
    return "/tmp/tmp_out.mp4"

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filemode='w',
                    stream=sys.stdout)

    parser = argparse.ArgumentParser(
                    prog='eval',
                    description='Calculate evaluation metrics on a random validation dataset')
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--share", action='store_true')
    args, hparams = do_arg_parse_with_hparams(parser)

    if not os.path.exists(args.checkpoint):
        logging.error('Missing checkpoint')
        exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MerkelNet(hparams).to(device)

    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()

    with gr.Blocks() as demo:
        with gr.Row():
            in_video = gr.Video(label="Input Video")
            out_video = gr.Video(label="Output Video")
        gr.Examples([os.path.join(os.path.dirname(__file__), "examples/cut.mp4"), os.path.join(os.path.dirname(__file__), "examples/short.mp4")], in_video, out_video, process_video, cache_examples=False)
        btn = gr.Button("Process")
        btn.click(process_video, inputs=in_video, outputs=out_video)

    demo.launch(share=args.share, server_name="0.0.0.0")