import argparse
import torch
import cv2
import os
import threading
import queue
from torchvision.transforms import Compose, Resize
from tqdm import tqdm
import numpy as np

from train_classifier import VideoClassifier
from train_lstm import LSTMClassifier


def frame_producer(video_path, clip_queue, config):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        clip_queue.put(None)
        return

    frames_buffer = []
    transform = Compose([Resize(size=config['target_size'])])
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        frames_buffer.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if len(frames_buffer) < config['frames_per_clip']: continue
            
        if frame_count % config['sampling_stride'] == 0:
            clip_frames = frames_buffer[-config['frames_per_clip']:]
            clip_np = np.stack(clip_frames)
            clip_tensor = torch.from_numpy(clip_np).permute(3, 0, 1, 2).float() / 255.0
            transformed_clip = transform(clip_tensor)
            clip_queue.put(transformed_clip)

        if len(frames_buffer) > config['frames_per_clip'] * 2:
            frames_buffer.pop(0)

    cap.release()
    clip_queue.put(None)

def inference_consumer(clip_queue, results_queue, config):
    device = config['device']
    x3d_model = VideoClassifier.load_from_checkpoint(config['x3d_checkpoint_path']).to(device)
    x3d_model.model.blocks[5].proj = torch.nn.Identity()
    x3d_model.eval()

    lstm_model = LSTMClassifier(
        config["input_dim"], config["hidden_dim"], config["num_layers"], config["num_classes"], config["dropout"]
    ).to(device)
    lstm_model.load_state_dict(torch.load(config['lstm_checkpoint_path'], map_location=device))
    lstm_model.eval()
    
    feature_vector_buffer = []
    
    with torch.no_grad():
        while True:
            clip_tensor = clip_queue.get()
            if clip_tensor is None:
                if feature_vector_buffer:
                    sequence_tensor = torch.stack(feature_vector_buffer).unsqueeze(0).to(device)
                    output_logit = lstm_model(sequence_tensor)
                    probability = torch.sigmoid(output_logit).item()
                    results_queue.put(probability)
                break
            
            clip_tensor_batched = clip_tensor.unsqueeze(0).to(device)
            feature_vector = x3d_model(clip_tensor_batched).squeeze(0).cpu()

            feature_vector_buffer.append(feature_vector)
            
            if len(feature_vector_buffer) == config['clips_per_sequence']:
                sequence_tensor = torch.stack(feature_vector_buffer).unsqueeze(0).to(device)
                
                output_logit = lstm_model(sequence_tensor)
                probability = torch.sigmoid(output_logit).item()
                
                results_queue.put(probability)
                feature_vector_buffer.clear() # Reset for the next chunk

    results_queue.put(None)


def main(config):
    clip_queue = queue.Queue(maxsize=config['queue_size'])
    results_queue = queue.Queue()

    producer_thread = threading.Thread(target=frame_producer, args=(config['demo_video_path'], clip_queue, config))
    consumer_thread = threading.Thread(target=inference_consumer, args=(clip_queue, results_queue, config))

    producer_thread.start()
    consumer_thread.start()

    cap = cv2.VideoCapture(config['demo_video_path'])
    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
    fps, total_frames = int(cap.get(5)), int(cap.get(7))

    output_path = config['output_video_path']
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    print(f"Processing {total_frames} frames. Output will be saved to: {output_path}")

    FRAMES_PER_CHUNK = config['clips_per_sequence'] * config['sampling_stride']
    frame_counter_for_chunk = 0
    
    # Initialize with a neutral "cold start" status
    latest_verdict = "ANALYZING..."
    latest_confidence_text = "Confidence: N/A"
    latest_color = (200, 200, 200) 

    with tqdm(total=total_frames, desc="Processing Video") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret: break

            if frame_counter_for_chunk == FRAMES_PER_CHUNK:
                try:
                    new_prob = results_queue.get(timeout=10) # 10-sec timeout for safety
                    if new_prob is None: break

                    # Update the status that will be used for the *next* chunk
                    latest_verdict = "CRASH DETECTED" if new_prob > 0.85 else "Normal Traffic"
                    latest_confidence_text = f"Confidence: {new_prob:.2f}"
                    latest_color = (0, 0, 255) if latest_verdict == "CRASH DETECTED" else (0, 255, 0)
                    
                    frame_counter_for_chunk = 0 
                except queue.Empty:
                    print("Warning: Consumer thread is not producing results. Exiting.")
                    break
            
            cv2.rectangle(frame, (10, 10), (350, 90), (0, 0, 0), -1)
            cv2.putText(frame, latest_verdict, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, latest_color, 3)
            cv2.putText(frame, latest_confidence_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            out.write(frame)
            pbar.update(1)
            frame_counter_for_chunk += 1

    cap.release()
    out.release()
    producer_thread.join()
    consumer_thread.join()
    print(f"\nProcessing complete. Annotated video saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_video', required=True, help="Path to input video file")
    parser.add_argument('--output_video', default="output.mp4", help="Path to save the output video file")
    args = parser.parse_args()

    CONFIG = {
        "demo_video_path": args.input_video,
        "output_video_path": args.output_video,
        "x3d_checkpoint_path": "checkpoints/x3d_m-classifier.ckpt",
        "lstm_checkpoint_path": "checkpoints/best_lstm_model.ckpt",
        "sampling_stride": 10, 
        "queue_size": 30,
        "clips_per_sequence": 10,
        "frames_per_clip": 16,
        "target_size": (224, 224),
        "input_dim": 2048,
        "hidden_dim": 256,
        "num_layers": 2,
        "num_classes": 1,
        "dropout": 0.5,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    main(CONFIG)