# You should already have videos splitted into test train val folders. Each containg real-fake folders contaning videos

import os
import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN
from tqdm import tqdm

class FaceExtractor:
    def __init__(self, conf_threshold=0.9, verif_threshold=0.9, min_face_size=40):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.detector = MTCNN(
            keep_all=True,
            thresholds=[0.7, 0.8, 0.9],
            post_process=False,
            device=self.device
        )
        self.conf_threshold = conf_threshold     # first‐pass detection threshold
        self.verif_threshold = verif_threshold   # second‐pass verification threshold
        self.min_face_size = min_face_size

    def extract_faces(self, video_path, output_dir, frames_per_video=10):
        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_step = max(1, total_frames // frames_per_video)

        saved_count = 0
        processed_frames = 0

        with torch.no_grad():
            for frame_num in tqdm(range(0, total_frames, frame_step),
                                  desc=f"Processing {os.path.basename(video_path)}"):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                if not ret:
                    continue

                processed_frames += 1

                # --- 1st Pass: detect & crop ---
                try:
                    # convert BGR->RGB and to tensor
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # MTCNN.detect can take numpy arrays directly
                    boxes, probs = self.detector.detect(rgb)

                    if boxes is None:
                        continue

                    # filter small / low‐conf faces
                    valid = [
                        i for i, (b, p) in enumerate(zip(boxes, probs))
                        if (b[2]-b[0] >= self.min_face_size and
                            b[3]-b[1] >= self.min_face_size and
                            p >= self.conf_threshold)
                    ]
                    if not valid:
                        continue

                    # pick highest‐confidence face
                    best = valid[np.argmax(probs[valid])]
                    x1, y1, x2, y2 = boxes[best].astype(int)

                    # expand by 20%
                    w, h = x2-x1, y2-y1
                    x1 = max(0, x1 - int(0.2*w))
                    y1 = max(0, y1 - int(0.2*h))
                    x2 = min(frame.shape[1], x2 + int(0.2*w))
                    y2 = min(frame.shape[0], y2 + int(0.2*h))

                    face_crop = frame[y1:y2, x1:x2]

                    # --- 2nd Pass: verify the crop is really a face ---
                    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    # re-run detection on the cropped patch
                    v_boxes, v_probs = self.detector.detect(face_rgb)

                    # if no detection or max prob < verif_threshold, skip
                    if v_boxes is None or np.max(v_probs) < self.verif_threshold:
                        # optionally log or count rejects here
                        continue

                    # passed verification — save!
                    out_path = os.path.join(output_dir, f"face_{saved_count:04d}.jpg")
                    cv2.imwrite(out_path, face_crop)
                    saved_count += 1

                    if saved_count >= frames_per_video:
                        break

                except Exception as e:
                    print(f"Frame {frame_num}: {e}")
                    continue

        cap.release()
        return saved_count > 0, processed_frames, saved_count



def process_split(split_dir, output_root,frames_per_video):
    """Process all videos in a split using batch processing for improved speed."""
    extractor = FaceExtractor()

    for split in ['train', 'val', 'test']:
        for label in ['real', 'fake']:
            input_dir = os.path.join(split_dir, split, label)
            video_files = [f for f in os.listdir(input_dir) if f.endswith('.mp4')]

            for video in tqdm(video_files, desc=f"Processing {split}/{label}"):
                video_path = os.path.join(input_dir, video)
                output_dir = os.path.join(output_root, split, label, video[:-4])
                os.makedirs(output_dir, exist_ok=True)
                extractor.extract_faces(video_path, output_dir,frames_per_video)

if __name__ == "__main__":
    frames_per_video=32
    process_split('Sample_data/FF++_videos/Splitted', 'Sample_data/FF++_videos/Extracted/',frames_per_video)
