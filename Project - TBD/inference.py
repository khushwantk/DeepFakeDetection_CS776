import os
import cv2
import time
import torch
import gradio as gr
import tempfile
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
from torchvision import models, transforms
from facenet_pytorch import MTCNN
import torch.nn as nn
from collections import OrderedDict
import zipfile
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import timm
import gc

# ----------------------------------------------------
# 1. Face Extraction Using MTCNN (with added stats and threshold support)
# ----------------------------------------------------
class FaceExtractor:
    def __init__(self, conf_threshold=0.9):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.detector = MTCNN(
            keep_all=True,
            thresholds=[0.7, 0.8, 0.9],
            post_process=False,
            device=self.device
        )
        self.min_face_size = 40
        self.conf_threshold = conf_threshold

    def extract_faces(self, video_path, output_dir, frames_per_video=10):
        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_step = max(1, total_frames // frames_per_video)
        saved_count = 0
        processed_frames = 0  # Count how many frames were processed

        with torch.no_grad():
            for frame_num in tqdm(range(0, total_frames, frame_step),
                                  desc=f"Processing {os.path.basename(video_path)}"):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                if not ret:
                    continue

                processed_frames += 1

                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_tensor = torch.from_numpy(frame_rgb).to(self.device)

                    # Detect faces
                    boxes, probs = self.detector.detect(frame_tensor)

                    if boxes is not None:
                        if isinstance(boxes, torch.Tensor):
                            boxes = boxes.cpu().numpy()
                            probs = probs.cpu().numpy()

                        valid_indices = [
                            i for i, (box, prob) in enumerate(zip(boxes, probs))
                            if (box[2] - box[0]) >= self.min_face_size and
                               (box[3] - box[1]) >= self.min_face_size and
                               prob > self.conf_threshold
                        ]

                        if valid_indices:
                            best_idx = valid_indices[np.argmax(probs[valid_indices])]
                            x1, y1, x2, y2 = boxes[best_idx].astype(int)

                            # Expand bounding box by 20%
                            w = x2 - x1
                            h = y2 - y1
                            x1 = max(0, x1 - int(w * 0.2))
                            y1 = max(0, y1 - int(h * 0.2))
                            x2 = min(frame.shape[1], x2 + int(w * 0.2))
                            y2 = min(frame.shape[0], y2 + int(h * 0.2))

                            face_img = frame[y1:y2, x1:x2]
                            output_path = os.path.join(output_dir, f"face_{saved_count:04d}.jpg")
                            cv2.imwrite(output_path, face_img)
                            saved_count += 1

                            if saved_count >= frames_per_video:
                                break
                except Exception as e:
                    print(f"Error processing frame {frame_num}: {str(e)}")
                    continue

        cap.release()
        success = saved_count > 0
        return success, processed_frames, saved_count

# ----------------------------------------------------
# 2. Model Definitions
# ----------------------------------------------------

# Modified FaceClassifierLSTM with backbone option
class FaceClassifierLSTM(nn.Module):
    def __init__(self, num_classes=2, latent_dim=None, lstm_layers=1,
                 hidden_dim=256, bidirectional=False, backbone='effb0'):
        super(FaceClassifierLSTM, self).__init__()
        self.backbone = backbone
        if backbone == 'effb0':
            cnn_model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
            self.cnn = nn.Sequential(cnn_model.features)
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            latent_dim = 1280
        elif backbone == 'resnext':
            cnn_model = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.DEFAULT)
            self.cnn = nn.Sequential(*list(cnn_model.children())[:-2])
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            latent_dim = 2048
        else:
            raise ValueError("Unknown backbone type")

        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers,
                            bidirectional=bidirectional, batch_first=True)
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        self.norm = nn.LayerNorm(lstm_output_dim)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.4)
        self.linear = nn.Linear(lstm_output_dim, num_classes)

    def forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(1)  # (batch_size, 1, channels, height, width)
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        features = self.cnn(x)
        pooled = self.avgpool(features)
        pooled = pooled.view(batch_size, seq_length, -1)
        lstm_out, _ = self.lstm(pooled)
        aggregated = torch.mean(lstm_out, dim=1)
        output = self.linear(self.dropout(self.relu(self.norm(aggregated))))
        return output

# The Swin model remains similar to before.
class FaceClassifierSwin(nn.Module):
    def __init__(self, num_classes=2, model_name='swin_tiny_patch4_window7_224',
                 load_pretrained=True, device='cuda'):
        super().__init__()
        self.device = device

        # Use a smaller Swin variant
        self.backbone = timm.create_model(
            model_name,
            pretrained=load_pretrained,
            features_only=True
        ).to(device)
        self.num_features = self.backbone.feature_info[-1]['num_chs']  # For swin_tiny
        self.dropout = nn.Dropout(p=0.5)
        self.norm = nn.LayerNorm(self.num_features)

        self.classifier = nn.Linear(self.num_features, num_classes).to(device)

    def forward(self, x):
        # AMP dtype casting
        with torch.amp.autocast(device_type=self.device, dtype=torch.float16):
            if x.dim() == 4:
                x = x.unsqueeze(1)

            batch_size, seq_length, c, h, w = x.shape
            x = x.view(batch_size * seq_length, c, h, w).to(self.device)

            # Forward pass
            features = self.backbone(x)[-1]  # Last feature map
            features = features.permute(0, 3, 1, 2)  # -> [N, C, H, W]

            # Pooling and aggregation and Layer Norm
            features = features.mean(dim=[2, 3])  # [N, C]
            features = features.view(batch_size, seq_length, -1)
            aggregated = features.mean(dim=1)
            aggregated = self.dropout(aggregated)
            aggregated = self.norm(aggregated)

        # Classifier in FP32 for stability
        with torch.amp.autocast(device_type=self.device, enabled=False):
            return self.classifier(aggregated.to(torch.float32))


# ----------------------------------------------------
# 3. Inference Pipeline: Process input and ensemble predictions
# ----------------------------------------------------
def process_input(video_file, folder_zip, image_file, frames_to_extract, face_conf_threshold):
    start_time = time.time()

    infer_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    provided_inputs = {
        "folder_zip": folder_zip is not None,
        "image_file": image_file is not None,
        "video_file": video_file is not None,
    }
    input_count = sum(provided_inputs.values())
    if input_count > 1:
        warning_msg = ("Warning: Multiple inputs detected. Provided inputs: " +
                       ", ".join([k for k, v in provided_inputs.items() if v]) +
                       ". Proceeding with the highest priority input: folder ZIP > image > video.")
        print(warning_msg)
        user_warning = warning_msg
    else:
        user_warning = ""

    # --- Process folder-of-frames if provided ---
    if folder_zip is not None:
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "frames.zip")
            if hasattr(folder_zip, "read"):
                zip_bytes = folder_zip.read()
            else:
                with open(folder_zip, "rb") as f:
                    zip_bytes = f.read()
            with open(zip_path, "wb") as f:
                f.write(zip_bytes)
            extract_dir = os.path.join(tmpdir, "extracted")
            os.makedirs(extract_dir, exist_ok=True)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            # Get image files
            face_files = sorted([
                os.path.join(root, file)
                for root, _, files in os.walk(extract_dir)
                for file in files if file.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])
            if len(face_files) > frames_to_extract:
                face_files = face_files[:int(frames_to_extract)]
            processed_frames = len(face_files)
            faces_extracted = processed_frames

            frames = []
            annotated_frames = []
            for file in face_files:
                img = Image.open(file).convert("RGB")
                frames.append(infer_transform(img))
                annotated_frames.append(img.resize((256, 256)))
    # --- Process single image if provided and no folder ---
    elif image_file is not None:
        if isinstance(image_file, np.ndarray):
            img = Image.fromarray(image_file.astype('uint8'), 'RGB')
        elif hasattr(image_file, "read"):
            img = Image.open(image_file).convert("RGB")
        else:
            img = Image.open(image_file).convert("RGB")
        frames = [infer_transform(img)]
        annotated_frames = [img.resize((256, 256))]
        processed_frames = 1
        faces_extracted = 1
    # --- Process video if neither folder nor image are provided ---
    elif video_file is not None:
        video_path = video_file.name if hasattr(video_file, "name") else video_file
        with tempfile.TemporaryDirectory() as tmpdir:
            face_dir = os.path.join(tmpdir, 'faces')
            extractor = FaceExtractor(conf_threshold=face_conf_threshold)
            success, processed_frames, faces_extracted = extractor.extract_faces(
                video_path, face_dir, frames_per_video=int(frames_to_extract))
            if not success:
                proc_details = (f"Processed {processed_frames} frames.\n"
                                "No faces extracted. Please try a different video or adjust the threshold.")
                return {"error": "No faces extracted."}, None, proc_details
            face_files = sorted([
                os.path.join(face_dir, f) for f in os.listdir(face_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])
            frames = []
            annotated_frames = []
            for file in face_files:
                img = Image.open(file).convert("RGB")
                frames.append(infer_transform(img))
                annotated_frames.append(img.resize((256, 256)))
    else:
        return {"error": "No input provided."}, None, "Please upload a video, an image, or a folder (as a zip file) of frames."

    if len(frames) == 0:
        proc_details = f"Processed {processed_frames} frames.\nNo valid face images found."
        return {"error": "No valid face images found."}, None, proc_details

    # Create tensor of shape (1, T, C, H, W)
    face_sequence = torch.stack(frames, dim=0).unsqueeze(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---------------------------
    # Instantiate and load the 3 models
    # ---------------------------

    # 1. EfficientNet-B0 + LSTM
    model_effb0 = FaceClassifierLSTM(num_classes=2, hidden_dim=256, backbone='effb0').to(device)
    model_effb0_path = 'model_weights/final_face_classifier_effb0.pt'
    if not os.path.exists(model_effb0_path):
        return {"error": f"Model file not found at {model_effb0_path}"}, None, "Model file missing."
    state_dict = torch.load(model_effb0_path, map_location=device)
    if any(k.startswith('module.') for k in state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k.replace('module.', '')] = v
        state_dict = new_state_dict
    model_effb0.load_state_dict(state_dict, strict=True)
    model_effb0.eval()

    # 2. ResNeXt + LSTM
    model_resnext = FaceClassifierLSTM(num_classes=2, hidden_dim=256, backbone='resnext').to(device)
    model_resnext_path = 'model_weights/final_face_classifier_resnext.pt'
    if not os.path.exists(model_resnext_path):
        return {"error": f"Model file not found at {model_resnext_path}"}, None, "Model file missing."
    state_dict = torch.load(model_resnext_path, map_location=device)
    if any(k.startswith('module.') for k in state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k.replace('module.', '')] = v
        state_dict = new_state_dict
    model_resnext.load_state_dict(state_dict, strict=True)
    model_resnext.eval()

    # 3. Swin Transformer
    model_swin = FaceClassifierSwin(model_name='swin_tiny_patch4_window7_224', load_pretrained=False).to(device)
    model_swin_path = 'model_weights/final_face_classifier_swin.pt'
    if not os.path.exists(model_swin_path):
        return {"error": f"Model file not found at {model_swin_path}"}, None, "Model file missing."
    state_dict = torch.load(model_swin_path, map_location=device)
    if any(k.startswith('module.') for k in state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k.replace('module.', '')] = v
        state_dict = new_state_dict
    model_swin.load_state_dict(state_dict, strict=True)
    model_swin.eval()


    # ---------------------------
    # Run inference with all 3 models
    # ---------------------------
    with torch.no_grad():
        output_effb0 = model_effb0(face_sequence.to(device))
        output_resnext = model_resnext(face_sequence.to(device))
        output_swin = model_swin(face_sequence.to(device))

    # print(output_effb0,output_resnext,output_swin)
    prob_effb0 = torch.softmax(output_effb0, dim=1).cpu().numpy()[0]
    prob_resnext = torch.softmax(output_resnext, dim=1).cpu().numpy()[0]
    prob_swin = torch.softmax(output_swin, dim=1).cpu().numpy()[0]

    # Determine prediction (Real/Fake)
    pred_effb0 = 'Real' if prob_effb0[0] >= prob_effb0[1] else 'Fake'
    pred_resnext = 'Real' if prob_resnext[0] >= prob_resnext[1] else 'Fake'
    pred_swin = 'Real' if prob_swin[0] >= prob_swin[1] else 'Fake'

    # Majority vote for ensemble decision.
    # real_votes = sum([pred_effb0 == "Real", pred_resnext == "Real", pred_swin == "Real"])
    # fake_votes = 3 - real_votes
    # ensemble_prob_real = real_votes / 3.0
    # ensemble_prob_fake = fake_votes / 3.0




    # Average the probabilities from all three models:
    ensemble_prob_real = (prob_effb0[0] + prob_resnext[0] + prob_swin[0]) / 3.0
    ensemble_prob_fake = (prob_effb0[1] + prob_resnext[1] + prob_swin[1]) / 3.0

    ensemble_label = "Real" if ensemble_prob_real >= ensemble_prob_fake else "Fake"

    label_dict = {
        "Real": float(ensemble_prob_real),
        "Fake": float(ensemble_prob_fake)
    }


    # Prepare a dictionary with model probabilities and ensemble prediction.
    details_text = (
            f"EfficientNet-B0+LSTM:\n"
            f"  Real: {prob_effb0[0] * 100:.1f}% | Fake: {prob_effb0[1] * 100:.1f}% | Prediction: {pred_effb0}\n\n"
            f"ResNeXt+LSTM:\n"
            f"  Real: {prob_resnext[0] * 100:.1f}% | Fake: {prob_resnext[1] * 100:.1f}% | Prediction: {pred_resnext}\n\n"
            f"Swin Transformer:\n"
            f"  Real: {prob_swin[0] * 100:.1f}% | Fake: {prob_swin[1] * 100:.1f}% | Prediction: {pred_swin}\n\n"
            f"-----------\n"
            f"Final Ensemble Probabilities:\n"
            f"  Real: {ensemble_prob_real * 100:.1f}% | Fake: {ensemble_prob_fake * 100:.1f}%\n"
            f"Ensemble Decision: {ensemble_label}"
        )

    # Annotate the gallery images with all model results.
    annotated_gallery = []
    for img in annotated_frames:
        draw = ImageDraw.Draw(img)
        draw.text(
            (10, 10),
            f"EffB0: {pred_effb0}\nRNXt: {pred_resnext}\nSwin: {pred_swin}\nEns: {ensemble_label}",
            fill="red"
        )
        annotated_gallery.append(np.array(img))

    elapsed_time = time.time() - start_time
    proc_details = (f"{user_warning}\nTotal frames processed: {processed_frames}\n"
                    f"Faces processed: {faces_extracted}\nInference time: {elapsed_time:.2f} seconds")

    # Optionally clear GPU cache and run garbage collection
    torch.cuda.empty_cache()
    gc.collect()

    # return annotated_gallery, prediction, proc_details
    return annotated_gallery, label_dict, f"{details_text}\n\n{proc_details}"

# ----------------------------------------------------
# 4. Gradio Interface with Three Input Options
# ----------------------------------------------------
if __name__ == "__main__":
    interface = gr.Interface(
        fn=process_input,
        flagging_mode="never",
        inputs=[
            gr.Video(label="Input Video (optional)"),
            gr.File(label="Upload Folder of Frames (zip file, optional)"),
            gr.Image(label="Input Image (optional)"),
            gr.Number(label="Number of Frames to Process", value=10, precision=0),
            gr.Slider(minimum=0.5, maximum=1.0, step=0.05, value=0.9,
                      label="Face Detection Confidence Threshold")
        ],
        outputs=[
            gr.Gallery(label="Processed Faces (Annotated)", columns=5),
            gr.Label(label="Prediction"),
            gr.Textbox(label="Processing Details", lines=5)
        ],
        title="Deepfake Detection System",
        description=(
            "Upload a video file, a zip file of a folder containing frames, or a single image. "
            "If multiple inputs are provided, the system will prioritize the folder of frames, then "
            "the image, and finally the video. The system extracts faces and classifies whether the input is real or fake "
            "using three different models with majority vote ensembling."
        )
    )
    interface.launch()
