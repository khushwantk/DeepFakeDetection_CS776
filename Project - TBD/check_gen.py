import os
import argparse
from pathlib import Path
from PIL import Image
from collections import OrderedDict

import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report

# Set device and random seeds for reproducibility
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# RunningAverage helper class (for loss, accuracy, etc.)
class RunningAverage:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, batch_size=1):
        self.val = val
        self.sum += val * batch_size
        self.count += batch_size
        self.avg = self.sum / self.count

# Custom dataset to handle nested directories
class CustomImageFolder(Dataset):
    def __init__(self, root, transform=None, num_frames=80):
        self.root = Path(root)
        self.transform = transform
        self.samples = []
        self.class_to_idx = {"real": 0, "fake": 1}

        # For each video folder (one per video)
        for class_name in ["real", "fake"]:
            class_path = self.root / class_name
            for video_folder in class_path.iterdir():
                if video_folder.is_dir():
                    jpg_paths = list(video_folder.glob("*.jpg"))
                    png_paths = list(video_folder.glob("*.png"))
                    frame_paths = sorted(jpg_paths + png_paths)
                    if len(frame_paths) >= num_frames:
                        indices = np.linspace(0, len(frame_paths) - 1, num_frames, dtype=int)
                        selected_frames = [frame_paths[i] for i in indices]
                        self.samples.append((selected_frames, self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        frame_paths, label = self.samples[index]
        frames = []
        for p in frame_paths:
            image = Image.open(p).convert("RGB")
            if self.transform:
                image = self.transform(image)
            frames.append(image)
        frames = torch.stack(frames)  # Shape: (num_frames, C, H, W)
        return frames, label

# Model Architectures
class FaceClassifierLSTM(nn.Module):
    def __init__(self, num_classes=2, cnn_type="efb0",latent_dim=2048, lstm_layers=1,
                 hidden_dim=2048, bidirectional=False):

        super(FaceClassifierLSTM, self).__init__()

        if cnn_type.lower() == "resnext":
            # Use ResNeXt-50
            from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights
            cnn_model = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.DEFAULT)
            self.cnn = nn.Sequential(*list(cnn_model.children())[:-2])
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            latent_dim = 2048
        elif cnn_type.lower() == "efb0":
            # Use EfficientNet-B0
            from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
            cnn_model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
            self.cnn = nn.Sequential(cnn_model.features)
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            latent_dim = 1280
        else:
            raise ValueError("cnn_type must be either 'efb0' or 'resnext'")

        # LSTM
        # batch_first=True allows input shape (batch, seq, features)
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers,
                            bidirectional=bidirectional, batch_first=True)

        # Adjust output dimension in case LSTM is bidirectional
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)

        self.norm = nn.LayerNorm(lstm_output_dim)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.4)
        self.linear = nn.Linear(lstm_output_dim, num_classes)


    def forward(self, x):
        # print("Input shape:", x.shape)

        # input is a 4D tensor, so we add a singleton sequence dimension
        if x.dim() == 4:
            x = x.unsqueeze(1)
            # Now shape becomes (batch_size, 1, channels, height, width)
            # print("After unsqueeze:", x.shape)

        # Unpack shape
        batch_size, seq_length, c, h, w = x.shape

        # Process each frame using the ResNeXt50/EFNext FE
        x = x.view(batch_size * seq_length, c, h, w)

        features = self.cnn(x)  # Shape: (batch_size*seq_length, C, H', W')


        pooled = self.avgpool(features)  # Shape: (batch_size*seq_length, C, 1, 1)
        pooled = pooled.view(batch_size, seq_length, -1)  # Shape: (batch_size, seq_length, latent_dim)

        # Pass the sequence of features to the LSTM
        lstm_out, _ = self.lstm(pooled)  # Shape: (batch_size, seq_length, hidden_dim)

        # Aggregate the LSTM outputs (avg pooling)
        aggregated = torch.mean(lstm_out, dim=1)  # Shape: (batch_size, hidden_dim)
        output = self.linear(self.dropout(self.relu(self.norm(aggregated))))


        return output



class FaceClassifierSwin(nn.Module):
    def __init__(self, num_classes=2, model_name='swin_tiny_patch4_window7_224',
                 load_pretrained=True, device='cuda'):
        super().__init__()
        self.device = device

        # Use smaller Swin variant
        self.backbone = timm.create_model(
            model_name,
            pretrained=load_pretrained,
            features_only=True
        ).to(device)
        self.num_features = self.backbone.feature_info[-1]['num_chs']
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
            features = features.permute(0, 3, 1, 2).contiguous()  # -> [N, C, H, W]

            # Pooling aggregation and Layer Norm
            features = features.mean(dim=[2, 3])  # [N, C]
            features = features.view(batch_size, seq_length, -1)
            aggregated = features.mean(dim=1)
            aggregated = self.dropout(aggregated)
            aggregated = self.norm(aggregated)

        # Classifier in FP32 for stability
        with torch.amp.autocast(device_type=self.device, enabled=False):
            return self.classifier(aggregated.to(torch.float32))

# Evaluation Utility Functions using RunningAverage

def calculate_accuracy(outputs, targets):
    _, preds = torch.max(outputs, 1)
    correct = (preds == targets).sum().item()
    return 100 * correct / targets.size(0)

def evaluate(data_loader, model, criterion):
    model.eval()
    loss_avg = RunningAverage()
    accuracy_avg = RunningAverage()
    all_preds = []
    all_labels = []
    total_batches = len(data_loader)

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            acc = calculate_accuracy(outputs, targets)

            # Update RunningAverage for loss and accuracy
            batch_size = inputs.size(0)
            loss_avg.update(loss.item(), batch_size)
            accuracy_avg.update(acc, batch_size)

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

            # Print marker with running averages
            print(f"\rEvaluation batch {i+1}/{total_batches} | "
                  f"Running Loss: {loss_avg.avg:.4f} | "
                f"Running Accuracy: {accuracy_avg.avg:.2f}%", end='', flush=True)

    return loss_avg.avg, accuracy_avg.avg, all_preds, all_labels

def print_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
    accuracy = np.trace(cm) / np.sum(cm) * 100
    print("Overall Accuracy: {:.2f}%".format(accuracy))

def generate_classification_report(y_true, y_pred, save_dir, classes=["Real", "Fake"]):
    os.makedirs(save_dir, exist_ok=True)
    report = classification_report(y_true, y_pred, target_names=classes, digits=4)
    print("Classification Report:\n", report)
    # Save the report as an image
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis('off')
    ax.text(0.5, 0.5, report, fontsize=12, fontfamily='monospace',
            va='center', ha='center', wrap=True)
    report_path = os.path.join(save_dir, 'classification_report.png')
    plt.savefig(report_path, bbox_inches='tight')
    plt.close()
    print(f"Classification report saved to {report_path}")

    # Generate and save the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    cm_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")

# Main function for testing only
def main(batch_size, num_frames, model_choice, model_weights, test_data):
    # Use a fixed image dimension (224x224) for all models
    img_dim = 224
    transform = transforms.Compose([
        transforms.Resize((img_dim, img_dim)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Create test dataset and dataloader using the provided test_data directory
    print("Creating test dataset...")
    test_dataset = CustomImageFolder(root=test_data, transform=transform, num_frames=num_frames)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"Test dataset size: {len(test_dataset)} samples")
    print(f"Test dataloader batches: {len(test_loader)}")

    # Initialize the model based on the selected architecture
    if model_choice in ["efb0", "resnext"]:
        model = FaceClassifierLSTM(num_classes=2, hidden_dim=256, cnn_type=model_choice).to(device)
    elif model_choice == "swin":
        model = FaceClassifierSwin(model_name='swin_tiny_patch4_window7_224').to(device)
    else:
        raise ValueError("Invalid model choice. Use 'efb0', 'resnext', or 'swin'.")

    # Load the model weights
    if not os.path.exists(model_weights):
        print(f"Error: Model file not found at {model_weights}")
        return
    state_dict = torch.load(model_weights, map_location=device)
    # Remove "module." prefix if it exists
    if any(k.startswith('module.') for k in state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k.replace('module.', '')] = v
        state_dict = new_state_dict
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print("Model loaded and set to evaluation mode.")

    # Evaluate on the test dataset
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, test_preds, test_labels = evaluate(test_loader, model, criterion)
    print("============================================================")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print("============================================================")
    print_confusion_matrix(test_labels, test_preds)
    generate_classification_report(test_labels, test_preds, save_dir="results")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deepfake Detection Testing Script")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for testing")
    parser.add_argument("--num_frames", type=int, default=30, help="Number of frames per video to use")
    parser.add_argument("--model", type=str, default="efb0",
                        choices=["efb0", "resnext", "swin"],
                        help="Model architecture to use: 'efb0', 'resnext', or 'swin'")
    parser.add_argument("--model_weights", type=str, default="model_weights/final_face_classifier_effb0.pt",
                        help="Path to the trained model weights file")
    parser.add_argument("--test_data", type=str,
                        default="/media/khushwant/Local Disk E/Deepfake datasets/CelebDF/test",
                        # /media/khushwant/Local Disk E/Deepfake datasets/CelebDF/test
                        # /media/khushwant/Local Disk E/Deepfake datasets/UADFV/train
                        help="Path to test dataset")
    args = parser.parse_args()

    print(f"Batch Size: {args.batch_size}")
    print(f"Number of Frames: {args.num_frames}")
    print(f"Selected Model: {args.model}")
    print(f"Model Weights Path: {args.model_weights}")
    print(f"Test Dataset Path: {args.test_data}")



    main(args.batch_size, args.num_frames, args.model, args.model_weights, args.test_data)
