import os
import time
import argparse
from pathlib import Path
from PIL import Image
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from torchvision import transforms, models, datasets
from torchvision.transforms import RandomErasing

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from sklearn.metrics import confusion_matrix, classification_report
from torch.amp import autocast, GradScaler


import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
scaler = GradScaler()

print("CUDA available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())
print("GPU Names:")
for i in range(torch.cuda.device_count()):
    print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")

print(f"Using device: ",device)
print("======================================================================")
# device = torch.device(device)

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
                    # frame_paths = sorted(video_folder.glob("*.jpg","*.png"))
                    jpg_paths = list(video_folder.glob("*.jpg"))
                    png_paths = list(video_folder.glob("*.png"))
                    frame_paths = sorted(jpg_paths + png_paths)
                    if len(frame_paths) >= num_frames:
                        # Compute evenly spaced indices over the entire range of frames
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
        # Create a tensor of shape (num_frames, C, H, W)
        frames = torch.stack(frames)
        return frames, label


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

        features = self.cnn(x)  # Shape: (batch_size*seq_length, C_out, H', W')


        pooled = self.avgpool(features)  # Shape: (batch_size*seq_length, C_out, 1, 1)
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

# Utility Functions
# Maintains the running average of a metric (e.g. loss, accuracy) for monitoring training performance over batches/epochs
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
        # value: latest observed value
        self.val = val
        self.sum += val * batch_size
        self.count += batch_size
        self.avg = self.sum / self.count


def calculate_accuracy(outputs, targets):
    _, preds = torch.max(outputs, 1)
    # print("Prediction for outputs:",preds)
    correct = (preds == targets).sum().item()
    return 100 * correct / targets.size(0)



def evaluate_and_visualize(train_loss, val_loss, train_acc, val_acc,save_dir):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(train_loss) + 1)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Plot Loss
    axs[0].plot(epochs, train_loss, 'g', label='Training Loss')
    axs[0].plot(epochs, val_loss, 'b', label='Validation Loss')
    axs[0].set_title('Loss Over Epochs')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))

    # Plot Accuracy
    axs[1].plot(epochs, train_acc, 'g', label='Training Accuracy')
    axs[1].plot(epochs, val_acc, 'b', label='Validation Accuracy')
    axs[1].set_title('Accuracy Over Epochs')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy (%)')
    axs[1].legend()
    axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()



def generate_classification_report(y_true, y_pred, save_dir, classes=["Real", "Fake"]):
    os.makedirs(save_dir, exist_ok=True)

    # Get the classification report as text
    report = classification_report(y_true, y_pred, target_names=classes, digits=4)
    print("Classification Report:\n", report)

    # Save the classification report as an image
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis('off')  # turn off the axis
    ax.set_position([0, 0, 1, 1])

    # Add the report text in a monospaced font
    ax.text(0.5, 0.5, report,
            fontsize=12,
            fontfamily='monospace',
            va='center',  # align text to the top
            ha='center',
            wrap=True,
            transform=ax.transAxes)
    report_path = os.path.join(save_dir, f'classification_report.png')
    plt.savefig(report_path, bbox_inches='tight')
    plt.close()
    print(f"Classification report saved to {report_path}")

    # Generate and save the confusion matrix as before
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    cm_path = os.path.join(save_dir, f'CM.png')
    plt.savefig(cm_path, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")



def print_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
    accuracy = np.trace(cm) / np.sum(cm) * 100
    print("Accuracy: {:.2f}%".format(accuracy))




def train_epoch(epoch, num_epochs, data_loader, model, criterion, optimizer):
    model.train()
    losses = RunningAverage()
    accuracies = RunningAverage()
    for i, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        with autocast(device_type=device):
            outputs = model(inputs)
            # print("Target:",targets)
            # print("Output:",outputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


        # outputs = model(inputs)
        # loss = criterion(outputs, targets)
        # loss.backward()
        # optimizer.step()

        acc = calculate_accuracy(outputs, targets)
        losses.update(loss.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))
        print( f"\rEpoch [{epoch + 1}/{num_epochs}] | "
            f"Batch [{i + 1}/{len(data_loader)}] | "
            f"Train Loss: {losses.avg:.6f} | "
            f"Train Accuracy: {accuracies.avg:.6f}%",
            end="")
    print("\n")
    return losses.avg, accuracies.avg

def evaluate(data_loader, model, criterion):
    model.eval()
    losses = RunningAverage()
    accuracies = RunningAverage()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for i,(inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            with autocast(device_type=device):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            acc = calculate_accuracy(outputs, targets)
            losses.update(loss.item(), inputs.size(0))
            accuracies.update(acc, inputs.size(0))
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

            print(
                f"\r[Batch {i + 1}/{len(data_loader)}] Evaluating... "
                f"[Loss: {losses.avg:.6f}, Accuracy: {accuracies.avg:.6f}%]",
                end=""
            )
    print("\n")
    return losses.avg, accuracies.avg, all_preds, all_labels


class EarlyStoppingWithCheckpoint:
    def __init__(self, patience=5, min_delta=0, path_best='checkpoint.pt'):
        self.patience = patience
        self.min_delta = min_delta
        self.path_best = path_best

        self.best_loss=float('inf')
        self.counter = 0
        self.early_stop = False

    # make it callable like a function
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.path_best)  # checkpoint
        else:
            self.counter += 1
            print(f"No improvement. Early stopping....{self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                model.load_state_dict(torch.load(self.path_best))



def main(batch_size,num_epochs,patience,num_frames,dataset_root,test2_dataset_root,model_choice):

    img_dim=224   # Change acc to model usage
    # ResNeXt50,EfficientNetB0,swin_tiny_patch4_window7_224  all use 224x224
    print("======================================================================")
    print("Sanity check for BS:3 => Expected [(Batch Size,num_classes] [3,2]")
    if model_choice in ["efb0", "resnext"]:
        model = FaceClassifierLSTM(num_classes=2, hidden_dim=256,cnn_type=model_choice).to(device)
        dummy_seq_input = torch.randn(3, 5, 3, 224, 224).to(device)
        output_seq = model(dummy_seq_input).to(device)
        print(f"Output shape for frame-sequence of inputs for {model_choice} :", output_seq.shape)

        # Test with a single-frame input (without the sequence dimension)
        dummy_single_input = torch.randn(3, 3, 224, 224).to(device)
        output_single = model(dummy_single_input).to(device)
        print(f"Output shape for single-frame  input for {model_choice} :", output_single.shape)


    elif model_choice == "swin":
        model = FaceClassifierSwin(model_name='swin_tiny_patch4_window7_224',device=device).to(device)
        dummy_input = torch.randn(2, 5, 3, 224, 224).to(device)
        output = model(dummy_input)
        print(f"Output shape for sequence of input for {model_choice} :", output.shape)


    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)



# Handle class imbalance
# For  (real = 1362, fake = 3997):
#     Total Ntotal=1362+3997=5359 and Nclasses=2
#     For real: weightreal=53592×1362≈1.966
#     For fake: weightfake=53592×3997≈0.67
    weights = torch.tensor([1.966, 0.67], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Calculate the total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print("Total Number Of Params: {:.3f}M".format(total_params / 1e6))
    print("======================================================================")
    # Data augmentation for the training set
    train_transform = transforms.Compose([
        transforms.Resize((img_dim, img_dim)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
    ])

    transform = transforms.Compose([
        transforms.Resize((img_dim, img_dim)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load datasets using CustomImageFolder class
    print("Creating training dataset")
    train_dataset = CustomImageFolder(root=f"{dataset_root}/train", transform=train_transform,num_frames=num_frames)
    print("Creating val dataset")
    val_dataset = CustomImageFolder(root=f"{dataset_root}/val", transform=transform,num_frames=num_frames)
    print("Creating test dataset")
    test_dataset = CustomImageFolder(root=f"{dataset_root}/test", transform=transform,num_frames=num_frames)

    # Data Loaders
    print("Creating train dataloaders")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    print("Creating val dataloaders")
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    print("Creating test dataloaders")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    print("======================================================================")
    images, labels = next(iter(train_loader))
    print("Train images shape:", images.shape)
    print("Train labels shape:", labels.shape)

    # for batch_idx, (images, labels) in enumerate(train_loader):
    #     print(f"Batch {batch_idx}:")
    #     print("Images shape:", images.shape)
    #     print("Labels shape:", labels.shape)
    #     print("Lable: ",labels)
    print("======================================================================")



    # Verify dataset loading
    print("Training set size:", len(train_dataset))
    print("Validation set size:", len(val_dataset))
    print("Test set size:", len(test_dataset))
    print("======================================================================")
    print("Training loader size:", len(train_loader))
    print("Validation loader size:", len(val_loader))
    print("Test loader size:", len(test_loader))
    print("======================================================================")

    checkpoint = "checkpoint_model"
    save_dir = "results/"
    path_best = f"results/{checkpoint}.pt"
    os.makedirs(save_dir, exist_ok=True)



    # Training Loop
    start_time = time.time()
    print("Training started...")
    num_epochs = num_epochs
    patience=patience
    train_loss_avg, val_loss_avg = [], []
    train_accuracy, val_accuracy = [], []
    early_stopping = EarlyStoppingWithCheckpoint(patience=patience, path_best=path_best)

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(epoch, num_epochs, train_loader, model, criterion, optimizer)
        scheduler.step()
        train_loss_avg.append(train_loss)
        train_accuracy.append(train_acc)
        val_loss, val_acc, _, _ = evaluate(val_loader, model, criterion)
        val_loss_avg.append(val_loss)
        val_accuracy.append(val_acc)
        early_stopping(val_loss, model)
        print("\nEpoch %d/%d, Train Loss: %.4f, Train Acc: %.2f%%, Val Loss: %.4f, Val Acc: %.2f%%" %
            (epoch+1, num_epochs, train_loss, train_acc, val_loss, val_acc))
        if early_stopping.early_stop:
            print(f"Early Stopping After {patience} Epochs did not imprve")
            break

    # Save model
    model_save_path = os.path.join(save_dir, f'final_face_classifier.pt')
    torch.save(model.state_dict(), model_save_path)

# Plot loss and accuracy curves
    evaluate_and_visualize(
        train_loss=train_loss_avg,
        val_loss=val_loss_avg,
        train_acc=train_accuracy,
        val_acc=val_accuracy,
        save_dir=save_dir
    )
    print(f"Training curves saved to {os.path.join(save_dir, 'training_curves.png')}")



    end_time = time.time()
    print(f"Training finished! Total Time: {(end_time - start_time)/60:.3f} min")

    print("======================================================================")

    print("Evaluating on Test Set")
    _, _, test_preds, test_labels = evaluate(test_loader, model, criterion)
    # print_confusion_matrix(test_labels, test_preds)
    generate_classification_report(test_labels, test_preds,save_dir)


    print("======================================================================")
    print("Creating New Test dataloader for another test dataset........")
    test_dataset2 = CustomImageFolder(root=test2_dataset_root, transform=transform,num_frames=num_frames)
    print("Creating test dataloaders")
    test_loader = DataLoader(test_dataset2, batch_size=batch_size, shuffle=False, num_workers=0)
    print("Evaluation:")
    _, _, test_preds, test_labels = evaluate(test_loader, model, criterion)
    print_confusion_matrix(test_labels, test_preds)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deepfake Detection Training Script")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--num_frames", type=int, default=10, help="Number of frames per video to use")
    parser.add_argument("--model", type=str, default="resnext",
                        choices=["efb0", "resnext", "swin"],
                        help="Model architecture to use: 'efb0' or 'resnext' (LSTM-based) or 'swin' for Swin VTransformer")
    args = parser.parse_args()



    dataset_root="Dataset/ExtractedFaces/ND"
    test2_dataset_root="Dataset/ExtractedFaces/ND/test"

    # dataset_root="/media/khushwant/Local Disk E/Deepfake datasets/UADFV"
    # test2_dataset_root="/media/khushwant/Local Disk E/Deepfake datasets/CelebDF/test"



    print(f"Batch Size: {args.batch_size}")
    print(f"Number of Epochs: {args.epochs}")
    print(f"Patience: {args.patience}")
    print(f"Number of Frames: {args.num_frames}")
    print(f"Selected Model: {args.model}")
    main(args.batch_size, args.epochs, args.patience, args.num_frames,
dataset_root, test2_dataset_root, args.model)
