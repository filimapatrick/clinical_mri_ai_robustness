import os
import sys
sys.path.insert(0, os.path.abspath("."))

import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import monai
    from monai.networks.nets import DenseNet121, resnet18
    HAS_MONAI = True
except ImportError:
    HAS_MONAI = False

from src.data.loader import SubjectMRIDataset, get_stratified_subject_splits

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate 3D CNN architectures (ResNet-18, DenseNet-121).")
    parser.add_argument("--catalog", type=str, default="./data/dataset_catalog.csv", help="Catalog CSV.")
    parser.add_argument("--processed_dir", type=str, default="./data/processed", help="Processed scans dir.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Config YAML.")
    parser.add_argument("--model", type=str, default="densenet121", choices=["resnet18", "densenet121"], help="Model architecture.")
    parser.add_argument("--task", type=str, default="task4", help="Task name (task1..task4).")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--out_dir", type=str, default="./checkpoints/densenet", help="Output directory for checkpoints.")
    return parser.parse_args()

def build_3d_cnn_model(model_name="densenet121", in_channels=3, num_classes=3, spatial_dims=3):
    if not HAS_MONAI:
        raise ImportError("MONAI is required for 3D CNN models. Please install monai.")

    if model_name == "densenet121":
        model = DenseNet121(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=num_classes,
            dropout_prob=0.2
        )
    elif model_name == "resnet18":
        model = resnet18(
            spatial_dims=spatial_dims,
            n_input_channels=in_channels,
            num_classes=num_classes,
            feed_forward=True
        )
    else:
        raise ValueError(f"Unsupported model {model_name}")

    return model

def train_cnn_model(args):
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using compute device: {device}")

    # Generate Stratified Subject Splits
    splits = get_stratified_subject_splits(args.catalog, config_path=args.config)
    
    # Save splits for reproduction
    os.makedirs(args.out_dir, exist_ok=True)
    splits['train'].to_csv(os.path.join(args.out_dir, "train_split.csv"), index=False)
    splits['val'].to_csv(os.path.join(args.out_dir, "val_split.csv"), index=False)
    splits['test'].to_csv(os.path.join(args.out_dir, "test_split.csv"), index=False)

    train_dataset = SubjectMRIDataset(args.catalog, args.processed_dir, task_name=args.task, config_path=args.config)
    val_dataset = SubjectMRIDataset(args.catalog, args.processed_dir, task_name=args.task, config_path=args.config)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    task_cfg = cfg['tasks'][args.task]
    num_classes = len(task_cfg['classes'])
    in_channels = len(cfg['modalities'])

    model = build_3d_cnn_model(model_name=args.model, in_channels=in_channels, num_classes=num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float('inf')
    best_model_path = os.path.join(args.out_dir, f"{args.model}_{args.task}_best.pth")

    print(f"🚀 Starting Training [{args.model.upper()}] for {args.epochs} epochs on {args.task}...")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            images = batch['image'].to(device) # (B, C, D, H, W)
            labels = batch['label'].to(device) # (B,)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        scheduler.step()

        epoch_train_loss = train_loss / max(1, total)
        epoch_train_acc = correct / max(1, total)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / max(1, val_total)
        epoch_val_acc = val_correct / max(1, val_total)

        if epoch % 5 == 0 or epoch == args.epochs:
            print(f"Epoch [{epoch:02d}/{args.epochs:02d}] "
                  f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | "
                  f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': epoch_val_loss,
                'val_acc': epoch_val_acc,
                'task': args.task,
                'model_name': args.model
            }, best_model_path)

    print(f"✅ Training completed. Best model checkpoint saved to: {best_model_path}")

if __name__ == "__main__":
    args = parse_args()
    train_cnn_model(args)
