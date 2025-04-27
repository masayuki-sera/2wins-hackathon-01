# ========== ライブラリインポート ==========
import os
import glob
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

# ========== ラベル定義 ==========
LABELS = ["玄関", "バルコニー", "浴室", "トイレ", "収納", "洋室", "クローゼット", "廊下", "ホール", "和室"]
label_to_idx = {label: i for i, label in enumerate(LABELS)}
idx_to_label = {i: label for label, i in label_to_idx.items()}

# ========== ユーティリティ関数 ==========
def set_seed(seed=42):
    """乱数シードを固定して再現性を確保"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ========== EarlyStopping クラス ==========
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, path="checkpoints/efficientnet_b0_best.pth"):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.verbose = verbose
        self.path = path

    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.path)
            if self.verbose:
                print(f"Validation loss improved. Saving model to {self.path}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

# ========== データセットクラス ==========
class ImageLabelDataset(Dataset):
    """画像とラベルを扱うDatasetクラス"""
    def __init__(self, image_dir, transform=None, with_label=True):
        self.image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.with_label = with_label

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert('L')
        img = self.transform(img)

        if self.with_label:
            filename = os.path.basename(path)
            label_name = filename.split("_")[0]
            label = label_to_idx[label_name]
            return img, label
        else:
            return img, os.path.basename(path)

# ========== モデル定義 ==========
class EfficientNetB0Modified(nn.Module):
    """1チャネル画像対応EfficientNetB0"""
    def __init__(self, num_classes=10):
        super(EfficientNetB0Modified, self).__init__()
        self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# ========== トレーニング関数 ==========
def train_model(model, loader, criterion, optimizer, device):
    """モデルを1エポック学習"""
    model.train()
    running_loss = 0.0
    for imgs, labels in tqdm(loader):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Train Loss: {running_loss / len(loader):.4f}")

# ========== 評価関数 ==========
def evaluate_model(model, loader, device, criterion=None):
    """バリデーションデータでモデル評価"""
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            if criterion:
                total_loss += criterion(outputs, labels).item()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    print("F1 Macro Score:", f1_score(all_labels, all_preds, average='macro'))
    if criterion:
        print(f"Validation Loss: {total_loss / len(loader):.4f}")
        return total_loss / len(loader)

# ========== メイン処理（5-Fold クロスバリデーション） ==========
if __name__ == "__main__":
    set_seed(713)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    full_dataset = ImageLabelDataset("data/train/low", transform=transform, with_label=True)
    labels = [full_dataset[i][1] for i in range(len(full_dataset))]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=713)

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels), 1):
        print(f"\n========== Fold {fold} ==========")

        train_dataset = Subset(full_dataset, train_idx)
        val_dataset = Subset(full_dataset, val_idx)

        class_counts = Counter([labels[i] for i in train_idx])
        sample_weights = [1.0 / class_counts[labels[i]] for i in train_idx]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

        train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

        model = EfficientNetB0Modified(num_classes=len(LABELS)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        best_path = f"checkpoints/efficientnet_b0_best_fold{fold}.pth"
        early_stopper = EarlyStopping(patience=5, verbose=True, path=best_path)

        for epoch in range(50):
            print(f"\nEpoch {epoch + 1}")
            train_model(model, train_loader, criterion, optimizer, device)
            val_loss = evaluate_model(model, val_loader, device, criterion)
            early_stopper(val_loss, model)
            if early_stopper.early_stop:
                print("Early stopping triggered.")
                break