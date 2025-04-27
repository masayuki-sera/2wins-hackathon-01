import os
import glob
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.utils import save_image

# ======== シード固定関数 =========
def set_seed(seed=42):
    """乱数シードを固定して再現性を確保する関数"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ======== デバイス設定 =========
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ======== データセットクラス =========
class ImagePairDataset(Dataset):
    """低画質と高画質の画像ペアを扱うDataset"""
    def __init__(self, low_dir, high_dir):
        self.low_paths = sorted(glob.glob(os.path.join(low_dir, "*.jpg")))
        self.high_paths = sorted(glob.glob(os.path.join(high_dir, "*.jpg")))
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.low_paths)

    def __getitem__(self, idx):
        low = Image.open(self.low_paths[idx]).convert("RGB")
        high = Image.open(self.high_paths[idx]).convert("RGB")
        return self.transform(low), self.transform(high)

# ======== EDSRモデル定義 =========
class ResidualBlock(nn.Module):
    """EDSR内部の残差ブロック"""
    def __init__(self, n_feats):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, 3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)

class EDSR(nn.Module):
    """EDSRモデル全体構成"""
    def __init__(self, in_channels=3, n_feats=64, num_blocks=16):
        super().__init__()
        self.head = nn.Conv2d(in_channels, n_feats, 3, padding=1)
        self.body = nn.Sequential(*[ResidualBlock(n_feats) for _ in range(num_blocks)])
        self.tail = nn.Conv2d(n_feats, in_channels, 3, padding=1)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        x = self.tail(res + x)
        return x

# ======== 学習関数（EarlyStopping対応）=========
def train(model, train_loader, valid_loader, criterion, optimizer, epochs=50, patience=5):
    """学習ループ（検証付き、EarlyStopping機能あり）"""
    best_loss = float("inf")
    best_model_path = "checkpoints/edsr_best.pth"
    wait = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for low, high in tqdm(train_loader, desc=f"[Epoch {epoch+1}] Training"):
            low, high = low.to(device), high.to(device)

            optimizer.zero_grad()
            output = model(low)
            loss = criterion(output, high)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ======== 検証フェーズ =========
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for low, high in valid_loader:
                low, high = low.to(device), high.to(device)
                output = model(low)
                loss = criterion(output, high)
                valid_loss += loss.item()
        valid_loss /= len(valid_loader)

        print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f}, Val Loss: {valid_loss:.4f}")

        # ======== EarlyStopping判定 =========
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model updated at epoch {epoch+1}")
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break

# ======== 推論（テストデータに対して高画質化）=========
def run_inference(model, input_dir, output_dir):
    """学習済みモデルを使った推論関数"""
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    transform = transforms.ToTensor()
    test_paths = sorted(glob.glob(os.path.join(input_dir, "*.jpg")))

    with torch.no_grad():
        for path in tqdm(test_paths, desc="Inference"):
            img = Image.open(path).convert("RGB")
            input_tensor = transform(img).unsqueeze(0).to(device)
            output = model(input_tensor).clamp(0.0, 1.0)

            filename = os.path.basename(path)
            save_image(output.squeeze(0), os.path.join(output_dir, filename))

# ======== メイン実行部 =========
if __name__ == "__main__":
    set_seed(42)

    # ======== データ読み込み・学習/検証分割 =========
    full_dataset = ImagePairDataset("data/train/low", "data/train/high")
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, valid_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

    # ======== モデル・損失関数・最適化手法 =========
    model = EDSR().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # ======== 学習開始 =========
    print("Training started...")
    os.makedirs("checkpoints", exist_ok=True)
    train(model, train_loader, valid_loader, criterion, optimizer, epochs=50, patience=5)

    # ======== 最良モデルで推論開始 =========
    print("Loading best model for inference...")
    model.load_state_dict(torch.load("checkpoints/edsr_best.pth", map_location=device))

    print("Inference started...")
    run_inference(model, input_dir="data/test/low", output_dir="data/test/high_edsr")
    print("Done. Check 'data/test/high_edsr'")
