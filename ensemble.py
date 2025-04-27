# ========== ライブラリインポート ==========
import os
import glob
import pandas as pd
from PIL import Image
from collections import defaultdict, Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import efficientnet_b0, resnet50
from torchvision.models import EfficientNet_B0_Weights, ResNet50_Weights

# ========== ラベル定義 ==========
LABELS = ["玄関", "バルコニー", "浴室", "トイレ", "収納", "洋室", "クローゼット", "廊下", "ホール", "和室"]
label_to_idx = {label: i for i, label in enumerate(LABELS)}
idx_to_label = {i: label for label, i in label_to_idx.items()}

# ========== データセットクラス ==========
class ImageLabelDataset(Dataset):
    """テストデータ用Dataset（ラベルなし）"""
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
        img = Image.open(path).convert('L')  # グレースケール化
        img = self.transform(img)
        return img, os.path.basename(path)

# ========== モデル定義 ==========
class EfficientNetB0Modified(nn.Module):
    """1チャネル対応EfficientNetB0モデル"""
    def __init__(self, num_classes=10):
        super(EfficientNetB0Modified, self).__init__()
        self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.model(x)

class ResNet50Modified(nn.Module):
    """1チャネル対応ResNet50モデル"""
    def __init__(self, num_classes=10):
        super(ResNet50Modified, self).__init__()
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# ========== ハードアンサンブル推論関数 ==========
def run_hard_ensemble(models_with_paths, output_csv):
    """複数モデルによる多数決アンサンブル推論とOCR予測の補完"""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    test_dataset = ImageLabelDataset("data/test/low", transform=transform, with_label=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    all_predictions = defaultdict(list)

    for model_class, weight_path in models_with_paths:
        print(f"Loading model from {weight_path}")
        model = model_class(num_classes=len(LABELS)).to(device)
        model.load_state_dict(torch.load(weight_path, map_location=device))
        model.eval()

        with torch.no_grad():
            for imgs, filenames in test_loader:
                imgs = imgs.to(device)
                outputs = model(imgs)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                labels = [idx_to_label[p] for p in preds]
                for fname, label in zip(filenames, labels):
                    all_predictions[fname].append(label)

    # アンサンブルのみの結果保存
    ensemble_preds = []
    for fname, preds in all_predictions.items():
        majority_label = Counter(preds).most_common(1)[0][0]
        ensemble_preds.append((fname, majority_label))

    ensemble_df = pd.DataFrame(ensemble_preds, columns=["id", "label"])
    ensemble_df["sort_key"] = ensemble_df["id"].str.extract(r'(\d+)')[0].astype(int)
    ensemble_df = ensemble_df.sort_values("sort_key").drop(columns="sort_key")
    os.makedirs("pred", exist_ok=True)
    ensemble_df.to_csv("pred/ensemble_only.csv", index=False)
    print("Saved pure ensemble predictions to pred/ensemble_only.csv")

    # OCR予測を読み込み
    ocr_df = pd.read_csv("pred/ocr_prediction.csv")
    ocr_preds = dict(zip(ocr_df["id"], ocr_df["label"]))

    # 最終予測を決定（不明のみアンサンブル結果で補完）
    final_preds = []
    for fname in ocr_preds.keys():
        if ocr_preds[fname] != "不明":
            final_preds.append((fname, ocr_preds[fname]))
        else:
            majority_label = Counter(all_predictions[fname]).most_common(1)[0][0]
            final_preds.append((fname, majority_label))

    # 最終提出用結果保存
    submission_df = pd.DataFrame(final_preds, columns=["id", "label"])
    submission_df["sort_key"] = submission_df["id"].str.extract(r'(\d+)')[0].astype(int)
    submission_df = submission_df.sort_values("sort_key").drop(columns="sort_key")
    submission_df.to_csv(output_csv, index=False)
    print(f"Saved final ensemble predictions to {output_csv}")

# ========== 実行部 ==========
if __name__ == "__main__":
    models_to_ensemble = [
        (EfficientNetB0Modified, "checkpoints/efficientnet_b0_best_fold1.pth"),
        (EfficientNetB0Modified, "checkpoints/efficientnet_b0_best_fold2.pth"),
        (EfficientNetB0Modified, "checkpoints/efficientnet_b0_best_fold3.pth"),
        (EfficientNetB0Modified, "checkpoints/efficientnet_b0_best_fold4.pth"),
        (EfficientNetB0Modified, "checkpoints/efficientnet_b0_best_fold5.pth"),
        (ResNet50Modified, "checkpoints/resnet50_best_fold1.pth"),
        (ResNet50Modified, "checkpoints/resnet50_best_fold2.pth"),
        (ResNet50Modified, "checkpoints/resnet50_best_fold3.pth"),
        (ResNet50Modified, "checkpoints/resnet50_best_fold4.pth"),
        (ResNet50Modified, "checkpoints/resnet50_best_fold5.pth")
    ]

    run_hard_ensemble(models_to_ensemble, "pred/final_submission.csv")
