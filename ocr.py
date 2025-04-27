import os
import glob
import pandas as pd
from PIL import Image
from google.cloud import vision
from tqdm import tqdm

# ========== ラベル定義 ==========
LABELS = ["玄関", "バルコニー", "浴室", "トイレ", "収納", "洋室", "クローゼット", "廊下", "ホール", "和室"]

# ========== 認証とGoogle Visionクライアント設定 ==========
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key.json"
client = vision.ImageAnnotatorClient()

# ========== OCR検出関数 ==========
def detect_text_from_image(path):
    """指定画像からOCRでテキストを検出"""
    with open(path, "rb") as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    return response.text_annotations  # texts[0] は全文、以降個別テキスト

# ========== ラベルマッピング関数（中央エリア優先） ==========
def map_text_to_label(texts, image_size, max_x_distance_ratio=0.1, center_ratio=0.1):
    """
    OCR結果を元に画像に最も適したラベルを推定する

    Parameters:
    - texts: Vision APIのtext_annotationsリスト
    - image_size: (幅, 高さ)
    - max_x_distance_ratio: 中心からの許容距離割合
    - center_ratio: 中央帯の幅割合
    """
    if not texts or len(texts) <= 1:
        return "不明"

    img_w, img_h = image_size
    img_cx, img_cy = img_w / 2, img_h / 2
    max_x_dist = img_w * max_x_distance_ratio
    cx_margin = img_w * center_ratio / 2
    cy_margin = img_h * center_ratio / 2

    matched = []
    center_area_has_text = False
    center_area_has_label = False

    for text_obj in texts[1:]:  # texts[0]は全文、スキップ
        desc = text_obj.description
        vertices = text_obj.bounding_poly.vertices
        xs = [v.x for v in vertices if v.x is not None]
        ys = [v.y for v in vertices if v.y is not None]
        if not xs or not ys:
            continue

        x_center = sum(xs) / len(xs)
        y_center = sum(ys) / len(ys)

        in_center_area = (
            abs(x_center - img_cx) <= cx_margin and
            abs(y_center - img_cy) <= cy_margin
        )

        if in_center_area:
            center_area_has_text = True

        for label in LABELS:
            if label in desc:
                matched.append((label, x_center, y_center))
                if in_center_area:
                    center_area_has_label = True
                break

    # 優先ルール適用
    if center_area_has_text and not center_area_has_label:
        return "不明"

    for label, x, y in matched:
        if abs(x - img_cx) <= cx_margin and abs(y - img_cy) <= cy_margin:
            return label

    if matched:
        closest = min(matched, key=lambda item: abs(item[1] - img_cx))
        if abs(closest[1] - img_cx) <= max_x_dist:
            return closest[0]

    return "不明"

# ========== メイン処理 ==========
if __name__ == "__main__":
    image_dir = "data/test/high_edsr"
    image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))
    results = []

    # OCRとラベル推定
    for path in tqdm(image_paths, desc="OCR中", ncols=80):
        file_name = os.path.basename(path)
        img = Image.open(path)
        texts = detect_text_from_image(path)
        label = map_text_to_label(texts, img.size)
        results.append((file_name, label))

    # 結果をDataFrameにまとめて保存
    df = pd.DataFrame(results, columns=["id", "label"])
    df["sort_key"] = df["id"].str.extract(r'(\d+)')[0].astype(int)
    df = df.sort_values("sort_key").drop(columns="sort_key")

    os.makedirs("pred", exist_ok=True)
    df.to_csv("pred/ocr_prediction.csv", index=False)

    print("Saved to pred/ocr_prediction.csv")
