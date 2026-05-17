import cv2
import easyocr
import numpy as np
import re
from pprint import pp

path = "/home/willie/storage/python312/gokart/data/debug/session-77/column_crops/col01.png"

img = cv2.imread(path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

scale = 4

# 先放大，避免小數點太小被吃掉
gray_big = cv2.resize(
    gray,
    None,
    fx=scale,
    fy=scale,
    interpolation=cv2.INTER_CUBIC
)

# 用來做「找文字列」的二值圖
_, bw = cv2.threshold(
    gray_big,
    0,
    255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

# 移除水平線，例如選取列底下那條黑線
h_kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (25 * scale, 1)
)
horizontal_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel)
bw = cv2.subtract(bw, horizontal_lines)

# 用水平投影找每一列文字
row_projection = (bw > 0).sum(axis=1)

# 門檻不要太高，避免小數點/細字被忽略
row_mask = row_projection > 5

ys = np.where(row_mask)[0]

line_ranges = []

if len(ys) > 0:
    start = ys[0]
    prev = ys[0]

    for y in ys[1:]:
        if y - prev <= 1:
            prev = y
        else:
            if prev - start > 5 * scale:
                line_ranges.append((start, prev))
            start = y
            prev = y

    if prev - start > 5 * scale:
        line_ranges.append((start, prev))

boxes = []

for y1, y2 in line_ranges:
    sub = bw[y1:y2 + 1, :]
    col_projection = (sub > 0).sum(axis=0)

    xs = np.where(col_projection > 0)[0]
    if len(xs) == 0:
        continue

    x1 = xs[0]
    x2 = xs[-1]

    margin_x = 3 * scale
    margin_y = 2 * scale

    x1 = max(0, x1 - margin_x)
    x2 = min(gray_big.shape[1] - 1, x2 + margin_x)
    y1 = max(0, y1 - margin_y)
    y2 = min(gray_big.shape[0] - 1, y2 + margin_y)

    boxes.append((x1, y1, x2, y2))

boxes = sorted(boxes, key=lambda b: b[1])

reader = easyocr.Reader(["en"], gpu=False)

def normalize_time(text):
    text = text.strip()
    text = text.replace(",", ".")
    text = re.sub(r"[^0-9.]", "", text)

    # 如果小數點漏掉，例如 2084 -> 20.84
    if "." not in text and len(text) >= 3:
        text = text[:-2] + "." + text[-2:]

    # 如果多個小數點，只保留最後兩位前的小數點邏輯
    digits = re.sub(r"[^0-9]", "", text)
    if len(digits) >= 3:
        text = digits[:-2] + "." + digits[-2:]

    return text

results = []

for i, (x1, y1, x2, y2) in enumerate(boxes, start=1):
    crop = gray_big[y1:y2 + 1, x1:x2 + 1]

    h, w = crop.shape[:2]

    rec = reader.recognize(
        crop,
        horizontal_list=[[0, w, 0, h]],
        free_list=[],
        detail=1,
        allowlist="0123456789.",
        decoder="greedy",
        paragraph=False
    )

    if rec:
        raw_text = rec[0][1]
        conf = float(rec[0][2])
    else:
        raw_text = ""
        conf = 0.0

    text = normalize_time(raw_text)

    results.append({
        "row": i,
        "raw": raw_text,
        "text": text,
        "conf": conf,
        "box": (x1, y1, x2, y2),
    })

pp(results)