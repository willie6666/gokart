# Gokart Discord Bot

這是一個用 Python 製作的卡丁車歷史紀錄 Discord bot。

## 功能

- 指定一個紀錄圖片頻道，只有該頻道的圖片會自動 OCR。
- Bot 會回覆辨識出的日期、時間、車號、最佳圈速與圈數。
- 使用者可按車號按鈕認領自己的車。
- 使用 JSON 檔保存紀錄，不需要資料庫。
- 使用 `/fix` 手動修正辨識錯誤。
- 使用 `/me`、`/leaderboard`、`/session`、`/laps` 查詢紀錄。

## 安裝

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
```

PaddleOCR 第一次使用會下載模型，啟動與首次辨識會比較久。

PaddleOCR 預設使用 `paddle_static` 推論引擎，因此必須安裝 `paddlepaddle`。本專案已把它列入依賴；如果你是在更新後遇到 `dependency 'paddlepaddle' is not installed`，請重新執行 `pip install -e '.[dev]'`。

Bot 會強制讓 PaddleOCR 使用 CPU 並關閉 MKLDNN/oneDNN，避免部分 Linux/Nix CPU 環境出現 Paddle Inference backend crash。

預設 OCR 使用 grid 流程：OpenCV 先找主表格、做透視校正、偵測格線並切 cell，OCR 只讀每個 cell。舊的整張圖 OCR 仍保留為 fallback，可用 `GOKART_OCR_MODE=legacy` 切回。

預設 PaddleOCR 使用 `PP-OCRv5_server_det` + `PP-OCRv5_server_rec`，並把圖片長邊縮到 `1600px`，目標是在保留表格小字辨識率的同時降低 RAM、硬碟 I/O 與等待時間。grid cell OCR 建議安裝 Tesseract；如果沒有 Tesseract，bot 會 fallback 到 legacy OCR，避免用 Paddle 逐格 OCR 造成 Discord gateway 斷線。

`GOKART_OCR_RECTIFY_TABLE=true` 會先用 OpenCV 偵測表格線，裁出主表格並依水平線旋轉校正；如果校正後仍有未知車號，bot 會用未校正原圖做一次備援辨識，只補缺少的車號。

## 設定

```bash
cp .env.example .env
```

編輯 `.env`：

```env
DISCORD_TOKEN=replace-with-your-bot-token
GOKART_DATA_PATH=data/gokart_records.json
GOKART_IMAGE_DIR=data/images
GOKART_MIN_OCR_CONFIDENCE=0.50
GOKART_OCR_MODE=grid
GOKART_DEBUG_OCR=true
GOKART_DEBUG_DIR=data/debug
GOKART_OCR_MAX_SIDE=1600
GOKART_OCR_DET_LIMIT_SIDE_LEN=1600
GOKART_OCR_DET_MODEL=PP-OCRv5_server_det
GOKART_OCR_REC_MODEL=PP-OCRv5_server_rec
GOKART_OCR_CPU_THREADS=1
GOKART_OCR_RECTIFY_TABLE=false
```

`GOKART_DEBUG_OCR=true` 會為每次辨識輸出 `original.jpg`、`table_warped.png`、`grid_overlay.png`、`cell_ocr.json` 與 `parsed.json` 到 `data/debug/session-{id}/`。

Discord Developer Portal 需要開啟 bot 的 `Message Content Intent`，否則 bot 無法監聽圖片訊息。

## 執行

```bash
python -m gokart_bot
```

也可以：

```bash
python bot.py
```

## 使用方式

1. 先用 `/setrecordchannel` 設定紀錄圖片頻道。
2. 在該頻道上傳 `example/` 類似的成績表照片。
3. Bot 會回覆辨識結果。
4. 使用者按下自己的車號按鈕認領。
5. 如果辨識錯誤，用 `/fix` 修正。

## Slash Commands

- `/me`：查詢自己的歷史紀錄。
- `/leaderboard limit:10`：查詢排行榜，使用較整齊的表格格式。
- `/session session_id:1`：查詢某場紀錄。
- `/laps session_id:1`：查詢某場所有車號的完整圈速，內容太長會自動附上文字檔。
- `/laps session_id:1 kart_no:12`：查詢某場特定車號的完整圈速。
- `/fix session_id:1 kart_no:12 best_lap:19.65`：修正某車最佳圈速。
- `/fix session_id:1 kart_no:12 laps:20.10,19.65,20.00`：用完整圈速修正某車紀錄。
- `/fix session_id:1 position:3 kart_no:7`：把第 3 欄的 `車號未知` 改成車號 7，保留原本辨識出的圈速。
- `/unclaim session_id:1`：取消自己在某場的認領。
- `/recordchannel`：查看目前紀錄圖片頻道。
- `/setrecordchannel channel:#records`：設定紀錄圖片頻道。需要管理伺服器權限。
- `/clearrecordchannel`：清除紀錄圖片頻道設定，停止自動辨識圖片。需要管理伺服器權限。
- `/sync`：重新同步 slash commands。

## JSON 資料

預設資料會存到 `data/gokart_records.json`。圖片會存到 `data/images/`。

## 注意

OCR 目前使用座標式 heuristic parser，對範例格式最佳。若照片太歪、太糊、表格被遮住，請用 `/fix` 手動修正。
