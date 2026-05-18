# Gokart Discord Bot

Discord bot for OCRing go-kart lap sheets and tracking claimed driver records.

## Features

- Watch one configured Discord channel for uploaded lap sheet images.
- Use OpenCV to find, warp, and grid-slice the table.
- Use PaddleOCR-only OCR for headers, kart numbers, lap indexes, and lap times.
- Store records in a local JSON file.
- Let drivers claim their kart result from Discord buttons.
- Show per-session results, full lap lists, personal records, and a claimed-driver leaderboard.
- Optionally send debug artifacts to a configured debug channel.

## Install

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
```

PaddleOCR downloads its model files on first use, so the first OCR run is slower.

## Configuration

```bash
cp .env.example .env
```

Edit `.env`:

```env
DISCORD_TOKEN=replace-with-your-bot-token
GOKART_DATA_PATH=data
GOKART_OCR_MAX_SIDE=1600
GOKART_OCR_WORKERS=1
GOKART_OCR_LANG=ch
GOKART_OCR_DEVICE=cpu
GOKART_OCR_ENABLE_MKLDNN=false
GOKART_OCR_CPU_THREADS=8
```

Settings:

- `DISCORD_TOKEN`: Discord bot token.
- `GOKART_DATA_PATH`: record data directory.
- `GOKART_OCR_MAX_SIDE`: image resize max side before table detection.
- `GOKART_OCR_WORKERS`: dedicated OCR process pool size. Use `1` for lowest CPU contention; increase only if the host can handle multiple OCR jobs.
- `GOKART_OCR_LANG`: PaddleOCR recognition language. Default `ch` supports Chinese/English mixed text, including `上午` and `下午`.
- `GOKART_OCR_DEVICE`: PaddleOCR device, usually `cpu`.
- `GOKART_OCR_ENABLE_MKLDNN`: enable PaddlePaddle MKLDNN CPU acceleration. Default is false because PaddlePaddle 3.2.0's Python API can hang on repeated predictions with MKLDNN enabled.
- `GOKART_OCR_CPU_THREADS`: PaddleOCR CPU thread count. Set this to your CPU core count for a single OCR worker.

Discord Developer Portal must enable `Message Content Intent`, otherwise the bot cannot see image messages.

## OCR Pipeline

The OCR path is grid-only:

- OpenCV scans and warps the paper/table.
- OpenCV detects grid lines and table cells.
- PaddleOCR recognizes all text.
- Lap columns are cropped from the detected grid, then PaddleOCR detects and recognizes row words in each column.
- Virtual lap rows map OCR words back to lap numbers.

PaddleOCR runs in dedicated child processes instead of the Discord bot process. Each OCR worker handles one image and exits, so a PaddlePaddle native crash should fail that OCR job without killing the bot. Keep MKLDNN disabled for stability unless you are testing locally.

## Run

```bash
python -m gokart_bot
```

## Usage

1. Use `/setrecordchannel` in Discord to set the image channel.
2. Upload a lap sheet image to that channel.
3. The bot replies with parsed results and claim buttons.
4. Drivers claim their result with the button/dropdown UI.
5. Use `/fix` if OCR needs manual correction.

## Slash Commands

- `/me`: show your claimed records.
- `/profile user:@driver`: show another user's claimed records.
- `/myrecords limit:5`: show your recent claimed records.
- `/leaderboard limit:10 ranking:最佳單圈`: show each claimed driver once, ranked by personal best lap.
- `/leaderboard limit:10 ranking:最佳平均`: show each claimed driver once, ranked by personal best average.
- `/session session_id:1`: show one parsed session.
- `/heat heat_id:1`: alias-style session lookup by record id.
- `/laps session_id:1`: show full lap times for a session.
- `/laps session_id:1 kart_no:12`: show full lap times for one kart.
- `/fix session_id:1 kart_no:12 best_lap:19.65`: correct a kart's best lap.
- `/fix session_id:1 kart_no:12 laps:20.10,19.65,20.00`: replace a kart's full lap list.
- `/fix session_id:1 position:3 kart_no:7`: set the kart number for an unknown column position.
- `/deleterecord session_id:1`: delete one record and its saved files.
- `/unclaim session_id:1`: remove your claim for one session.
- `/recordchannel`: show the current image channel.
- `/setrecordchannel channel:#records`: set the image channel. Requires Manage Server.
- `/clearrecordchannel`: clear the image channel. Requires Manage Server.
- `/debugchannel`: show the current OCR debug channel.
- `/setdebugchannel channel:#debug`: send each record's CSV, JSON, and image files to a channel. Requires Manage Server.
- `/cleardebugchannel`: clear the OCR debug channel.
- `/ping`: health check.

## Debug Artifacts

Debug files are always saved with each record. JSON files are written to `data/<record_number>/`:

- `session.json`
- `raw_ocr.json`
- `parsed.json`
- `cell_ocr.json`
- `ocr_words.json`

CSV files are written to `data/<record_number>/`:

- `karts.csv`
- `laps.csv`

Pre-processing images are written to `data/<record_number>/images/`:

- `original_resized.png`
- `paper_contours.png`
- `paper_warped.png`
- `line_mask.png`
- `line_mask_horizontal.png`
- `line_mask_vertical.png`
- `table_contour.png`
- `table_warped.png`
- `grid_overlay.png`

OCR images are written to `data/<record_number>/images/`:

- `paddleocr_overlay.png`
- `grid_parsed_overlay.png`
- `virtual_rows_overlay.png`
- `source.<ext>`

If `/setdebugchannel` is configured, the bot uploads images first, then CSV/JSON data files.

## Data

Records are stored in `data/<record_number>/` by default. `session.json` stores record metadata only, `raw_ocr.json` stores OCR summary data, `karts.csv` stores kart-level results/claims, and `laps.csv` stores lap times. Bot settings are stored in `data/settings.json`.

## Notes

The parser is tuned for the lap sheet layout in `example/`. If a photo is too blurry, skewed, cropped, or uses a different table layout, use `/fix` to correct the stored record.
