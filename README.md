# Gokart Discord Bot

Discord bot for OCRing go-kart lap sheets and tracking claimed driver records.

## Features

- Watch one configured Discord channel for uploaded lap sheet images.
- Use OpenCV to find, warp, and grid-slice the table.
- Use EasyOCR-only OCR for headers, kart numbers, lap indexes, and lap times.
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

EasyOCR downloads its model files on first use, so the first OCR run is slower.

## Configuration

```bash
cp .env.example .env
```

Edit `.env`:

```env
DISCORD_TOKEN=replace-with-your-bot-token
GOKART_DATA_PATH=data/gokart_records.json
GOKART_IMAGE_DIR=data/images
GOKART_DEBUG_OCR=true
GOKART_DEBUG_DIR=data/debug
GOKART_OCR_MAX_SIDE=1600
GOKART_OCR_WORKERS=1
```

Settings:

- `DISCORD_TOKEN`: Discord bot token.
- `GOKART_DATA_PATH`: JSON store path.
- `GOKART_IMAGE_DIR`: saved upload image directory.
- `GOKART_DEBUG_OCR`: write debug artifacts to `GOKART_DEBUG_DIR` when true.
- `GOKART_DEBUG_DIR`: local debug artifact directory.
- `GOKART_OCR_MAX_SIDE`: image resize max side before table detection.
- `GOKART_OCR_WORKERS`: dedicated OCR thread pool size. Use `1` for lowest CPU contention; increase only if the host can handle multiple OCR jobs.

Discord Developer Portal must enable `Message Content Intent`, otherwise the bot cannot see image messages.

## OCR Pipeline

The OCR path is grid-only:

- OpenCV scans and warps the paper/table.
- OpenCV detects grid lines and crops cells/columns.
- EasyOCR recognizes all text.
- Lap columns use OpenCV projection to split each row, then EasyOCR recognizes each row crop.
- Virtual lap rows map OCR words back to lap numbers.

EasyOCR Reader instances are cached per OCR worker thread. The bot uses a dedicated OCR executor so long-running image recognition does not run on the Discord event loop.

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
- `/leaderboard limit:10`: show each claimed driver once, using their personal best lap.
- `/session session_id:1`: show one parsed session.
- `/heat heat_id:1`: alias-style session lookup by record id.
- `/laps session_id:1`: show full lap times for a session.
- `/laps session_id:1 kart_no:12`: show full lap times for one kart.
- `/fix session_id:1 kart_no:12 best_lap:19.65`: correct a kart's best lap.
- `/fix session_id:1 kart_no:12 laps:20.10,19.65,20.00`: replace a kart's full lap list.
- `/fix session_id:1 position:3 kart_no:7`: set the kart number for an unknown column position.
- `/unclaim session_id:1`: remove your claim for one session.
- `/recordchannel`: show the current image channel.
- `/setrecordchannel channel:#records`: set the image channel. Requires Manage Server.
- `/clearrecordchannel`: clear the image channel. Requires Manage Server.
- `/debugchannel`: show the current OCR debug channel.
- `/setdebugchannel channel:#debug`: send selected debug artifacts to a channel. Requires Manage Server.
- `/cleardebugchannel`: clear the OCR debug channel. Requires Manage Server.
- `/ping`: health check.

## Debug Artifacts

When `GOKART_DEBUG_OCR=true`, local debug files are written to `data/debug/session-{id}/`:

- `original.jpg`
- `paper_warped.png`
- `table_warped.png`
- `grid_overlay.png`
- `virtual_rows_overlay.png`
- `cell_ocr.json`
- `ocr_words.json`
- `parsed.json`
- `column_crops/`
- `cells/`

If `/setdebugchannel` is configured, the bot uploads these artifacts after each OCR regardless of `GOKART_DEBUG_OCR`:

- `grid_overlay.png`
- `table_warped.png`
- `virtual_rows_overlay.png`
- `parsed.json`

## Data

Records are stored in `data/gokart_records.json` by default. Uploaded images are saved in `data/images/`.

## Notes

The parser is tuned for the lap sheet layout in `example/`. If a photo is too blurry, skewed, cropped, or uses a different table layout, use `/fix` to correct the stored record.
