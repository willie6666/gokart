# Repository Notes

## Setup And Commands
- Python package requires `>=3.12`; install with `python -m venv .venv && . .venv/bin/activate && pip install -e '.[dev]'`.
- Prefer `.venv/bin/python -m pytest` in this checkout; bare `pytest` may use an ambient Python missing project deps such as `python-dotenv`.
- Run one test file with `.venv/bin/python -m pytest tests/test_parser.py`; run one test with `.venv/bin/python -m pytest tests/test_parser.py::test_find_time`.
- No lint, formatter, typecheck, CI, or pre-commit config is currently defined in the repo.

## Runtime
- Bot entrypoint is `python -m gokart_bot`; root `bot.py` is only a thin wrapper around `gokart_bot.__main__.main()`.
- Runtime config is loaded from `.env` by `gokart_bot/config.py`; copy `.env.example` and set `DISCORD_TOKEN` before running the bot.
- Discord Developer Portal must enable Message Content Intent, matching `discord.Intents.default()` plus `message_content=True` in `gokart_bot/bot.py`.
- Default data path is `data/`; `.env`, `.venv`, and `data/` are intentionally gitignored.

## Architecture
- `gokart_bot/bot.py` wires Discord slash commands, attachment handling, claim UI, and OCR worker process management.
- OCR flow is grid-only: `ocr.py` calls `table_grid.py` for OpenCV table extraction, `cell_ocr.py` for PaddleOCR words/cells, then `grid_parser.py` and `lap_row_detector.py` for parsed lap records.
- OCR runs in a spawned `ProcessPoolExecutor` with `max_tasks_per_child=1`; `BrokenProcessPool` is handled by recreating the executor so native Paddle crashes do not kill the bot process.
- Keep `GOKART_OCR_ENABLE_MKLDNN=false` unless deliberately testing Paddle CPU acceleration; README notes PaddlePaddle 3.2.0 can hang with MKLDNN on repeated predictions.
- Persistent records are split across `data/<session_id>/session.json`, `raw_ocr.json`, `karts.csv`, and `laps.csv`; `settings.json` stores channel/user settings.

## Tests And Fixtures
- `pyproject.toml` sets `testpaths = ["tests"]` and `pythonpath = ["."]`.
- `tests/test_ocr_pipeline_samples.py` currently skips because fixture JSON references `example/...` while checked-in images are under `examples/...`; do not assume OCR image coverage ran unless the path issue is fixed or an `example/` path exists.
- Tests avoid live Discord and full PaddleOCR predictions; most OCR coverage is for OpenCV grid extraction and parser/postprocess helpers.
