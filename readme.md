# OCR Performance Benchmark Project

A comparison of different OCR engines on Turkish documents such as driver's licenses and bank receipts, evaluated for accuracy and speed. The goal is to produce a data-driven report for selecting the most suitable OCR solution for **client-side use on an older/low-spec computer**.

## Table of Contents

- [Engines Tested](#engines-tested)
- [Project Architecture](#project-architecture)
- [Installation](#installation)
- [Folder Structure](#folder-structure)
- [Usage](#usage)
- [Config Files](#config-files)
- [Preprocessing](#preprocessing)
- [Output (JSON) Schema](#output-json-schema)
- [Running PaddleOCR via Docker](#running-paddleocr-via-docker)
- [Adding a New Engine](#adding-a-new-engine)

## Engines Tested

| Engine | Type | Notes |
|---|---|---|
| **Tesseract** | Classical (with LSTM) | Lightweight, no real model-loading cost |
| **EasyOCR** | Deep learning | PyTorch-based |
| **doctr** | Deep learning | Separate detection and recognition models |
| **TrOCR** | Transformer | Designed for single-line text, no word-level confidence |
| **RapidOCR** | Deep learning (ONNX) | ONNX-converted PaddleOCR models, bundled with the package (works offline) |
| **PaddleOCR** | Deep learning | The original model RapidOCR is based on; downloads models from the internet on first use |

> **Note:** RapidOCR and PaddleOCR are built on the same underlying (PP-OCR) models — the difference between them is expected to show up not in accuracy, but in **deployment/installation complexity** (see [Running PaddleOCR via Docker](#running-paddleocr-via-docker)).

## Project Architecture

Each engine is defined in `runners/run_<engine>.py` as a function following the **same contract**:

```python
def run_<engine>(image_path, config_path, **optional_model_kwargs) -> dict
```

All runners return the same JSON schema (see [Output Schema](#output-json-schema)), so reporting/comparison code can work independently of which engine produced the result.

### Shared module: `runners/_common.py`

Engine-agnostic helper functions shared by all runners:

- `load_config` — reads the YAML config
- `preprocess_image` — a 10-step, opt-in image preprocessing pipeline (see [Preprocessing](#preprocessing))
- `resolve_font_path` — finds a font that renders Turkish characters correctly for the masked visualization
- `filter_valid_kwargs` — restricts config-provided parameters to whatever the target function actually accepts
- `get_viz_dirs` / `save_preprocessed_image` — output folder management

### Registry pattern: `runners/registry.py`

`main.py` learns which engines exist, their runner function, and how their model/reader should be loaded from the `ENGINES` dictionary in `registry.py`. **You don't need to touch `main.py` when adding a new engine** — see [Adding a New Engine](#adding-a-new-engine).

### Model sharing

`main.py` loads each engine's model/reader **once** and shares it across all images (it does not reload per image). The load time is reflected in the JSON via `load_time_seconds` and `load_time_is_shared: true` — when reporting, count this value **once per engine**, not once per image.

### Fault tolerance

If an image fails to process (corrupt file, unsupported format, etc.), `main.py` reports it clearly to the console and **continues processing the remaining images** — a single bad file does not halt the entire batch.

## Installation

```bash
# Virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Shared dependencies
pip install opencv-python numpy pillow pyyaml

# Per-engine dependencies (install only what you need)
pip install pytesseract          # + system package: brew install tesseract tesseract-lang (Mac)
pip install easyocr
pip install python-doctr
pip install transformers torch   # for TrOCR
pip install rapidocr
pip install paddleocr paddlepaddle   # see Docker note — paddlepaddle can be problematic on Mac
```

> The **Tesseract binary** must be installed separately from the Python package: `brew install tesseract tesseract-lang` (Mac) / `apt install tesseract-ocr tesseract-ocr-tur` (Linux).

## Folder Structure

```
project/
├── main.py                      # Orchestrator — runs all images through one engine
├── runners/
│   ├── _common.py               # Shared helper functions
│   ├── registry.py              # Engine registry
│   └── run_<engine>.py          # Runner for each engine (6 files)
├── configurations/
│   └── <engine>/model_v1.yaml   # One config per engine + version
├── inputs/
│   ├── images/                  # Test images (licenses, receipts, etc.) — excluded from git
│   └── truths/                  # Ground-truth .txt files (optional, for CER/WER)
└── outputs/
    └── <engine>/<model_name>/
        ├── <image_name>.json    # OCR result + timing + confidence + ground truth
        └── viz/
            ├── highlighted/     # Detected words marked with red boxes
            ├── masked/          # Text re-rendered on a blank white page
            └── preprocessed/    # The exact image the engine received (post-preprocessing)
```

`inputs/images`, `inputs/truths`, and `outputs/` are excluded via `.gitignore` (they may contain sensitive document images).

## Usage

### Run one engine against all images

```python
import main as main_module

main_module.process_pipeline('tesseract', 'model_v1')
main_module.process_pipeline('rapidocr', 'model_v2_tiny')
```

The first argument is the engine name (one of the keys in `registry.py`'s `ENGINES`), the second is the YAML filename (without extension) under `configurations/<engine>/`.

### Test a single image

```python
from runners.run_tesseract import run_tesseract

result = run_tesseract("inputs/images/sample.png", "configurations/tesseract/model_v1.yaml")
print(result["text"], result["avg_confidence"])
```

### Run main.py from anywhere

All paths are resolved relative to `main.py`'s own location (absolute paths) — it works regardless of which directory you run it from.

## Config Files

Each engine's `configurations/<engine>/model_v1.yaml` file contains three main sections:

1. **Engine-specific settings** (`ocr_settings` / `model_settings` / `reader_settings`) — language, model size, GPU usage, etc.
2. **`preprocessing`** — the same schema across all engines (see below)
3. Some engines have additional blocks, e.g. **`generate_settings`** for TrOCR

You can create multiple configs for the same engine (`model_v1.yaml`, `model_v2_tiny.yaml`, etc.) to compare different settings — each is written to its own folder under `outputs/<engine>/<config_name>/`.

## Preprocessing

The `preprocess_image()` function in `_common.py` provides a 10-step image processing pipeline applied before the image reaches the OCR engine — every step is independently toggleable via config, and **all are disabled by default**:

| Step | Purpose |
|---|---|
| `resize` | Upscales low-resolution images |
| `deskew` | Corrects tilted text |
| `autocrop_border` | Crops empty margins |
| `illumination_correct` | Fixes shadow/lighting unevenness |
| `grayscale` | Color → grayscale |
| `clahe` | Localized contrast enhancement |
| `denoise` | Removes small noise/specks |
| `sharpen` | Reduces blur |
| `morphology` | Merges thin/broken characters |
| `threshold` | Reduces to black/white (binarization) |

**Important:** Enabling preprocessing does not guarantee improvement — in our testing, aggressive threshold/contrast settings sometimes *lowered* confidence, particularly for deep-learning-based engines. Which combination actually helps should be measured against your real data (via ground-truth comparison).

The final image sent to the engine (after preprocessing) is always saved to `outputs/<engine>/<model>/viz/preprocessed/` — even with preprocessing disabled, in which case it's simply a copy of the raw image.

## Output (JSON) Schema

Common fields present in every per-image JSON output:

```json
{
  "text": "recognized text (single line)",
  "load_time_seconds": 0.0,
  "load_time_is_shared": true,
  "image_load_time_seconds": 0.01,
  "preprocessing_time_seconds": 0.0,
  "execution_time_seconds": 0.5,
  "total_time_seconds": 0.51,
  "avg_confidence": 97.78,
  "model_used": "engine-detail",
  "device_used": "cpu",
  "settings_used": { },
  "preprocessing_used": { },
  "ground_truth": null,
  "has_ground_truth": false
}
```

- **`load_time_seconds`**: Time taken to load the model/reader. If `load_time_is_shared: true`, every image for this engine carries the same value — count it once when aggregating, not once per image.
- **`total_time_seconds`**: Image reading + preprocessing + the engine itself (excluding model loading) — i.e. "the real end-to-end cost of processing one image."
- **`avg_confidence`**: On a 0–100 scale, how confident the engine was in its own output. `None` means confidence wasn't measurable under this engine/setting (not to be confused with an actual 0).
- Some engines have additional, engine-specific fields (e.g. `doc_write_time_seconds` for doctr, `pil_conversion_time_seconds` for TrOCR).

## Running PaddleOCR via Docker

On Apple Silicon Macs, some `paddlepaddle` backends (`paddle_static`) aren't supported. In that case, you can run PaddleOCR inside a Linux container instead:

```bash
docker compose build
docker compose run --rm paddleocr          # runs with model_v1
docker compose run --rm paddleocr model_v2 # runs with a different config
```

The container mounts your `inputs/`, `outputs/`, and `configurations/` folders as volumes — the JSON it produces is written directly into your `outputs/paddleocr/` folder, in the same format as the other engines running natively on your Mac.

## Adding a New Engine

1. Write `runners/run_<engine>.py` following the same contract as the existing runners (same returned JSON schema, using the shared helpers in `_common.py`).
2. Add an entry to the `ENGINES` dictionary in `runners/registry.py`:
   ```python
   "new_engine": {
       "run_function": run_new_engine,
       "loader": _load_new_engine_model,  # or None if there's no real model-loading step
       "shared_kwargs": {"model": "model"},
   },
   ```
3. Create `configurations/new_engine/model_v1.yaml`.

**You don't need to touch `main.py`** — `process_pipeline('new_engine', 'model_v1')` will work automatically.
