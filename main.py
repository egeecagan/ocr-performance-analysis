import os
import json
import time
from pathlib import Path
import yaml

from runners.registry import ENGINES

# =============================================================================
# Base paths
# =============================================================================
# All paths are resolved relative to this file's own location, so main.py
# works correctly no matter which directory you run it from. It expects:
#   inputs/images/                      - input images
#   inputs/truths/                      - optional ground-truth .txt files
#   configurations/<engine>/<name>.yaml - per-engine config files
#   outputs/                            - created automatically if missing
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent

IMAGE_DIR = BASE_DIR / "inputs" / "images"
TRUTHS_DIR = BASE_DIR / "inputs" / "truths"
OUTPUTS_DIR = BASE_DIR / "outputs"
CONFIGS_DIR = BASE_DIR / "configurations"


def load_ground_truth(img_name, truths_dir=TRUTHS_DIR):
    """
    Loads the ground-truth text file matching an image, if one exists.
    Example: image1.png -> inputs/truths/image1.yaml

    Returns:
        (text, True)  if the file exists
        (None, False) otherwise
    """
    truth_path = Path(truths_dir) / f"{Path(img_name).stem}.yaml"

    if truth_path.exists():
        with open(truth_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

            # if data is none then it returns empty dictionary!!!
            if data is None: 
                return None, False
                
        return data, True
    return None, False


def process_pipeline(engine, model_name):
    if engine not in ENGINES:
        print(f"Error: '{engine}' is not registered in registry.py.")
        print(f"Registered engines: {list(ENGINES.keys())}")
        return

    engine_spec = ENGINES[engine]
    run_function = engine_spec["run_function"]
    loader = engine_spec["loader"]
    shared_kwargs_map = engine_spec["shared_kwargs"]

    output_dir = OUTPUTS_DIR / engine / model_name
    config_path = CONFIGS_DIR / engine / f"{model_name}.yaml"

    output_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Load the model/reader/processor ONCE, share it across all images
    # =========================================================================
    # If `loader` is set, it's called a single time here, and the resulting
    # object(s) are passed into every run_function call via shared_kwargs.
    # This avoids reloading the model for every image (which would be slow
    # and pointless for heavier engines like TrOCR or doctr). `loader` is
    # None for engines with no real loading cost (e.g. Tesseract).
    shared_objects = {}
    shared_load_time = 0.0
    if loader is not None:
        print(f"[{engine.upper()}] Loading model (once)...")
        load_start = time.time()
        try:
            loaded = loader(str(config_path))
        except Exception as e:
            # If loading fails, report it clearly and stop processing for
            # this engine — but this does not affect other engines you may
            # call afterward in the same script.
            print(
                f"[{engine.upper()}] ERROR: Failed to load model for "
                f"'{engine}'. Likely causes: no internet connection (model "
                f"download), an unsupported language/config setting, or a "
                f"missing dependency.\n"
                f"  Config: {config_path}\n"
                f"  Original error: {e}"
            )
            return
        shared_load_time = round(time.time() - load_start, 4)
        for loaded_key, kwarg_name in shared_kwargs_map.items():
            shared_objects[kwarg_name] = loaded[loaded_key]
        print(f"[{engine.upper()}] Model loaded ({shared_load_time}s), processing images.")

    image_files = sorted(
        f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )

    for img_name in image_files:
        img_path = IMAGE_DIR / img_name

        try:
            output_data = run_function(str(img_path), str(config_path), **shared_objects)
        except Exception as e:
            # A failure on one image (corrupt file, unsupported format, etc.)
            # is reported and skipped — the batch continues with the rest.
            print(f"[{engine.upper()} - {model_name}] ERROR: failed to process {img_name} — {e}")
            continue

        ground_truth_text, has_ground_truth = load_ground_truth(img_name)
        output_data["ground_truth"] = ground_truth_text
        output_data["has_ground_truth"] = has_ground_truth

        # The model-loading time measured once above is written into every
        # image's JSON for this engine. `load_time_is_shared: true` flags
        # this, so when aggregating across images you count it ONCE per
        # engine, not once per image (the load only happened once).
        output_data["load_time_seconds"] = shared_load_time
        output_data["load_time_is_shared"] = loader is not None

        output_file = output_dir / f"{Path(img_name).stem}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)

        print(
            f"[{engine.upper()} - {model_name}] Done: {img_name} | "
            f"Time: {output_data.get('execution_time_seconds')}s | "
            f"GT: {has_ground_truth}"
        )


if __name__ == "__main__":
    process_pipeline("easyocr", "model_v1")
    process_pipeline("tesseract", "model_v1")
    process_pipeline("doctr", "model_v1")
    process_pipeline("rapidocr", "model_v1")