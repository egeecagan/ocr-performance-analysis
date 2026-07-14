"""
main.py

Entry point for running the OCR pipeline across one or more engine/model
combinations. For each (engine, model_name) pair passed to
process_pipeline():

  - loads the engine's model/reader ONCE (via runners/registry.py), if it
    has a real loading cost,
  - runs every image in inputs/images/ through the engine's run_function,
  - scores the result against per-image ground truth (inputs/truths/*.yaml)
    and against document-type-specific common fields
    (inputs/truths/common_fields/), both via the LCS/CER matcher in
    runners/lcs_cer.py,
  - writes one JSON file per image to outputs/<engine>/<model_name>/.

Running this file directly (`python main.py`) clears outputs/, processes
the (engine, model_name) pairs listed in the `if __name__ == "__main__":`
block below, and then generates the aggregate comparison report via
generate_report.py.

=== TO RUN A NEW ENGINE/MODEL ===
Add a process_pipeline("<engine>", "<model_name>") call in the
`if __name__ == "__main__":` block. `<engine>` must be a key in
runners/registry.py's ENGINES dict; `<model_name>` must match an existing
configurations/<engine>/<model_name>.yaml file.
"""

import os
import json
import time
import shutil
from pathlib import Path
from typing import Any
import yaml
from runners.registry import ENGINES
from runners.accuracy import (
    detect_common_fields_file,
    load_common_fields,
)
from runners.lcs_cer import check_all_fields_lcs_cer_with_bbox, enrich_words_with_field_matches_lcs
from generate_report import generate_report

BASE_DIR = Path(__file__).resolve().parent

IMAGE_DIR = BASE_DIR / "inputs" / "images"
TRUTHS_DIR = BASE_DIR / "inputs" / "truths"
COMMON_FIELDS_DIR = BASE_DIR / "inputs" / "truths" / "common_fields"
OUTPUTS_DIR = BASE_DIR / "outputs"
CONFIGS_DIR = BASE_DIR / "configurations"

PROCESSED_MODELS = []


def detect_common_fields_by_content(words: list, common_fields_dir: Path) -> Path | None:
    """
    Dosya adından belge türü tespit EDILEMEYEN durumlarda (örn. kullanıcı
    'photo.jpg' ya da 'IMG_1234.jpg' gibi genel bir isimle dosya yüklediğinde)
    devreye giren içerik tabanlı otomatik belge türü algılayıcı.

    Mevcut tüm _c.txt dosyalarını tarar ve OCR çıktısındaki kelimelerle
    karşılaştırır; en fazla kelime eşleşmesi veren türü döner.

    Parametreler:
        words           : OCR çıktısındaki kelime listesi (her biri dict,
                          'text' anahtarı içeriyor)
        common_fields_dir: inputs/truths/common_fields/ dizini

    Döner:
        En iyi eşleşen _c.txt dosyasının Path'i, ya da hiç eşleşme
        yoksa None.
    """
    from runners.accuracy import normalize_text
    from rapidfuzz import fuzz

    # OCR metnini hem Türkçe hem de ASCII olarak normalize edilmiş tek bir string'e çevir
    ocr_text = normalize_text(
        " ".join(w.get("text", "") for w in words if isinstance(w, dict)),
        ascii_normalize=True
    )

    if not ocr_text.strip():
        return None

    best_file  = None
    best_score = 0

    for txt_path in common_fields_dir.glob("*_c.txt"):
        cf = load_common_fields(txt_path)   # {kelime: kelime}
        if not cf:
            continue

        matched = 0
        for field_val in cf.values():
            norm_val = normalize_text(field_val, ascii_normalize=True)
            # Kelimenin kendisi OCR metninde fuzzy olarak %65 ve üzeri oranla geçiyor mu?
            if norm_val and fuzz.partial_ratio(norm_val, ocr_text) >= 65.0:
                matched += 1

        # En az 1 eşleşme gereken kısmen/tam eşleşme için "kelime adedi" skoru
        score = matched
        if score > best_score:
            best_score = score
            best_file  = txt_path

    return best_file if best_score > 0 else None

def load_ground_truth(img_name: str, truths_dir: str | Path = TRUTHS_DIR) -> tuple[dict[str, Any] | None, bool]:
    """
    Loads the ground-truth YAML file matching an image, if one exists.
    Example: image1.png -> inputs/truths/image1.yaml

    Returns:
        (data, True)  if the file exists and contains data — data is the
                       FULL parsed YAML dict, e.g. {"fields": {...}}
        (None, False) otherwise
    """
    truth_path = Path(truths_dir) / f"{Path(img_name).stem}.yaml"

    if truth_path.exists():
        with open(truth_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

            if data is None:
                return None, False

        return data, True
    return None, False


def process_single_image(img_path: str, engine: str, model_name: str, original_name: str = None) -> dict:
    """
    API tarafindan cagrilir. Tek bir gorsel dosyasini isler ve
    OCR cikti sozlugunu (words, bbox, metrikler vb.) geri dondurur.

    Parametreler:
        img_path   : Gorselin tam dosya yolu (str)
        engine     : Kullanilacak OCR motoru (ornek: 'tesseract')
        model_name : Model versiyonu (ornek: 'model_v1')
        original_name : Gorselin orijinal dosya adi (str, opsiyonel)

    Doner:
        OCR sonuc sozlugu (words, confidence, field_results ...)
        veya hata durumunda {"error": "aciklama"} sozlugu.
    """
    if engine not in ENGINES:
        return {"error": f"Engine '{engine}' kayitli degil. Gecerli motorlar: {list(ENGINES.keys())}"}

    engine_spec = ENGINES[engine]
    run_function = engine_spec["run_function"]
    loader       = engine_spec["loader"]
    shared_kwargs_map = engine_spec.get("shared_kwargs", {"model": "model"})

    config_path = CONFIGS_DIR / engine / f"{model_name}.yaml"
    if not config_path.exists():
        return {"error": f"Konfigurasyon dosyasi bulunamadi: {config_path}"}

    # Modeli yukle (once)
    shared_objects = {}
    shared_load_time = 0.0
    if loader is not None:
        load_start = time.time()
        try:
            loaded = loader(str(config_path))
        except Exception as e:
            return {"error": f"Model yuklenemedi: {e}"}
        shared_load_time = round(time.time() - load_start, 4)
        for loaded_key, kwarg_name in shared_kwargs_map.items():
            shared_objects[kwarg_name] = loaded[loaded_key]

    # OCR calistir
    try:
        output_data = run_function(str(img_path), str(config_path), **shared_objects)
    except Exception as e:
        return {"error": f"OCR islemi basarisiz: {e}"}

    img_name = original_name if original_name else Path(img_path).name

    # Ground truth (varsa)
    ground_truth_data, has_ground_truth = load_ground_truth(img_name)
    output_data["ground_truth"]     = ground_truth_data
    output_data["has_ground_truth"] = has_ground_truth

    fields = (ground_truth_data or {}).get("fields", {})

    # Kelimeleri field eslesmesiyle zenginlestir
    if "words" in output_data:
        output_data["words"] = enrich_words_with_field_matches_lcs(
            output_data["words"], fields
        )

    # Alan bazli dogruluk
    field_check = check_all_fields_lcs_cer_with_bbox(fields, output_data.get("words", []))
    output_data["fields_found"]       = field_check["fields_found"]
    output_data["fields_total"]       = field_check["fields_total"]
    output_data["field_match_ratio"]  = (
        round(field_check["fields_found"] / field_check["fields_total"], 4)
        if field_check["fields_total"] else None
    )
    output_data["field_results"]      = field_check["field_results"]

    # Ortak alan kontrolu
    # 1. Once dosya adina gore tespit dene (mevcut davranis)
    common_fields_file = detect_common_fields_file(img_name, COMMON_FIELDS_DIR)
    # 2. Bulunamazsa (dosya adi belge turunu yansitmiyorsa), OCR metnine
    #    bakarak hangi _c.txt dosyasinin en cok kapsadigini bul
    if common_fields_file is None:
        common_fields_file = detect_common_fields_by_content(
            output_data.get("words", []), COMMON_FIELDS_DIR
        )
    common_fields      = load_common_fields(common_fields_file) if common_fields_file else {}
    common_field_check = check_all_fields_lcs_cer_with_bbox(
        common_fields, output_data.get("words", [])
    )
    output_data["common_fields_found"]       = common_field_check["fields_found"]
    output_data["common_fields_total"]       = common_field_check["fields_total"]
    output_data["common_field_match_ratio"]  = (
        round(common_field_check["fields_found"] / common_field_check["fields_total"], 4)
        if common_field_check["fields_total"] else None
    )
    output_data["common_field_results"]  = common_field_check["field_results"]
    output_data["common_fields_source"]  = (
        str(common_fields_file) if common_fields_file else None
    )
    output_data["load_time_seconds"]     = shared_load_time
    output_data["load_time_is_shared"]   = loader is not None

    return output_data


def process_pipeline(
    engine: str,
    model_name: str,
    image_files: list[str] | None = None,
    output_subdir: str | None = None,
) -> None:
    """
    engine, model_name ile OCR pipeline'ini calistirir.

    Parametreler:
        engine       : OCR motoru adi
        model_name   : Konfigürasyon adi
        image_files  : Islenecek gorsel yollarinin listesi. None ise
                       IMAGE_DIR altindaki tum gorseller otomatik alinir.
        output_subdir: outputs/ altinda olusturulacak alt klasor adi
                       (ornek: 'custom_run_1720000000'). None ise
                       klasik '<engine>/<model_name>' yolu kullanilir.
    """
    if engine not in ENGINES:
        print(f"Error: '{engine}' is not registered in registry.py.")
        print(f"Registered engines: {list(ENGINES.keys())}")
        return

    engine_spec = ENGINES[engine]
    run_function = engine_spec["run_function"]
    loader = engine_spec["loader"]

    if output_subdir:
        output_dir = OUTPUTS_DIR / output_subdir / engine / model_name
    else:
        output_dir = OUTPUTS_DIR / engine / model_name
    config_path = CONFIGS_DIR / engine / f"{model_name}.yaml"

    output_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Load the model/reader/processor ONCE, share it across all images
    # =========================================================================
    # If `loader` is set, it's called a single time here, and the resulting
    # object is passed into every run_function call as the `model` kwarg.
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
        shared_kwargs_map = engine_spec.get("shared_kwargs", {"model": "model"})
        for loaded_key, kwarg_name in shared_kwargs_map.items():
            shared_objects[kwarg_name] = loaded[loaded_key]
        print(f"[{engine.upper()}] Model loaded ({shared_load_time}s), processing images.")

    # image_files parametresi verilmisse o listeyi kullan (tam yollar);
    # verilmemisse IMAGE_DIR altindaki tum gorselleri otomatik tara.
    if image_files is not None:
        resolved_images = [(Path(p).name, str(p)) for p in image_files]
    else:
        resolved_images = [
            (f, str(IMAGE_DIR / f))
            for f in sorted(os.listdir(IMAGE_DIR))
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

    for img_name, img_path in resolved_images:

        try:
            output_data = run_function(str(img_path), str(config_path), **shared_objects)
        except Exception as e:
            # A failure on one image (corrupt file, unsupported format, etc.)
            # is reported and skipped — the batch continues with the rest.
            print(f"[{engine.upper()} - {model_name}] ERROR: failed to process {img_name} — {e}")
            continue

        ground_truth_data, has_ground_truth = load_ground_truth(img_name)
        output_data["ground_truth"] = ground_truth_data
        output_data["has_ground_truth"] = has_ground_truth

        # =====================================================================
        # Field-based accuracy (if ground truth exists)
        # =====================================================================
        # ground_truth_data is the FULL parsed content of inputs/truths/*.yaml
        # — shaped as {"fields": {...}}. ALL accuracy checking now goes
        # through lcs_cer.py (longest-common-substring alignment + Levenshtein
        # based CER/WER); the old fuzzy-based check_all_fields in accuracy.py
        # is NO LONGER USED. check_all_fields_lcs_cer_with_bbox operates on
        # the "words" list (boxes with bboxes), so each field result also
        # carries its OWN pixel location (bbox/bboxes) — a web frontend can
        # read this straight from the JSON without recomputing anything.
        #
        # If fields is empty/missing (e.g. the .yaml file exists but has no
        # "fields" block), check_all_fields_lcs_cer_with_bbox already returns
        # safe empty/None values — no extra check needed here.
        fields = (ground_truth_data or {}).get("fields", {})

        # 1) Enrich every "words" box with which ground-truth field it best
        # matches — a web frontend can read this ready-made data on hover
        # instead of computing it itself. Now LCS/CER based (see
        # lcs_cer.py); the old fuzzy-based enrich_words_with_field_matches
        # in accuracy.py is NO LONGER USED.
        if "words" in output_data:
            output_data["words"] = enrich_words_with_field_matches_lcs(
                output_data["words"], fields
            )

        # 2) Document-level summary: how many fields were found out of the
        # total, now via the LCS/CER method (see lcs_cer.py). Each entry in
        # field_results also carries matched_substring, cer, wer, bbox,
        # bboxes, and matched_word_indices.
        field_check = check_all_fields_lcs_cer_with_bbox(fields, output_data.get("words", []))
        output_data["fields_found"] = field_check["fields_found"]
        output_data["fields_total"] = field_check["fields_total"]
        output_data["field_match_ratio"] = (
            round(field_check["fields_found"] / field_check["fields_total"], 4)
            if field_check["fields_total"] else None
        )
        # We also keep field_results — a detailed record of which fields
        # were found or not, what text they matched, AND at which pixel
        # location (usable in a web frontend's side panel for a per-field
        # list + drawing boxes on the image).
        output_data["field_results"] = field_check["field_results"]

        # =====================================================================
        # Common (fixed/shared) word check — the PRIMARY check
        # =====================================================================
        # field_results above checks document-SPECIFIC information (name,
        # date, amount) — these CHANGE on every image. This check instead
        # looks for the fixed words that NEVER change within a document type
        # (e.g. "RECEIPT", "DRIVER'S LICENSE") — read from
        # inputs/truths/common_fields/<document_type>.txt, selected
        # AUTOMATICALLY based on the image filename.
        #
        # The two checks are PARALLEL and INDEPENDENT of each other — neither
        # replaces the other, and both are written to the JSON as separate
        # fields. BOTH now use the SAME (LCS/CER + bbox) method — the old
        # fuzzy-based check_all_fields in accuracy.py and the old (bbox-less)
        # check_all_fields_lcs_cer in lcs_cer.py are NO LONGER USED.
        common_fields_file = detect_common_fields_file(img_name, COMMON_FIELDS_DIR)
        common_fields = load_common_fields(common_fields_file) if common_fields_file else {}

        common_field_check = check_all_fields_lcs_cer_with_bbox(
            common_fields, output_data.get("words", [])
        )
        output_data["common_fields_found"] = common_field_check["fields_found"]
        output_data["common_fields_total"] = common_field_check["fields_total"]
        output_data["common_field_match_ratio"] = (
            round(common_field_check["fields_found"] / common_field_check["fields_total"], 4)
            if common_field_check["fields_total"] else None
        )
        output_data["common_field_results"] = common_field_check["field_results"]
        # We also keep which .txt file was used — useful for debugging, and
        # for a web frontend that wants to show "this image was detected as
        # this document type".
        output_data["common_fields_source"] = (
            str(common_fields_file) if common_fields_file else None
        )

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
            f"GT: {has_ground_truth} | "
            f"Fields: {output_data['fields_found']}/{output_data['fields_total']} | "
            f"Common: {output_data['common_fields_found']}/{output_data['common_fields_total']}"
        )

    PROCESSED_MODELS.append((engine, model_name))


if __name__ == "__main__":
    # Physically delete all previous outputs (clean slate)
    if OUTPUTS_DIR.exists():
        print(f"[CLEANUP] Deleting outputs/ directory entirely...")
        shutil.rmtree(OUTPUTS_DIR)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    process_pipeline("tesseract", "model_v1")

    # Automatically generate the comparison report once all pipelines are done
    print("\n[REPORT] Generating comparison report...")
    generate_report(
        outputs_dir=str(OUTPUTS_DIR),
        common_fields_dir=str(COMMON_FIELDS_DIR),
        models_to_process=PROCESSED_MODELS,
    )
