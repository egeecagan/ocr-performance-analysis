import os
import json
import time
import shutil
from pathlib import Path
import yaml
from runners.registry import ENGINES
from runners.accuracy import (
    detect_common_fields_file,
    load_common_fields,
)
from runners.lcs_cer import check_all_fields_lcs_cer_with_bbox, enrich_words_with_field_matches_lcs
from generate_report import generate_report

# =============================================================================
# Base paths
# =============================================================================
# All paths are resolved relative to this file's own location, so main.py
# works correctly no matter which directory you run it from. It expects:
#   inputs/images/                      - input images
#   inputs/truths/                      - optional ground-truth .yaml files
#   configurations/<engine>/<name>.yaml - per-engine config files
#   outputs/                            - created automatically if missing
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent

IMAGE_DIR = BASE_DIR / "inputs" / "images"
TRUTHS_DIR = BASE_DIR / "inputs" / "truths"
COMMON_FIELDS_DIR = BASE_DIR / "inputs" / "truths" / "common_fields"
OUTPUTS_DIR = BASE_DIR / "outputs"
CONFIGS_DIR = BASE_DIR / "configurations"

PROCESSED_MODELS = []


def load_ground_truth(img_name, truths_dir=TRUTHS_DIR):
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

            # if data is none then it returns empty dictionary!!!
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

        ground_truth_data, has_ground_truth = load_ground_truth(img_name)
        output_data["ground_truth"] = ground_truth_data
        output_data["has_ground_truth"] = has_ground_truth

        # =====================================================================
        # Field-based accuracy (ground truth varsa)
        # =====================================================================
        # ground_truth_data, inputs/truths/*.yaml dosyasının TAMAMI —
        # {"fields": {...}} şeklinde. TÜM doğruluk kontrolü artık lcs_cer.py
        # (longest-common-substring konumlandırma + Levenshtein tabanlı
        # CER/WER) üzerinden yapılıyor — accuracy.py'deki eski fuzzy tabanlı
        # check_all_fields ARTIK KULLANILMIYOR. check_all_fields_lcs_cer_
        # with_bbox, "words" listesi (bbox'lı kutular) üzerinde çalıştığı
        # için, her alan sonucu KENDİ piksel konumunu (bbox/bboxes) da
        # taşıyor — web arayüzü bunu doğrudan JSON'dan okuyabilir, ayrıca
        # hesaplama YAPMASI gerekmiyor.
        #
        # fields boş/yoksa (örn. .yaml dosyası var ama "fields" bloğu
        # tanımlı değilse), check_all_fields_lcs_cer_with_bbox zaten boş/
        # None değerlerle güvenli şekilde dönüyor — burada ekstra bir
        # kontrol gerekmiyor.
        fields = (ground_truth_data or {}).get("fields", {})

        # 1) Her "words" kutusunu, ground truth'taki HANGİ alana en çok
        # benzediği bilgisiyle zenginleştir — web arayüzü mouse hover'da
        # bu hazır veriyi okuyacak, kendi hesaplama YAPMAYACAK. ARTIK
        # LCS/CER tabanlı (bkz. lcs_cer.py) — accuracy.py'deki eski fuzzy
        # tabanlı enrich_words_with_field_matches ARTIK KULLANILMIYOR.
        if "words" in output_data:
            output_data["words"] = enrich_words_with_field_matches_lcs(
                output_data["words"], fields
            )

        # 2) Belge genelinde özet: kaç alan bulundu / toplam kaç alan,
        # ARTIK LCS/CER yöntemiyle (bkz. lcs_cer.py). field_results
        # içindeki HER alan artık matched_substring, cer, wer, bbox,
        # bboxes, matched_word_indices alanlarını da taşıyor.
        field_check = check_all_fields_lcs_cer_with_bbox(fields, output_data.get("words", []))
        output_data["fields_found"] = field_check["fields_found"]
        output_data["fields_total"] = field_check["fields_total"]
        output_data["field_match_ratio"] = (
            round(field_check["fields_found"] / field_check["fields_total"], 4)
            if field_check["fields_total"] else None
        )
        # field_results'ı da saklıyoruz — hangi alanın bulunup
        # bulunmadığını, hangi metinle eşleştiğini VE HANGİ PİKSELDE
        # olduğunu ayrıntılı görmek için (web arayüzünde sağ panelde,
        # alan-bazlı bir liste + görsel üzerine kutu çizmek için
        # kullanılabilir).
        output_data["field_results"] = field_check["field_results"]

        # =====================================================================
        # Common (sabit/ortak) kelime kontrolü — ASIL ÖNCELİKLİ kontrol
        # =====================================================================
        # Yukarıdaki field_results, belgeye ÖZEL bilgileri (ad, tarih,
        # tutar) kontrol ediyor — bunlar her görselde DEĞİŞİR. Buradaki
        # kontrol ise her belge tipinde (örn. her dekontta, her ehliyette)
        # HİÇ DEĞİŞMEYEN sabit kelimeleri (örn. "DEKONT", "SÜRÜCÜ BELGESİ")
        # kontrol ediyor — inputs/truths/common_fields/<belge_tipi>.txt
        # dosyasından okunur, görsel dosya adına göre OTOMATİK seçilir.
        #
        # İkisi PARALEL, birbirinden BAĞIMSIZ iki ayrı kontrol — biri
        # diğerinin yerini almaz, ikisi de JSON'a ayrı alanlar olarak
        # yazılır. İKİSİ DE artık AYNI (LCS/CER + bbox) yöntemi kullanıyor
        # — accuracy.py'deki eski fuzzy tabanlı check_all_fields ve
        # lcs_cer.py'deki eski (bbox'sız) check_all_fields_lcs_cer ARTIK
        # KULLANILMIYOR.
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
        # Hangi .txt dosyasının kullanıldığını da saklıyoruz — debug için
        # ve web arayüzünde "bu görsel hangi belge tipi olarak algılandı"
        # diye göstermek isterseniz işe yarar.
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
    # Eski tüm çıktıları fiziksel olarak tamamen sil (Clean slate)
    if OUTPUTS_DIR.exists():
        print(f"[CLEANUP] outputs/ klasoru tamamen siliniyor...")
        shutil.rmtree(OUTPUTS_DIR)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    process_pipeline("tesseract", "model_v1")

    # Tüm pipeline'lar bittikten sonra karşılaştırma raporunu otomatik üret
    print("\n[REPORT] Karsilastirma raporu olusturuluyor...")
    generate_report(
        outputs_dir=str(OUTPUTS_DIR),
        common_fields_dir=str(COMMON_FIELDS_DIR),
        models_to_process=PROCESSED_MODELS,
    )
