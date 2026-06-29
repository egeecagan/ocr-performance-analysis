import os
import json
import time
from pathlib import Path

from runners.registry import ENGINES

# =============================================================================
# Mutlak yol temeli
# =============================================================================
# main.py'nin KENDİ konumuna göre tüm yolları hesaplıyoruz. main.py'yi
# nereden çalıştırırsanız çalıştırın, her zaman kendi yanındaki
# inputs/configurations/outputs klasörlerini bulur.
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent

IMAGE_DIR = BASE_DIR / "inputs" / "images"
TRUTHS_DIR = BASE_DIR / "inputs" / "truths"
OUTPUTS_DIR = BASE_DIR / "outputs"
CONFIGS_DIR = BASE_DIR / "configurations"


def load_ground_truth(img_name, truths_dir=TRUTHS_DIR):
    """
    Görsel adıyla eşleşen ground truth .txt dosyasını okur.
    Örnek: resim1.png -> inputs/truths/resim1.txt

    Dosya varsa: (metin_içeriği, True)
    Dosya yoksa: (None, False)
    """
    truth_path = Path(truths_dir) / f"{Path(img_name).stem}.txt"

    if truth_path.exists():
        with open(truth_path, "r", encoding="utf-8") as f:
            ground_truth_text = f.read().strip()
        return ground_truth_text, True

    return None, False


def process_pipeline(engine, model_name):
    if engine not in ENGINES:
        print(f"Hata: '{engine}' motoru registry.py'de tanımlı değil.")
        print(f"Tanımlı motorlar: {list(ENGINES.keys())}")
        return

    engine_spec = ENGINES[engine]
    run_function = engine_spec["run_function"]
    loader = engine_spec["loader"]
    shared_kwargs_map = engine_spec["shared_kwargs"]

    output_dir = OUTPUTS_DIR / engine / model_name
    config_path = CONFIGS_DIR / engine / f"{model_name}.yaml"

    output_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Model/reader/processor'ı BİR KEZ yükle, tüm görseller için paylaş
    # =========================================================================
    # ÖNCEDEN: her runner'ın kendi __main__ bloğunda belirttiğimiz gibi,
    # main.py her görsel için run_easyocr/run_trocr/run_doctr çağırırken
    # model parametresini hiç vermiyordu — bu da modelin HER GÖRSEL İÇİN
    # YENİDEN yüklenmesi anlamına geliyordu (20 görsel = 20 kez yükleme,
    # TrOCR/doctr gibi ağır modellerde bu dakikalar sürebilir).
    #
    # ŞİMDİ: loader varsa (Tesseract'ta yok, gerçek bir model yükleme
    # maliyeti olmadığı için None) burada SADECE BİR KEZ çağrılıyor, dönen
    # nesne(ler) shared_kwargs eşlemesine göre run_function'a her görsel
    # çağrısında aynı şekilde geçiriliyor.
    shared_objects = {}
    shared_load_time = 0.0
    if loader is not None:
        print(f"[{engine.upper()}] Model yükleniyor (bir kez)...")
        load_start = time.time()
        try:
            loaded = loader(str(config_path))
        except Exception as e:
            # =====================================================================
            # Model yüklenemiyorsa, tüm motor için işlemi NET bir mesajla durdur
            # =====================================================================
            # ÖNCEDEN: loader() çağrısı sarılmamıştı — bir model yüklenemediğinde
            # (örn. PaddleOCR'da internet bağlantısı olmadan model indirilemediğinde)
            # çıplak bir Python traceback'i görünüyordu. Bu okunabilir olsa da,
            # "bu motoru atla, diğer motorlarla devam et" kararını vermek için
            # programatik bir sinyal yoktu, ve hangi config/motorun sorumlu
            # olduğu traceback'in içine gömülüydü.
            #
            # ŞİMDİ: hatayı yakalayıp NET bir mesajla bildiriyoruz, hangi
            # motorun yüklenemediğini söylüyoruz, ve bu motor için işlemi
            # durduruyoruz (görsel döngüsüne hiç girmiyoruz) — ama bu, başka
            # bir motoru çalıştırmaya çalıştığınızda main.py'nin kendisini
            # durdurmaz, sadece process_pipeline() bu çağrı için erken döner.
            print(
                f"[{engine.upper()}] HATA: Model/motor yüklenemedi, "
                f"'{engine}' için işlem durduruldu. Olası sebepler: "
                f"internet bağlantısı (model indirme), desteklenmeyen "
                f"dil/config ayarı, ya da eksik bağımlılık. "
                f"Config: {config_path}\n"
                f"  Orijinal hata: {e}"
            )
            return
        shared_load_time = round(time.time() - load_start, 4)
        for loaded_key, kwarg_name in shared_kwargs_map.items():
            shared_objects[kwarg_name] = loaded[loaded_key]
        print(f"[{engine.upper()}] Model yüklendi ({shared_load_time} sn), görseller işleniyor.")

    image_files = sorted(
        f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )

    for img_name in image_files:
        img_path = IMAGE_DIR / img_name

        try:
            output_data = run_function(str(img_path), str(config_path), **shared_objects)
        except Exception as e:
            # =====================================================================
            # Bir görseldeki hata TÜM pipeline'ı durdurmasın
            # =====================================================================
            # ÖNCEDEN: run_tesseract/run_easyocr/run_doctr/run_trocr içindeki
            # `raise ValueError(...)` (bozuk/okunamayan görsel) ya da başka
            # bir beklenmeyen hata, process_pipeline'ı olduğu yerde durdurup
            # KALAN TÜM GÖRSELLERİ atlıyordu — 20 görselden 1'i bozuksa,
            # kalan 19'u da işlenmiyordu.
            #
            # ŞİMDİ: hatayı yakalayıp konsola NET bir şekilde basıyoruz,
            # bu görsel için JSON üretmiyoruz, ama döngü diğer görsellerle
            # DEVAM EDİYOR.
            print(f"[{engine.upper()} - {model_name}] HATA: {img_name} işlenemedi — {e}")
            continue

        ground_truth_text, has_ground_truth = load_ground_truth(img_name)
        output_data["ground_truth"] = ground_truth_text
        output_data["has_ground_truth"] = has_ground_truth

        # =====================================================================
        # Paylaşılan model yükleme süresini JSON'a yansıt
        # =====================================================================
        # run_function (örn. run_easyocr) zaten reader/model dışarıdan
        # geldiği için "load_time_seconds": 0.0 yazmıştı — bu, O GÖRSEL
        # için doğruydu (gerçekten o çağrıda yükleme olmadı) ama gerçek
        # yükleme süresi (yukarıda shared_load_time olarak ölçtüğümüz)
        # hiçbir JSON'a yansımıyordu, kayboluyordu.
        #
        # Burada her görselin JSON'una motor için BİR KEZ ölçülmüş gerçek
        # yükleme süresini yazıyoruz. ÖNEMLİ: bu değeri 20 görsel boyunca
        # TOPLARSANIZ yükleme süresini 20 kez saymış olursunuz — oysa
        # yükleme gerçekte 1 kez oldu. Bu yüzden "load_time_is_shared": true
        # işaretini de ekliyoruz; raporlama/accuracy scripti bu motor için
        # load_time_seconds'ı yalnızca BİR KEZ (örn. ilk görselden) sayıp
        # diğerlerini atlamalı.
        output_data["load_time_seconds"] = shared_load_time
        output_data["load_time_is_shared"] = loader is not None

        output_file = output_dir / f"{Path(img_name).stem}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)

        print(
            f"[{engine.upper()} - {model_name}] İşlendi: {img_name} | "
            f"Süre: {output_data.get('execution_time_seconds')} sn | "
            f"GT: {has_ground_truth}"
        )


if __name__ == "__main__":
    #process_pipeline("paddleocr", "model_v1")
    process_pipeline("easyocr", "model_v1")
    process_pipeline("trocr", "model_v1")
    process_pipeline("tesseract", "model_v1")
    process_pipeline("doctr", "model_v1")
    process_pipeline("rapidocr", "model_v1")