"""
docker_run_paddleocr.py

Container İÇİNDE çalışır. main.py'deki process_pipeline() mantığının
SADECE PaddleOCR'a özel, bağımsız bir kopyasıdır — registry.py'ye veya
diğer 5 motora hiç bağımlı değildir, çünkü container'a sadece
run_paddleocr.py + _common.py kopyalanıyor.

Mac'teki main.py'nin ürettiğiyle BİREBİR AYNI klasör yapısına ve JSON
şemasına yazar:
    outputs/paddleocr/{model_name}/{görsel_adı}.json
    outputs/paddleocr/{model_name}/viz/{highlighted,masked,preprocessed}/

Volume mount sayesinde (docker run -v ile), bu container'ın yazdığı
outputs/ klasörü, Mac'teki gerçek outputs/ klasörünün AYNISI olur — diğer
5 motorun ürettiği JSON'larla aynı yerde durur, main.py'nin sonradan
okuyacağı format farketmez container'da mı Mac'te mi üretildiği.
"""

import os
import json
import sys
from pathlib import Path

from runners.run_paddleocr import run_paddleocr

# Container içinde sabit yollar — docker run'da bu yollara volume mount
# yapılacak (bkz. README/komut örneği).
BASE_DIR = Path("/data")
IMAGE_DIR = BASE_DIR / "inputs" / "images"
TRUTHS_DIR = BASE_DIR / "inputs" / "truths"
OUTPUTS_DIR = BASE_DIR / "outputs"
CONFIGS_DIR = BASE_DIR / "configurations"


def load_ground_truth(img_name, truths_dir=TRUTHS_DIR):
    truth_path = Path(truths_dir) / f"{Path(img_name).stem}.txt"
    if truth_path.exists():
        with open(truth_path, "r", encoding="utf-8") as f:
            ground_truth_text = f.read().strip()
        return ground_truth_text, True
    return None, False


def main():
    # Çalışma dizinini /data'ya taşıyoruz çünkü run_paddleocr.py içindeki
    # get_viz_dirs() fonksiyonu varsayılan olarak GÖRECELİ "outputs/" yolunu
    # kullanıyor (Mac'teki main.py'nin de BASE_DIR ile çözdüğü, ama bu
    # bağımsız script'in BASE_DIR mantığı yok). os.chdir ile, _common.py'nin
    # ürettiği göreceli yollar /data/outputs/... olarak doğru çözülür.
    os.chdir(BASE_DIR)

    model_name = sys.argv[1] if len(sys.argv) > 1 else "model_v1"

    output_dir = OUTPUTS_DIR / "paddleocr" / model_name
    config_path = CONFIGS_DIR / "paddleocr" / f"{model_name}.yaml"

    if not config_path.exists():
        print(f"HATA: config bulunamadı: {config_path}")
        print("Volume mount'ların doğru yapıldığından emin olun.")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # PaddleOCR motorunu BİR KEZ yükle, tüm görseller için paylaş — Mac'teki
    # main.py/registry.py'deki aynı prensip (model paylaşımı).
    from paddleocr import PaddleOCR
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    settings = config.get("ocr_settings", {})

    print(f"[PADDLEOCR-DOCKER] Model yükleniyor (bir kez)...")
    engine = PaddleOCR(
        lang=settings.get("lang", "tr"),
        use_doc_orientation_classify=settings.get("use_doc_orientation_classify", False),
        use_doc_unwarping=settings.get("use_doc_unwarping", False),
        use_textline_orientation=settings.get("use_textline_orientation", False),
        enable_mkldnn=False,
    )
    print(f"[PADDLEOCR-DOCKER] Model yüklendi, görseller işleniyor.")

    image_files = sorted(
        f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )

    if not image_files:
        print(f"UYARI: {IMAGE_DIR} içinde görsel bulunamadı. Volume mount'u kontrol edin.")

    for img_name in image_files:
        img_path = IMAGE_DIR / img_name

        try:
            output_data = run_paddleocr(str(img_path), str(config_path), engine=engine)
        except Exception as e:
            print(f"[PADDLEOCR-DOCKER] HATA: {img_name} işlenemedi — {e}")
            continue

        ground_truth_text, has_ground_truth = load_ground_truth(img_name)
        output_data["ground_truth"] = ground_truth_text
        output_data["has_ground_truth"] = has_ground_truth

        output_file = output_dir / f"{Path(img_name).stem}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)

        print(
            f"[PADDLEOCR-DOCKER] İşlendi: {img_name} | "
            f"Süre: {output_data.get('execution_time_seconds')} sn | "
            f"GT: {has_ground_truth}"
        )

    print("[PADDLEOCR-DOCKER] Tamamlandı.")


if __name__ == "__main__":
    main()