import os
import json
from pathlib import Path

from runners.run_tesseract import run_tesseract
from runners.run_easyocr import run_easyocr
from runners.run_trocr import run_trocr
from runners.run_doctr import run_doctr


def load_ground_truth(img_name, truths_dir='inputs/truths'):
    """
    Görsel adıyla eşleşen ground truth .txt dosyasını okur.
    Örnek: resim1.png -> inputs/truths/resim1.txt

    Dosya varsa: (metin_içeriği, True)
    Dosya yoksa: (None, False)
    """
    truth_path = os.path.join(truths_dir, f"{Path(img_name).stem}.txt")

    if os.path.exists(truth_path):
        with open(truth_path, 'r', encoding='utf-8') as f:
            ground_truth_text = f.read().strip()
        return ground_truth_text, True

    return None, False


def process_pipeline(engine, model_name):

    BASE_DIR = Path(__file__).resolve().parent

    image_dir = BASE_DIR / 'inputs' / 'images'
    truths_dir = BASE_DIR / 'inputs' / 'truths'
    output_dir = BASE_DIR / 'outputs' / engine / model_name
    config_path = BASE_DIR / 'configurations' / engine / f'{model_name}.yaml'
    
    os.makedirs(output_dir, exist_ok=True)

    for img_name in os.listdir(image_dir):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(image_dir, img_name)

            if engine == 'tesseract':
                output_data = run_tesseract(img_path, config_path)
            elif engine == 'easyocr':
                output_data = run_easyocr(img_path, config_path)
            elif engine == 'trocr':
                output_data = run_trocr(img_path, config_path)
            elif engine == 'doctr':
                output_data = run_doctr(img_path, config_path)
            else:
                print(f"Hata: {engine} motoru bulunamadı.")
                return

            # Ground truth'u oku ve JSON'a ekle
            ground_truth_text, has_ground_truth = load_ground_truth(img_name, truths_dir)
            output_data['ground_truth'] = ground_truth_text
            output_data['has_ground_truth'] = has_ground_truth

            output_file = os.path.join(output_dir, f"{Path(img_name).stem}.json")

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=4)

            print(f"[{engine.upper()} - {model_name}] İşlendi: {img_name} | Süre: {output_data.get('execution_time_seconds')} sn | GT: {has_ground_truth}")


if __name__ == "__main__":
    process_pipeline('easyocr', 'model_v1')
    process_pipeline('tesseract', 'model_v1')