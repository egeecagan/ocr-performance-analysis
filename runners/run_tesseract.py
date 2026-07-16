import os
import time
import cv2
import numpy as np
import pytesseract
from pytesseract import Output
from PIL import Image, ImageDraw, ImageFont

from runners._common import load_config, resolve_font_path, preprocess_image, get_viz_dirs, save_preprocessed_image


def build_tesseract_config(ocr_settings):
    """
    Config dosyasındaki TÜM ocr_settings.extra_params anahtarlarını otomatik
    olarak '-c key=value' şeklinde Tesseract config string'ine ekler.

    Yeni bir parametre eklemek/çıkarmak istediğinde SADECE YAML dosyasını
    değiştirmen yeterli — bu fonksiyona dokunmana gerek kalmaz.

    Not: Bu fonksiyon Tesseract'a ÖZELDİR (--psm/--oem/-c sentaksı sadece
    Tesseract CLI'a ait), bu yüzden _common.py'ye taşınmadı, burada kalıyor.
    """
    psm = ocr_settings.get("psm", 6)
    oem = ocr_settings.get("oem", 3)
    dpi = ocr_settings.get("dpi")

    config_parts = [f"--psm {psm}", f"--oem {oem}"]

    if dpi:
        config_parts.append(f"--dpi {dpi}")

    extra_params = ocr_settings.get("extra_params", {}) or {}

    for key, value in extra_params.items():
        if value is None or value == "":
            continue
        config_parts.append(f'-c {key}={value}')

    return " ".join(config_parts)


def run_tesseract(image_path, config_path):
    config = load_config(config_path)
    ocr_settings = config.get("ocr_settings", {})
    preprocessing_settings = config.get("preprocessing", {}) or {}

    lang = ocr_settings.get("lang", "tur+eng")
    font_path_config = ocr_settings.get("font_path")
    tess_config = build_tesseract_config(ocr_settings)

    timeout = ocr_settings.get("timeout", 0)

    image_load_start = time.time()
    img = cv2.imread(image_path)
    image_load_time = round(time.time() - image_load_start, 4)

    if img is None:
        raise ValueError(
            f"Görsel okunamadı (bozuk dosya, desteklenmeyen format ya da "
            f"yol hatalı olabilir): {image_path}"
        )

    load_time = 0.0
    preprocessing_start = time.time()
    ocr_input = preprocess_image(img, preprocessing_settings)
    preprocessing_time = round(time.time() - preprocessing_start, 4)

    start_time = time.time()

    try:
        d = pytesseract.image_to_data(
            ocr_input, lang=lang, config=tess_config, output_type=Output.DICT, timeout=timeout,
        )
    except RuntimeError as e:
        raise RuntimeError(
            f"Tesseract zaman aşımına uğradı (timeout={timeout}s): {image_path}. "
            f"Orijinal hata: {e}"
        ) from e

    execution_time = round(time.time() - start_time, 4) 

    highlighted_dir, masked_dir, preprocessed_dir = get_viz_dirs(config_path)

    save_preprocessed_image(ocr_input, preprocessed_dir, os.path.basename(image_path))

    overlay = img.copy()
    pil_img = Image.new("RGB", (img.shape[1], img.shape[0]), (255, 255, 255))
    draw = ImageDraw.Draw(pil_img)

    font_path = resolve_font_path(font_path_config)

    n_boxes = len(d['text'])
    recognized_words = []
    confidences = []
    words = [] 

    for i in range(n_boxes):
        if int(d['conf'][i]) > 0 and d['text'][i].strip() != "":
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            text_found = d['text'][i]
            word_confidence = int(d['conf'][i])
            recognized_words.append(text_found)
            confidences.append(word_confidence)
            words.append({
                "text": text_found,
                "bbox": [int(x), int(y), int(x + w), int(y + h)],
                "confidence": round(word_confidence, 2),
                "line_id": f"{d['block_num'][i]}-{d['par_num'][i]}-{d['line_num'][i]}",
            })

            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), -1)

            font_size = max(int(h * 0.8), 1)
            try:
                font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
            except Exception:
                font = ImageFont.load_default()

            draw.text((x, y), text_found, font=font, fill=(0, 0, 0))

    avg_confidence = round(sum(confidences) / len(confidences), 2) if confidences else 0.0

    img_name = os.path.basename(image_path)

    alpha = 0.3
    highlighted_img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    cv2.imwrite(os.path.join(highlighted_dir, f"highlighted_{img_name}"), highlighted_img)

    masked_final = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(masked_dir, f"masked_{img_name}"), masked_final)

    total_time = round(image_load_time + preprocessing_time + execution_time, 4)

    return {
        "text": " ".join(recognized_words),
        "words": words,
        "load_time_seconds": load_time,
        "image_load_time_seconds": image_load_time,
        "preprocessing_time_seconds": preprocessing_time,
        "execution_time_seconds": execution_time,
        "total_time_seconds": total_time,
        "avg_confidence": avg_confidence,
        # Hangi dil ayarıyla çalıştığı artık model_used'da görünüyor.
        "model_used": f"tesseract-{lang}",
        "settings_used": tess_config,
        "preprocessing_used": preprocessing_settings,
    }


if __name__ == "__main__":
    # main.py'nin çağırdığı gibi: config_path = f'configurations/{engine}/{model_name}.yaml'
    result = run_tesseract(
        image_path="inputs/images/ornek.png",
        config_path="configurations/tesseract/model_v1.yaml",
    )
    print(result)