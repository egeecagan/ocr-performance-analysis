import os
import time
import cv2
import numpy as np
import pytesseract
from pytesseract import Output
from PIL import Image, ImageDraw, ImageFont

from runners._common import load_config, resolve_font_path, preprocess_image, get_viz_dirs


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
        # Boş string veya None olan değerleri atla (kullanıcı "kapalı" demek istemiştir)
        if value is None or value == "":
            continue
        config_parts.append(f'-c {key}={value}')

    return " ".join(config_parts)


def run_tesseract(image_path, config_path):
    config = load_config(config_path)
    ocr_settings = config.get("ocr_settings", {})
    preprocessing_settings = config.get("preprocessing", {}) or {}

    lang = ocr_settings.get("lang", "tur+eng")
    font_path_config = ocr_settings.get("font_path")  # opsiyonel, extra_params'a da eklenebilir
    tess_config = build_tesseract_config(ocr_settings)

    img = cv2.imread(image_path)

    if img is None:
        # cv2.imread, dosya bulunamadığında veya bozuk/okunamaz olduğunda
        # hata FIRLATMAZ, sessizce None döndürür. Kontrol etmezsek bir
        # sonraki satırda (img.shape) anlamsız bir AttributeError ile
        # patlar ve 20 görsellik toplu işlemde HANGİ dosyanın sorunlu
        # olduğunu anlamak zorlaşır. Burada hatayı net bir mesajla,
        # dosya yolunu göstererek fırlatıyoruz.
        raise ValueError(
            f"Görsel okunamadı (bozuk dosya, desteklenmeyen format ya da "
            f"yol hatalı olabilir): {image_path}"
        )

    # Tesseract'ta EasyOCR/TrOCR/doctr'daki gibi ayrı bir "model yükleme"
    # adımı yok (binary her çağrıda hafifçe başlatılıyor, maliyeti ihmal
    # edilebilir düzeyde). 4 motorun JSON şeması tutarlı olsun diye bu alan
    # sabit 0.0 olarak ekleniyor.
    load_time = 0.0

    # Preprocessing artık ortak _common.preprocess_image() üzerinden çalışır.
    # Config'te kapalıysa img ham haliyle döner, davranış değişmez.
    ocr_input = preprocess_image(img, preprocessing_settings)

    start_time = time.time()  # Süre BURADA başlar — preprocessing süreye dahil
    # DEĞİL, çünkü ölçmek istediğimiz "OCR motorunun kendisi ne kadar sürdü"

    # Tek OCR çağrısı — hem text hem layout (bbox/conf) bundan geliyor
    d = pytesseract.image_to_data(ocr_input, lang=lang, config=tess_config, output_type=Output.DICT)

    execution_time = round(time.time() - start_time, 4)  # Süre BURADA donar

    # --- Buradan sonrası süreye dahil değil: görselleştirme/raporlama ---
    highlighted_dir, masked_dir = get_viz_dirs(config_path)

    # Highlight/mask çizimleri orijinal RENKLİ görsel üzerine yapılır —
    # preprocessing sadece OCR motoruna giden girdiyi etkiler, görselleştirme
    # her zaman okunabilir/renkli orijinal üzerinden üretilir.
    overlay = img.copy()
    pil_img = Image.new("RGB", (img.shape[1], img.shape[0]), (255, 255, 255))
    draw = ImageDraw.Draw(pil_img)

    font_path = resolve_font_path(font_path_config)

    n_boxes = len(d['text'])
    recognized_words = []
    confidences = []

    for i in range(n_boxes):
        if int(d['conf'][i]) > 0 and d['text'][i].strip() != "":
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            text_found = d['text'][i]
            recognized_words.append(text_found)
            confidences.append(int(d['conf'][i]))

            # --- Highlight (Kırmızı kutular) ---
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), -1)

            # --- Masked (PIL ile metin basımı) ---
            font_size = max(int(h * 0.8), 1)
            try:
                font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
            except Exception:
                font = ImageFont.load_default()

            draw.text((x, y), text_found, font=font, fill=(0, 0, 0))

    # Tesseract'ın kendi tahminlerine ortalama güveni (0-100). Ground truth
    # olmayan görsellerde bile "model ne kadar eminmiş" sorusuna cevap verir.
    avg_confidence = round(sum(confidences) / len(confidences), 2) if confidences else 0.0

    # Kayıtlar
    img_name = os.path.basename(image_path)

    alpha = 0.3
    highlighted_img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    cv2.imwrite(os.path.join(highlighted_dir, f"highlighted_{img_name}"), highlighted_img)

    masked_final = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(masked_dir, f"masked_{img_name}"), masked_final)

    return {
        "text": " ".join(recognized_words),
        "load_time_seconds": load_time,
        "execution_time_seconds": execution_time,
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