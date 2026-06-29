import os
import time
import inspect
import platform
import torch
import yaml
import cv2
import numpy as np
from pathlib import Path
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image, ImageDraw, ImageFont


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def filter_valid_kwargs(func, kwargs):
    """
    Config'ten gelen kwargs sözlüğünü, hedef fonksiyonun (örn. model.generate)
    GERÇEKTEN kabul ettiği parametrelerle sınırlar.

    Not: model.generate(**model_kwargs) gibi bazı HF fonksiyonları **kwargs
    ile serbest argüman da kabul edebiliyor, bu durumda inspect.signature
    VAR_KEYWORD parametresini görür ve her şeyi geçerli sayar — bu istenen
    davranış, çünkü generate() çoğu zaman generation_config üzerinden de
    parametre kabul eder.
    """
    sig = inspect.signature(func)
    params = sig.parameters

    # Eğer fonksiyon **kwargs kabul ediyorsa (VAR_KEYWORD), her şeyi geçir
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)

    valid_params = set(params.keys())
    return {k: v for k, v in kwargs.items() if k in valid_params}


def resolve_font_path(config_font_path=None):
    """
    Masked görselde metni yazdırmak için gerçek bir TrueType font bulur.
    ImageFont.load_default() Türkçe karakterleri (ş, ğ, ı, ö, ü, ç) düzgün
    render etmediği için tercih edilmez.
    """
    if config_font_path and os.path.exists(config_font_path):
        return config_font_path

    system = platform.system()
    candidates = []
    if system == "Darwin":
        candidates = ["/Library/Fonts/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf"]
    elif system == "Linux":
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]
    elif system == "Windows":
        candidates = ["C:\\Windows\\Fonts\\arial.ttf"]

    for path in candidates:
        if os.path.exists(path):
            return path

    return None


def run_trocr(image_path, config_path, processor=None, model=None):
    config = load_config(config_path)
    settings = config.get("ocr_settings", {})
    generate_settings = config.get("generate_settings", {}) or {}

    model_name = settings.get("model_name", "microsoft/trocr-base-handwritten")
    font_path_config = settings.get("font_path")
    font_size = settings.get("font_size", 24)

    # Modelleri yükle — dışarıdan geçirilmediyse burada bir kere yüklenir (süreye dahil değil)
    if processor is None:
        processor = TrOCRProcessor.from_pretrained(model_name)
    if model is None:
        model = VisionEncoderDecoderModel.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Config'teki TÜM generate_settings anahtarlarını, model.generate'in kabul
    # ettiği gerçek parametrelerle filtreleyip otomatik geçiriyoruz.
    # Yeni bir parametre eklemek istediğinde SADECE YAML dosyasını
    # değiştirmen yeterli — bu fonksiyona dokunmana gerek kalmaz.
    valid_generate_kwargs = filter_valid_kwargs(model.generate, generate_settings)

    start_time = time.time()  # Süre BURADA başlar

    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

    generated_ids = model.generate(pixel_values, **valid_generate_kwargs)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    execution_time = round(time.time() - start_time, 4)  # Süre BURADA donar

    # --- Buradan sonrası süreye dahil değil: görselleştirme/raporlama ---

    # viz klasörünü config_path'ten türet: configurations/trocr/model_v1.yaml
    # -> outputs/trocr/model_v1/viz/...
    config_p = Path(config_path)
    engine = config_p.parent.name        # "trocr"
    config_model_name = config_p.stem     # "model_v1"

    viz_dir = os.path.join('outputs', engine, config_model_name, 'viz')
    highlighted_dir = os.path.join(viz_dir, 'highlighted')
    masked_dir = os.path.join(viz_dir, 'masked')
    os.makedirs(highlighted_dir, exist_ok=True)
    os.makedirs(masked_dir, exist_ok=True)

    # --- Masked: beyaz sayfaya metni yaz (gerçek font + otomatik satır kaydırma) ---
    font_path = resolve_font_path(font_path_config)
    try:
        font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    masked_img = Image.new("RGB", image.size, (255, 255, 255))
    draw = ImageDraw.Draw(masked_img)

    # Basit kelime-bazlı satır kaydırma — metin görsel genişliğini taşmasın
    margin = 10
    max_width = image.size[0] - 2 * margin
    words = text.split()
    lines, current_line = [], ""

    for word in words:
        trial = f"{current_line} {word}".strip()
        if draw.textlength(trial, font=font) <= max_width:
            current_line = trial
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)

    y = margin
    line_height = font_size + 6
    for line in lines:
        draw.text((margin, y), line, font=font, fill=(0, 0, 0))
        y += line_height

    # --- Highlighted: TrOCR'da bbox olmadığı için tüm görsele GERÇEK şeffaf kırmızı katman ---
    # PIL'in RGB modunda fill alpha'sı yok sayılır; bu yüzden OpenCV ile addWeighted kullanıyoruz
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    overlay = np.full_like(img_cv, (0, 0, 255))  # tamamen kırmızı katman
    alpha = 0.3
    highlighted_img = cv2.addWeighted(overlay, alpha, img_cv, 1 - alpha, 0)

    # Kayıtlar
    img_name = os.path.basename(image_path)
    masked_img.save(os.path.join(masked_dir, f"masked_{img_name}"))
    cv2.imwrite(os.path.join(highlighted_dir, f"highlighted_{img_name}"), highlighted_img)

    return {
        "text": text,
        "execution_time_seconds": execution_time,
        "model_used": model_name,
        "settings_used": valid_generate_kwargs,  # Hangi generate parametreleriyle çalıştığını görmek için
    }


if __name__ == "__main__":
    # main.py'nin çağırdığı gibi: config_path = f'configurations/{engine}/{model_name}.yaml'
    result = run_trocr(
        image_path="inputs/images/ornek.png",
        config_path="configurations/trocr/model_v1.yaml",
    )
    print(result)