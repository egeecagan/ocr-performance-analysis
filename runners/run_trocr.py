import os
import time
import cv2
import torch
import numpy as np
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image, ImageDraw, ImageFont

from runners._common import (
    load_config,
    resolve_font_path,
    filter_valid_kwargs,
    preprocess_image,
    get_viz_dirs,
    save_preprocessed_image,
)


def run_trocr(image_path, config_path, processor=None, model=None):
    config = load_config(config_path)
    settings = config.get("ocr_settings", {})
    generate_settings = config.get("generate_settings", {}) or {}
    preprocessing_settings = config.get("preprocessing", {}) or {}

    model_name = settings.get("model_name", "microsoft/trocr-base-handwritten")
    font_path_config = settings.get("font_path")
    font_size = settings.get("font_size", 24)

    # --- GPU seçimi ---
    # Diğer 3 motorla (EasyOCR, doctr) tutarlı: "auto" -> otomatik tespit,
    # true/false -> zorlama. Önceki versiyonda bu sabitti
    # (torch.cuda.is_available() direkt kullanılıyordu, config'ten
    # kontrol edilemiyordu) — artık CPU'da test etmek istediğinizde
    # GPU'nuz olsa bile zorlayabilirsiniz.
    gpu_setting = settings.get("gpu", "auto")
    if gpu_setting == "auto":
        gpu = torch.cuda.is_available()
    else:
        gpu = bool(gpu_setting) and torch.cuda.is_available()
    device = "cuda" if gpu else "cpu"

    # --- load_time_seconds ---
    # EasyOCR/doctr'daki gibi burada da GERÇEK bir maliyet var:
    # TrOCRProcessor.from_pretrained(...) ve VisionEncoderDecoderModel.
    # from_pretrained(...) modelleri diskten/internetten yükler — TrOCR
    # transformer tabanlı olduğu için bu genelde EasyOCR/doctr'dan da AĞIR
    # olabilir. processor/model dışarıdan enjekte edilmemişse (main.py her
    # görsel için yeniden oluşturuyorsa) bu maliyet her çağrıda tekrar
    # tekrar ödenir.
    needs_loading = (processor is None) or (model is None)

    if needs_loading:
        load_start = time.time()
        try:
            if processor is None:
                processor = TrOCRProcessor.from_pretrained(model_name)
            if model is None:
                model = VisionEncoderDecoderModel.from_pretrained(model_name)
        except Exception as e:
            raise RuntimeError(
                f"TrOCR modeli/processor'ı oluşturulamadı (model_name="
                f"{model_name}, config: {config_path}). Orijinal hata: {e}"
            ) from e
        model.to(device)
        load_time = round(time.time() - load_start, 4)
    else:
        # Hem processor hem model dışarıdan (main.py'de bir kez oluşturulup
        # tüm görseller için tekrar kullanılıyorsa) geldiyse, bu çağrıda
        # yükleme maliyeti yok. Cihaza taşıma da zaten önceden yapılmış
        # olmalı, burada tekrar etmiyoruz.
        load_time = 0.0

    # --- image_load_time_seconds: görseli diskten okuma süresi ---
    image_load_start = time.time()
    img_bgr = cv2.imread(image_path)
    image_load_time = round(time.time() - image_load_start, 4)

    if img_bgr is None:
        raise ValueError(
            f"Görsel okunamadı (bozuk dosya, desteklenmeyen format ya da "
            f"yol hatalı olabilir): {image_path}"
        )

    # --- preprocessing_time_seconds ---
    # Sadece preprocess_image() çağrısının kendisi ölçülüyor. PIL'e
    # dönüştürme (TrOCR'a özgü bir adım, diğer 3 motorda yok) ayrı bir
    # alanda (pil_conversion_time_seconds) tutuluyor.
    preprocessing_start = time.time()
    ocr_input = preprocess_image(img_bgr, preprocessing_settings)
    preprocessing_time = round(time.time() - preprocessing_start, 4)

    # --- Kanal güvenliği ---
    # preprocess_image bazı adımlardan sonra tek kanallı (2D) görsel
    # döndürebilir. TrOCR'ın processor'ı 3 kanallı RGB bekler — bu yüzden
    # tek kanallıysa önce 3 kanala (BGR), sonra RGB'ye çeviriyoruz.
    if len(ocr_input.shape) == 2:
        ocr_input = cv2.cvtColor(ocr_input, cv2.COLOR_GRAY2BGR)

    # TrOCR/HuggingFace processor'ı PIL Image (RGB) bekler; cv2 BGR
    # kullanır. Bu dönüşüm sadece TrOCR'a özgü (diğer 3 motorda yok), bu
    # yüzden preprocessing_time'dan AYRI, kendi alanında ölçülüyor.
    pil_conversion_start = time.time()
    image = Image.fromarray(cv2.cvtColor(ocr_input, cv2.COLOR_BGR2RGB))
    pil_conversion_time = round(time.time() - pil_conversion_start, 4)

    highlighted_dir, masked_dir, preprocessed_dir = get_viz_dirs(config_path)
    img_name = os.path.basename(image_path)

    # OCR motoruna giden görseli (preprocessing kapalıysa ham hali) her
    # zaman kaydediyoruz — diğer 3 motorla tutarlı.
    save_preprocessed_image(ocr_input, preprocessed_dir, img_name)

    # Config'teki TÜM generate_settings anahtarlarını, model.generate'in kabul
    # ettiği gerçek parametrelerle filtreleyip otomatik geçiriyoruz.
    valid_generate_kwargs = filter_valid_kwargs(model.generate, generate_settings)

    # generate() çağrısının kendi confidence/score bilgisini alabilmek için
    # output_scores ve return_dict_in_generate'i zorluyoruz — bu, config'te
    # ne yazılırsa yazılsın avg_confidence hesaplamamız için gerekli.
    # filter_valid_kwargs zaten generate VAR_KEYWORD kabul ettiği için bu
    # ek parametreler reddedilmeyecek.
    confidence_kwargs = dict(valid_generate_kwargs)
    confidence_kwargs["output_scores"] = True
    confidence_kwargs["return_dict_in_generate"] = True

    start_time = time.time()  # Süre BURADA başlar — model yükleme VE
    # preprocessing süreye dahil değil, sadece "OCR motorunun kendisi ne
    # kadar sürdü" ölçülüyor (diğer 3 motordaki ile aynı prensip).

    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

    try:
        generated = model.generate(pixel_values, **confidence_kwargs)
        generated_ids = generated.sequences
        scores_available = True
    except Exception:
        # Bazı TrOCR/transformers sürüm kombinasyonlarında output_scores
        # beklenmeyen bir generate() davranışına yol açabilir (örn. bazı
        # generation stratejileriyle uyumsuzluk). Bu durumda güvenli
        # şekilde confidence'sız çalışmaya geri dönüyoruz — OCR'ın kendisi
        # bundan etkilenmemeli.
        generated_ids = model.generate(pixel_values, **valid_generate_kwargs)
        scores_available = False

    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    execution_time = round(time.time() - start_time, 4)  # Süre BURADA donar

    # --- avg_confidence ---
    # TrOCR'da Tesseract/EasyOCR/doctr'daki gibi kelime bazlı bir confidence
    # yok (TrOCR bbox da üretmiyor). Bunun yerine, generate() çağrısından
    # token bazlı olasılıkları (transition_scores) alıp ortalamasını
    # 0-100 skalasına çeviriyoruz — diğer motorlarla aynı eksende
    # kıyaslanabilir bir sayı elde etmek için. scores hiç alınamadıysa
    # (yukarıdaki fallback'e düşüldüyse) None döndürüyoruz; bu "ölçülmedi"
    # anlamına gelir, gerçek bir 0 değil.
    if scores_available:
        try:
            transition_scores = model.compute_transition_scores(
                generated.sequences, generated.scores, normalize_logits=True
            )
            # -inf olabilecek pad/eos sonrası skorları filtrele
            valid_scores = transition_scores[transition_scores > -float("inf")]
            if len(valid_scores) > 0:
                avg_log_prob = valid_scores.mean().item()
                avg_confidence = round(float(np.exp(avg_log_prob)) * 100, 2)
            else:
                avg_confidence = None
        except Exception:
            # compute_transition_scores bazı model/sürüm kombinasyonlarında
            # farklı davranabilir; hata durumunda sessizce None'a düşüyoruz,
            # OCR sonucunu (text) etkilemiyoruz.
            avg_confidence = None
    else:
        avg_confidence = None

    # --- Buradan sonrası süreye dahil değil: görselleştirme/raporlama ---

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
    masked_img.save(os.path.join(masked_dir, f"masked_{img_name}"))
    cv2.imwrite(os.path.join(highlighted_dir, f"highlighted_{img_name}"), highlighted_img)

    total_time = round(
        image_load_time + preprocessing_time + pil_conversion_time + execution_time, 4
    )

    return {
        "text": text,
        "load_time_seconds": load_time,
        "image_load_time_seconds": image_load_time,
        "preprocessing_time_seconds": preprocessing_time,
        # pil_conversion_time_seconds: TrOCR'a özgü bir adım (diğer 3
        # motorda yok) — OpenCV (BGR) görselini PIL Image (RGB) formatına
        # çevirme süresi. Genelde ihmal edilebilir küçüklükte ama
        # tutarlılık için ayrı ölçülüyor.
        "pil_conversion_time_seconds": pil_conversion_time,
        "execution_time_seconds": execution_time,
        # total_time_seconds: görsel okuma + preprocessing + PIL dönüşümü +
        # OCR motorunun kendisi. load_time_seconds (model yükleme) DAHIL
        # DEĞİL — o, motor için bir kez ödenen, main.py'de paylaşılan bir
        # maliyet.
        "total_time_seconds": total_time,
        "avg_confidence": avg_confidence,
        "model_used": model_name,
        # "eski donanım" senaryonuzda süre rakamlarının hangi donanımda
        # ölçüldüğünü bilmek kritik (diğer motorlardaki ile aynı gerekçe).
        "device_used": device,
        "settings_used": valid_generate_kwargs,
        "preprocessing_used": preprocessing_settings,
    }


if __name__ == "__main__":
    # main.py'nin çağırdığı gibi: config_path = f'configurations/{engine}/{model_name}.yaml'
    result = run_trocr(
        image_path="inputs/images/ornek.png",
        config_path="configurations/trocr/model_v1.yaml",
    )
    print(result)

    # --- ÖNEMLİ: main.py entegrasyonu için ---
    # EasyOCR/doctr'daki ile AYNI sorun burada da var, hatta daha ciddi:
    # TrOCR'ın transformer modeli genelde EasyOCR/doctr'dan da AĞIR. main.py
    # her görsel için run_trocr(img, config) çağırırken processor/model
    # parametrelerini hiç vermiyorsa, ikisi de HER GÖRSEL İÇİN YENİDEN
    # YÜKLENİR. main.py güncellemesinde processor+model'i BİR KEZ kurup
    # tüm görseller için paylaşmamız gerekiyor:
    #
    #   from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    #   shared_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    #   shared_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    #   for img_path in tüm_görseller:
    #       result = run_trocr(img_path, config_path, processor=shared_processor, model=shared_model)