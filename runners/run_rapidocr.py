import os
import time
import cv2
import numpy as np
from rapidocr import RapidOCR
from rapidocr.utils.typings import ModelType
from PIL import Image, ImageDraw, ImageFont

from runners._common import (
    load_config,
    resolve_font_path,
    preprocess_image,
    get_viz_dirs,
    save_preprocessed_image,
    filter_valid_kwargs,
)


def build_rapidocr_params(settings):
    """
    Config'teki model_selection ve advanced_settings bloklarından,
    RapidOCR(params=...) için gereken sözlüğü üretir.

    === model_selection (Det/Cls/Rec mimari seçimi) ===
    Det/Cls/Rec için BAĞIMSIZ olarak ayarlanabilir:
      - model_type: "tiny" | "small" | "mobile" | "medium" | "server"
        (hangi boyutların geçerli olduğu modüle göre değişir — örn. Cls
        sadece "mobile" destekler, Det/Rec daha fazla seçeneğe sahiptir;
        geçersiz bir kombinasyon RapidOCR'ın kendi hata mesajıyla bildirilir)
      - model_path: kendi .onnx dosyanızın yolu — verilirse model_type
        YOK SAYILIR, RapidOCR doğrudan bu dosyayı yükler. Bu, resmi
        PP-OCRv6 modelleri dışında, kendi eğittiğiniz veya farklı bir
        kaynaktan indirdiğiniz modelleri kullanmanızı sağlar.

    === advanced_settings (eşik/limit ayarları, Global + Det/Cls/Rec) ===
    RapidOCR'ın OCR davranışını (tespit hassasiyeti, kabul eşikleri,
    görsel boyut limitleri) etkileyen sayısal parametreler. Bu blokta
    YAZDIĞINIZ HER ANAHTAR doğrudan "Modül.anahtar" formatında params'a
    eklenir (örn. advanced_settings.Det.box_thresh -> "Det.box_thresh").
    Hangi anahtarların geçerli olduğu RapidOCR'ın kendi config şemasına
    bağlıdır — yazım hatası içeren bir anahtar RapidOCR tarafından
    sessizce yok sayılabilir ya da hataya yol açabilir, RapidOCR sürümüne
    göre değişir. Yaygın/pratik olanlar configurations/rapidocr/
    model_full_reference.yaml dosyasında örneklenmiştir.

    Not: EngineConfig (ONNX Runtime/OpenVINO/Paddle/Torch/TensorRT thread
    ve GPU ayarları) BİLEREK bu fonksiyonun kapsamı dışında tutulmuştur —
    bu proje sadece varsayılan ONNX Runtime backend'ini kullanıyor, diğer
    4 backend'in (ve TensorRT'nin GPU-shape profillerinin) konfigürasyonu
    "eski donanım, CPU'da çalışacak basit kurulum" senaryonuzla ilgisizdir.

    Hiçbir ayar verilmezse, RapidOCR'ın kendi varsayılanları kullanılır —
    davranış önceki haliyle birebir aynı kalır.
    """
    params = {}
    model_selection = settings.get("model_selection", {}) or {}
    advanced_settings = settings.get("advanced_settings", {}) or {}

    for module in ("Det", "Cls", "Rec"):
        module_settings = model_selection.get(module, {}) or {}

        model_path = module_settings.get("model_path")
        if model_path:
            # model_path verilmişse model_type'ı hiç göndermiyoruz —
            # RapidOCR'da bu iki ayar birlikte anlamsız (model_path
            # zaten hangi modelin kullanılacağını belirtiyor).
            params[f"{module}.model_path"] = model_path
            continue

        model_type_str = module_settings.get("model_type")
        if model_type_str:
            # RapidOCR ham string kabul etmiyor, ModelType enum'u bekliyor
            # — burada string'i enum'a çeviriyoruz. Geçersiz bir isim
            # (örn. yazım hatası) burada AttributeError ile patlar; bu
            # kasıtlı, çünkü "tiny_" gibi bir yazım hatasını sessizce
            # yutup varsayılana dönmek yanıltıcı olurdu.
            params[f"{module}.model_type"] = ModelType[model_type_str.upper()]

        # advanced_settings'teki bu modüle ait her anahtar, "Modül.anahtar"
        # formatında doğrudan ekleniyor (örn. Det.box_thresh, Det.thresh,
        # Rec.rec_batch_num, Cls.cls_thresh).
        module_advanced = advanced_settings.get(module, {}) or {}
        for key, value in module_advanced.items():
            params[f"{module}.{key}"] = value

    # Global ayarlar (text_score, min_height, max_side_len, vb.) — modüle
    # özel değil, "Global.anahtar" formatında eklenir.
    global_advanced = advanced_settings.get("Global", {}) or {}
    for key, value in global_advanced.items():
        params[f"Global.{key}"] = value

    return params


def run_rapidocr(image_path, config_path, engine=None):
    config = load_config(config_path)
    settings = config.get("ocr_settings", {})
    preprocessing_settings = config.get("preprocessing", {}) or {}

    # call_settings: RapidOCR.__call__()'ın (her görsel çağrısında) kabul
    # ettiği parametreler — EasyOCR'daki readtext_settings'in karşılığı.
    # Config'te kapalıysa boş kalır, RapidOCR kendi varsayılanlarını
    # kullanır (önceki davranışla birebir aynı).
    call_settings = config.get("call_settings", {}) or {}

    # RapidOCR'da dil parametresi Tesseract'tan ("tur+eng") FARKLI bir
    # formatta: tek bir lang_type kodu (örn. "tr", "en", "ch"). PP-OCRv6
    # modelinde Türkçe doğrudan "tr" kodu ile destekleniyor — "latin" genel
    # kategorisi PP-OCRv6'da YOK, doğrudan dil kodu gerekiyor. Bu config
    # alanı Tesseract'taki "lang" ile karıştırılmamalı, farklı bir sözleşme.
    lang_type = settings.get("lang_type", "tr")
    font_path_config = settings.get("font_path")

    # --- load_time_seconds ---
    # EasyOCR/doctr/TrOCR'daki gibi burada da GERÇEK bir maliyet var:
    # RapidOCR(...) detection + classification + recognition modellerini
    # yükler. engine dışarıdan enjekte edilmemişse (main.py her görsel için
    # yeniden oluşturuyorsa) bu maliyet her çağrıda tekrar tekrar ödenir.
    if engine is None:
        load_start = time.time()
        try:
            rapidocr_params = {"Rec.lang_type": lang_type}
            rapidocr_params.update(build_rapidocr_params(settings))
            engine = RapidOCR(params=rapidocr_params)
        except Exception as e:
            raise RuntimeError(
                f"RapidOCR motoru oluşturulamadı (lang_type={lang_type}, "
                f"config: {config_path}). Olası sebepler: bu dil kodu "
                f"PP-OCRv6 tarafından desteklenmiyor olabilir, ya da "
                f"model_selection'daki model_type/model_path ayarı "
                f"geçersiz. Orijinal hata: {e}"
            ) from e
        load_time = round(time.time() - load_start, 4)
    else:
        # Engine dışarıdan (main.py'de bir kez oluşturulup tüm görseller
        # için tekrar kullanılıyorsa) geldiyse, bu çağrıda yükleme yok.
        load_time = 0.0

    # --- image_load_time_seconds: görseli diskten okuma süresi ---
    image_load_start = time.time()
    img = cv2.imread(image_path)
    image_load_time = round(time.time() - image_load_start, 4)

    if img is None:
        # cv2.imread bulunamayan/bozuk dosyada sessizce None döner; burada
        # bunu net bir hataya çeviriyoruz ki toplu işlemde hangi dosyanın
        # sorunlu olduğu hemen görülsün. (Bu kontrol görselleştirme için
        # kullanılan img'i koruyor; RapidOCR'a asıl giden ocr_input da
        # bundan türetiliyor.)
        raise ValueError(
            f"Görsel okunamadı (bozuk dosya, desteklenmeyen format ya da "
            f"yol hatalı olabilir): {image_path}"
        )

    # --- preprocessing_time_seconds ---
    preprocessing_start = time.time()
    ocr_input = preprocess_image(img, preprocessing_settings)
    preprocessing_time = round(time.time() - preprocessing_start, 4)

    # --- Kanal güvenliği ---
    # preprocess_image bazı adımlardan sonra tek kanallı (2D) görsel
    # döndürebilir. RapidOCR numpy array kabul ediyor ve tek kanallıyı da
    # işleyebiliyor, ama diğer 4 motorla (EasyOCR, doctr, TrOCR) tutarlı
    # davranış için burada da 3 kanala (BGR) geri çeviriyoruz — görsel
    # İÇERİK değişmez, sadece kanal sayısı garanti edilir.
    if len(ocr_input.shape) == 2:
        ocr_input = cv2.cvtColor(ocr_input, cv2.COLOR_GRAY2BGR)

    highlighted_dir, masked_dir, preprocessed_dir = get_viz_dirs(config_path)
    img_name = os.path.basename(image_path)

    # OCR motoruna giden görseli (preprocessing kapalıysa ham hali) her
    # zaman kaydediyoruz — diğer 4 motorla tutarlı.
    save_preprocessed_image(ocr_input, preprocessed_dir, img_name)

    start_time = time.time()  # Süre BURADA başlar — model yükleme VE
    # preprocessing süreye dahil değil, sadece "OCR motorunun kendisi ne
    # kadar sürdü" ölçülüyor (diğer 4 motordaki ile aynı prensip).
    # Not: RapidOCR'ın kendi döndürdüğü 'elapse' alanı KASITLI olarak
    # KULLANILMIYOR — kendi iç ölçümü preprocessing/I-O sınırlarımızla
    # birebir örtüşmeyebilir, bu yüzden diğer motorlardaki gibi kendi
    # time.time() ölçümümüzü yapıyoruz, tutarlılık için.
    # Config'teki TÜM call_settings anahtarlarını, RapidOCR.__call__'ın
    # GERÇEKTEN kabul ettiği parametrelerle filtreleyip otomatik geçiriyoruz
    # — EasyOCR'ın readtext_settings'inde kullandığımız aynı mekanizma. Bu
    # sayede use_det, use_cls, use_rec, return_word_box,
    # return_single_char_box, text_score, box_thresh, unclip_ratio gibi
    # __call__'ın desteklediği HER parametre, config'e eklemeniz yeterli
    # olacak şekilde otomatik çalışır — kod değişikliği gerekmez.
    valid_call_kwargs = filter_valid_kwargs(engine.__call__, call_settings)
    result = engine(ocr_input, **valid_call_kwargs)

    execution_time = round(time.time() - start_time, 4)  # Süre BURADA donar

    # --- Buradan sonrası süreye dahil değil: görselleştirme/raporlama ---

    # txts/scores boş tuple olabilir (hiç metin bulunamadıysa)
    texts = result.txts if result.txts else ()
    scores = result.scores if result.scores else ()
    boxes = result.boxes if result.boxes is not None else []

    recognized_text = " ".join(texts)

    # --- avg_confidence ---
    # RapidOCR her satır için 0-1 arası bir skor döndürür. Diğer motorlarla
    # aynı eksende (0-100) kıyaslanabilsin diye burada da çeviriyoruz.
    # Hiç metin bulunamadıysa (gerçek bir "0 sonuç" durumu) 0.0; texts var
    # ama scores boşsa (beklenmeyen bir durum) None.
    if not texts:
        avg_confidence = 0.0
    elif not scores:
        avg_confidence = None
    else:
        avg_confidence = round((sum(scores) / len(scores)) * 100, 2)

    # --- Görselleştirme ---
    # Highlight/mask çizimleri orijinal RENKLİ görsel üzerine yapılır —
    # preprocessing sadece OCR motoruna giden girdiyi etkiler.
    overlay = img.copy()
    pil_img = Image.new("RGB", (img.shape[1], img.shape[0]), (255, 255, 255))
    draw = ImageDraw.Draw(pil_img)

    font_path = resolve_font_path(font_path_config)

    for box, text in zip(boxes, texts):
        # box: 4 köşe noktası [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        xs = [pt[0] for pt in box]
        ys = [pt[1] for pt in box]
        x, y = int(min(xs)), int(min(ys))
        w, h = int(max(xs) - min(xs)), int(max(ys) - min(ys))

        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), -1)

        font_size = max(int(h * 0.8), 1)
        try:
            font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()

        draw.text((x, y), text, font=font, fill=(0, 0, 0))

    alpha = 0.3
    highlighted_img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    cv2.imwrite(os.path.join(highlighted_dir, f"highlighted_{img_name}"), highlighted_img)

    masked_final = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(masked_dir, f"masked_{img_name}"), masked_final)

    total_time = round(image_load_time + preprocessing_time + execution_time, 4)

    return {
        "text": recognized_text,
        "load_time_seconds": load_time,
        "image_load_time_seconds": image_load_time,
        "preprocessing_time_seconds": preprocessing_time,
        "execution_time_seconds": execution_time,
        # total_time_seconds: görsel okuma + preprocessing + OCR motorunun
        # kendisi. load_time_seconds (model yükleme) DAHIL DEĞİL.
        "total_time_seconds": total_time,
        "avg_confidence": avg_confidence,
        # model_used: hangi dil/sürüm modeliyle çalıştığı görünsün —
        # diğer motorlardaki model_used formatıyla tutarlı.
        "model_used": f"rapidocr-{lang_type}",
        # RapidOCR varsayılan olarak CPU'da (ONNX Runtime) çalışır; GPU
        # desteği farklı bir backend (CUDA execution provider) gerektirir
        # ve şu an config'te ayrı bir ayar olarak sunulmuyor — bu yüzden
        # diğer motorlardaki gibi "auto" tespiti yok, sabit "cpu".
        "device_used": "cpu",
        "settings_used": {
            "lang_type": lang_type,
            "model_selection": settings.get("model_selection", {}),
            "advanced_settings": settings.get("advanced_settings", {}),
            "call_settings_used": valid_call_kwargs,
        },
        "preprocessing_used": preprocessing_settings,
    }


if __name__ == "__main__":
    # main.py'nin çağırdığı gibi: config_path = f'configurations/{engine}/{model_name}.yaml'
    result = run_rapidocr(
        image_path="inputs/images/ornek.png",
        config_path="configurations/rapidocr/model_v1.yaml",
    )
    print(result)

    # --- ÖNEMLİ: main.py entegrasyonu için ---
    # Diğer 4 motordaki ile AYNI sorun burada da var: main.py her görsel
    # için run_rapidocr(img, config) çağırırken `engine` parametresini hiç
    # vermiyorsa, RapidOCR(...) HER GÖRSEL İÇİN YENİDEN YÜKLENİR. main.py
    # güncellemesinde (registry.py) engine'i BİR KEZ kurup tüm görseller
    # için paylaşmamız gerekiyor:
    #
    #   from rapidocr import RapidOCR
    #   shared_engine = RapidOCR(params={"Rec.lang_type": "tr"})
    #   for img_path in tüm_görseller:
    #       result = run_rapidocr(img_path, config_path, engine=shared_engine)