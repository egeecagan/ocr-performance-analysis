import os
import time
import cv2
import numpy as np
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont

from runners._common import (
    load_config,
    resolve_font_path,
    preprocess_image,
    get_viz_dirs,
    save_preprocessed_image,
    filter_valid_kwargs,
)


def run_paddleocr(image_path, config_path, engine=None):
    config = load_config(config_path)
    settings = config.get("ocr_settings", {})
    preprocessing_settings = config.get("preprocessing", {}) or {}

    # PaddleOCR'da dil parametresi "lang" — RapidOCR'ın "lang_type"ından
    # farklı bir isimle ama aynı mantıkla çalışıyor: tek bir dil kodu
    # ("tr", "en", "ch" gibi). PP-OCRv6'nın birleşik (unified) modeli
    # Türkçe'yi Latin grubunun bir parçası olarak doğrudan destekliyor.
    lang = settings.get("lang", "tr")
    font_path_config = settings.get("font_path")

    # PaddleOCR'ın doc-orientation/unwarping/textline-orientation gibi ek
    # ön-işleme adımları VAR ve varsayılan olarak bazıları açık olabilir.
    # Biz bunları config'ten KAPATIYORUZ çünkü kendi preprocess_image()
    # pipeline'ımızla çakışmasın, kıyas adil kalsın — PaddleOCR'ın "kendi
    # içinde gizli bir önişleme" yapıp yapmadığını biz kontrol edelim.
    use_doc_orientation_classify = settings.get("use_doc_orientation_classify", False)
    use_doc_unwarping = settings.get("use_doc_unwarping", False)
    use_textline_orientation = settings.get("use_textline_orientation", False)

    # Config'teki TÜM ocr_settings anahtarlarını, PaddleOCR.__init__'in
    # GERÇEKTEN kabul ettiği parametrelerle filtreleyip otomatik
    # geçiriyoruz — EasyOCR/doctr'da kullandığımız aynı mekanizma. Bu
    # sayede text_det_thresh, text_det_box_thresh, text_det_unclip_ratio,
    # text_rec_score_thresh, ocr_version gibi PaddleOCR.__init__'in
    # desteklediği HER parametre, config'e eklemeniz yeterli olacak
    # şekilde otomatik çalışır — kod değişikliği gerekmez.
    #
    # ÖNEMLİ: bu hesaplama BİLEREK `if engine is None:` bloğunun DIŞINDA
    # tutuluyor — engine dışarıdan (main.py/registry.py üzerinden,
    # paylaşılan bir nesne olarak) geldiğinde de extra_init_kwargs'ın
    # TANIMLI olması gerekiyor, çünkü aşağıda settings_used alanına
    # yazılırken kullanılıyor. Sadece if bloğunun içinde tanımlanırsa,
    # engine dışarıdan geldiğinde bu değişken hiç oluşmaz ve
    # settings_used'a erişirken UnboundLocalError fırlatılır (doctr
    # runner'ında karşılaştığımız ve düzelttiğimiz aynı hata sınıfı).
    #
    # lang/use_doc_orientation_classify/use_doc_unwarping/
    # use_textline_orientation zaten kendi değişkenleriyle ayrı tutulup
    # elle geçiriliyor; geri kalan her şey ocr_settings'ten filtrelenip
    # eklenir. PaddleOCR.__init__ **kwargs aldığı için (doctr'daki gibi),
    # filter_valid_kwargs yazım hatalarını eleyemez — geçersiz bir anahtar
    # PaddleOCR'a kadar gidip orada hata verebilir, bu durumda try/except
    # net bir mesajla yakalıyor.
    extra_init_kwargs = filter_valid_kwargs(PaddleOCR.__init__, settings)
    extra_init_kwargs.pop("lang", None)
    extra_init_kwargs.pop("use_doc_orientation_classify", None)
    extra_init_kwargs.pop("use_doc_unwarping", None)
    extra_init_kwargs.pop("use_textline_orientation", None)
    # font_path bizim kendi alanımız, PaddleOCR'ın gerçek bir parametresi
    # değil — **kwargs yüzünden filtrelenemediği için elle çıkarıyoruz.
    extra_init_kwargs.pop("font_path", None)

    # call_settings: PaddleOCR.predict()'in (her görsel çağrısında) kabul
    # ettiği parametreler — EasyOCR'daki readtext_settings'in karşılığı.
    # Config'te kapalıysa boş kalır, PaddleOCR kendi (kurulumda verilen ya
    # da varsayılan) ayarlarını kullanır.
    call_settings = config.get("call_settings", {}) or {}

    # --- load_time_seconds ---
    # Diğer motorlardaki gibi burada da GERÇEK bir maliyet var. ÖNEMLİ:
    # PaddleOCR'ın ilk kullanımda model dosyalarını İNTERNETTEN indirmesi
    # gerekiyor (HuggingFace/ModelScope/AIStudio/BOS) — RapidOCR'ın aksine
    # modeller pakete gömülü DEĞİL. Offline/kapalı bir ağda bu adım hata
    # verir (bunu test ederken biz de karşılaştık). "Eski donanım, hızlı
    # kurulum" senaryonuzda bu önemli bir dezavantaj olarak not edilmeli.
    if engine is None:
        load_start = time.time()
        try:
            engine = PaddleOCR(
                lang=lang,
                use_doc_orientation_classify=use_doc_orientation_classify,
                use_doc_unwarping=use_doc_unwarping,
                use_textline_orientation=use_textline_orientation,
                **extra_init_kwargs,
            )
        except Exception as e:
            raise RuntimeError(
                f"PaddleOCR motoru oluşturulamadı (lang={lang}, config: "
                f"{config_path}). Olası sebepler: (1) bu dil kodu "
                f"desteklenmiyor olabilir, (2) model dosyaları internetten "
                f"indirilemedi — PaddleOCR'ın modelleri RapidOCR gibi "
                f"pakete gömülü değildir, ilk kullanımda internet "
                f"bağlantısı gerektirir. Orijinal hata: {e}"
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
        raise ValueError(
            f"Görsel okunamadı (bozuk dosya, desteklenmeyen format ya da "
            f"yol hatalı olabilir): {image_path}"
        )

    # --- preprocessing_time_seconds ---
    preprocessing_start = time.time()
    ocr_input = preprocess_image(img, preprocessing_settings)
    preprocessing_time = round(time.time() - preprocessing_start, 4)

    # --- Kanal güvenliği ---
    # Diğer motorlardaki ile aynı: preprocess_image tek kanallı (2D)
    # görsel döndürebilir, PaddleOCR'ın iç pipeline'ı 3 kanal (BGR/RGB)
    # bekler — görsel İÇERİK değişmeden 3 kanala geri çeviriyoruz.
    if len(ocr_input.shape) == 2:
        ocr_input = cv2.cvtColor(ocr_input, cv2.COLOR_GRAY2BGR)

    highlighted_dir, masked_dir, preprocessed_dir = get_viz_dirs(config_path)
    img_name = os.path.basename(image_path)

    # OCR motoruna giden görseli (preprocessing kapalıysa ham hali) her
    # zaman kaydediyoruz — diğer 5 motorla tutarlı.
    save_preprocessed_image(ocr_input, preprocessed_dir, img_name)

    start_time = time.time()  # Süre BURADA başlar — model yükleme VE
    # preprocessing süreye dahil değil, sadece "OCR motorunun kendisi ne
    # kadar sürdü" ölçülüyor (diğer 5 motordaki ile aynı prensip).

    # Config'teki TÜM call_settings anahtarlarını, PaddleOCR.predict'in
    # GERÇEKTEN kabul ettiği parametrelerle filtreleyip otomatik
    # geçiriyoruz — EasyOCR'ın readtext_settings'inde kullandığımız aynı
    # mekanizma. predict() **kwargs almıyor (sabit parametre listesi var),
    # bu yüzden yazım hatalı bir anahtar burada GÜVENLE sessizce elenir.
    valid_call_kwargs = filter_valid_kwargs(engine.predict, call_settings)

    # predict() bir liste döndürür (çok sayfalı girdi desteği için); biz
    # tek görsel verdiğimiz için sadece ilk (ve tek) sonucu kullanıyoruz.
    results = engine.predict(ocr_input, **valid_call_kwargs)
    result = results[0] if results else None

    execution_time = round(time.time() - start_time, 4)  # Süre BURADA donar

    # --- Buradan sonrası süreye dahil değil: görselleştirme/raporlama ---

    # PaddleOCR 3.x sonuç nesnesi dict-like'tır: result["rec_texts"],
    # result["rec_polys"], result["rec_scores"] ile erişilir.
    if result is not None:
        texts = result.get("rec_texts", []) or []
        scores = result.get("rec_scores", []) or []
        polys = result.get("rec_polys", []) or []
    else:
        texts, scores, polys = [], [], []

    recognized_text = " ".join(texts)

    # --- avg_confidence ---
    # PaddleOCR her satır için 0-1 arası bir skor döndürür. Diğer
    # motorlarla aynı eksende (0-100) kıyaslanabilsin diye çeviriyoruz.
    if not texts:
        avg_confidence = 0.0  # gerçekten hiç metin bulunamadı
    elif not scores:
        avg_confidence = None  # texts var ama score gelmedi (beklenmeyen)
    else:
        avg_confidence = round((sum(scores) / len(scores)) * 100, 2)

    # --- Görselleştirme ---
    overlay = img.copy()
    pil_img = Image.new("RGB", (img.shape[1], img.shape[0]), (255, 255, 255))
    draw = ImageDraw.Draw(pil_img)

    font_path = resolve_font_path(font_path_config)

    words = []  # GUI/web arayüzü için: her kelimenin metni, bbox'ı ve
    # kendi confidence'ı. scores, texts/polys ile aynı indekste eşleşiyor.

    for i, (poly, text) in enumerate(zip(polys, texts)):
        poly_arr = np.array(poly)
        x, y = int(poly_arr[:, 0].min()), int(poly_arr[:, 1].min())
        w = int(poly_arr[:, 0].max() - poly_arr[:, 0].min())
        h = int(poly_arr[:, 1].max() - poly_arr[:, 1].min())

        word_confidence = round(scores[i] * 100, 2) if i < len(scores) else None
        words.append({
            "text": text,
            "bbox": [x, y, x + w, y + h],
            "confidence": word_confidence,
        })

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
        # words: GUI/web arayüzü için kelime bazlı bbox+confidence listesi.
        "words": words,
        "load_time_seconds": load_time,
        "image_load_time_seconds": image_load_time,
        "preprocessing_time_seconds": preprocessing_time,
        "execution_time_seconds": execution_time,
        "total_time_seconds": total_time,
        "avg_confidence": avg_confidence,
        "model_used": f"paddleocr-{lang}",
        # PaddleOCR varsayılan olarak CPU'da çalışır (GPU desteği farklı
        # bir kurulum/parametre seti gerektirir, "device" parametresi ile
        # ayarlanabilir ama şu an config'te ayrı bir ayar olarak sunulmuyor).
        "device_used": "cpu",
        "settings_used": {
            "lang": lang,
            "use_doc_orientation_classify": use_doc_orientation_classify,
            "use_doc_unwarping": use_doc_unwarping,
            "use_textline_orientation": use_textline_orientation,
            # Any extra ocr_settings keys (text_det_thresh,
            # text_det_box_thresh, text_rec_score_thresh, ocr_version,
            # etc.) actually passed to PaddleOCR.__init__ — without this,
            # these settings would be used to build the engine but
            # invisible in the JSON output.
            **extra_init_kwargs,
            "call_settings_used": valid_call_kwargs,
        },
        "preprocessing_used": preprocessing_settings,
    }


if __name__ == "__main__":
    # main.py'nin çağırdığı gibi: config_path = f'configurations/{engine}/{model_name}.yaml'
    result = run_paddleocr(
        image_path="inputs/images/ornek.png",
        config_path="configurations/paddleocr/model_v1.yaml",
    )
    print(result)

    # --- ÖNEMLİ: main.py entegrasyonu için ---
    # Diğer motorlardaki ile AYNI sorun burada da var: main.py her görsel
    # için run_paddleocr(img, config) çağırırken `engine` parametresini
    # hiç vermiyorsa, PaddleOCR(...) HER GÖRSEL İÇİN YENİDEN YÜKLENİR — ve
    # PaddleOCR'da bu özellikle maliyetli çünkü ilk yüklemede model
    # indirme/diskten okuma adımı var. main.py güncellemesinde (registry.py)
    # engine'i BİR KEZ kurup tüm görseller için paylaşmamız gerekiyor:
    #
    #   from paddleocr import PaddleOCR
    #   shared_engine = PaddleOCR(lang="tr", use_doc_orientation_classify=False,
    #                              use_doc_unwarping=False, use_textline_orientation=False)
    #   for img_path in tüm_görseller:
    #       result = run_paddleocr(img_path, config_path, engine=shared_engine)