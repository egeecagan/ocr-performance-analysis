import os
import time
import cv2
import numpy as np
import easyocr
import torch
from PIL import Image, ImageDraw, ImageFont

from runners._common import (
    load_config,
    resolve_font_path,
    filter_valid_kwargs,
    preprocess_image,
    get_viz_dirs,
    save_preprocessed_image,
)


def run_easyocr(image_path, config_path, reader=None):
    config = load_config(config_path)
    reader_settings = config.get("reader_settings", {})
    readtext_settings = config.get("readtext_settings", {}) or {}
    preprocessing_settings = config.get("preprocessing", {}) or {}

    languages = reader_settings.get("languages", ["tr", "en"])
    font_path_config = reader_settings.get("font_path")  # opsiyonel, config'te elle belirtilebilir

    # --- GPU seçimi ---
    # Config'te "auto" (varsayılan) yazılırsa, makinede GPU var mı diye
    # otomatik tespit edilir — TrOCR'daki torch.cuda.is_available() mantığıyla
    # tutarlı. "true"/"false" yazarsanız zorlama olarak kullanılır (örn.
    # GPU'nuz olsa bile testi CPU'da koşmak isteyebilirsiniz).
    gpu_setting = reader_settings.get("gpu", "auto")
    if gpu_setting == "auto":
        gpu = torch.cuda.is_available()
    else:
        gpu = bool(gpu_setting)

    # --- load_time_seconds ---
    # Tesseract'ın aksine burada GERÇEK bir maliyet var: easyocr.Reader(...)
    # modelleri diskten/internetten yükler, bu saniyeler sürebilir. Eğer
    # reader dışarıdan enjekte edilmemişse (main.py her görsel için Reader'ı
    # YENİDEN oluşturuyorsa) bu maliyet her çağrıda tekrar tekrar ödenir —
    # tam da "eski donanımda ilk açılış" senaryonuzda önemli olan şey bu.
    # reader=None ise burada GERÇEKTEN ölçüyoruz, sabit 0.0 yazmıyoruz.
    if reader is None:
        load_start = time.time()
        try:
            reader = easyocr.Reader(languages, gpu=gpu)
        except Exception as e:
            # EasyOCR bazı dil kombinasyonlarını desteklemez (örn. bazı
            # Latin-olmayan scriptler birlikte kullanılamaz). Kütüphanenin
            # kendi hata mesajı genelde yeterli ama hangi config/dillerin
            # bu hataya yol açtığını da ekleyerek 20 görsellik toplu
            # işlemde debug'ı kolaylaştırıyoruz.
            raise RuntimeError(
                f"EasyOCR Reader oluşturulamadı (diller: {languages}, "
                f"config: {config_path}). Olası sebep: bu dil kombinasyonu "
                f"EasyOCR tarafından desteklenmiyor olabilir. Orijinal hata: {e}"
            ) from e
        load_time = round(time.time() - load_start, 4)
    else:
        # Reader dışarıdan (örn. main.py'de bir kez oluşturulup tüm
        # görseller için tekrar kullanılıyorsa) geldiyse, bu çağrıda
        # yükleme maliyeti yok.
        load_time = 0.0

    # --- image_load_time_seconds: görseli diskten okuma süresi ---
    image_load_start = time.time()
    img = cv2.imread(image_path)
    image_load_time = round(time.time() - image_load_start, 4)

    if img is None:
        # cv2.imread bulunamayan/bozuk dosyada sessizce None döner; burada
        # bunu net bir hataya çeviriyoruz ki 20 görsellik toplu işlemde
        # hangi dosyanın sorunlu olduğu hemen görülsün.
        raise ValueError(
            f"Görsel okunamadı (bozuk dosya, desteklenmeyen format ya da "
            f"yol hatalı olabilir): {image_path}"
        )

    # --- preprocessing_time_seconds ---
    # Önceden bu süre HİÇ ölçülmüyordu. execution_time_seconds (OCR
    # motorunun kendisi) bundan ETKİLENMİYOR, ayrı tutuluyor.
    preprocessing_start = time.time()
    ocr_input = preprocess_image(img, preprocessing_settings)
    preprocessing_time = round(time.time() - preprocessing_start, 4)

    # --- Kanal güvenliği ---
    # preprocess_image bazı adımlardan sonra (grayscale, threshold, clahe,
    # denoise, morphology, sharpen, illumination_correct...) TEK KANALLI
    # (2D) bir görsel döndürebilir. EasyOCR'ın readtext() fonksiyonu çoğu
    # sürümde tek kanallı girdiyi kabul etse de, bazı sürümlerde/iç
    # işlemlerde 3 kanal (BGR) beklenir ve sessizce yanlış sonuç ya da hata
    # üretebilir. Bunu garantiye almak için: çıktı tek kanallıysa BGR'a
    # geri çeviriyoruz (gri değerler R=G=B olacak şekilde tekrarlanır,
    # GÖRSEL İÇERİK değişmez, sadece kanal sayısı 3'e çıkar).
    if len(ocr_input.shape) == 2:
        ocr_input = cv2.cvtColor(ocr_input, cv2.COLOR_GRAY2BGR)

    # Config'teki TÜM readtext_settings anahtarlarını, readtext'in kabul
    # ettiği gerçek parametrelerle filtreleyip otomatik geçiriyoruz.
    valid_readtext_kwargs = filter_valid_kwargs(reader.readtext, readtext_settings)

    start_time = time.time()  # Süre BURADA başlar — model yükleme VE
    # preprocessing süreye dahil değil, sadece "OCR motorunun kendisi ne
    # kadar sürdü" ölçülüyor (Tesseract'taki ile aynı prensip).

    # OCR işlemini yap — config'teki tüm ayarlarla
    results = reader.readtext(ocr_input, **valid_readtext_kwargs)

    execution_time = round(time.time() - start_time, 4)  # Süre BURADA donar

    # --- Buradan sonrası süreye dahil değil: görselleştirme/raporlama ---
    highlighted_dir, masked_dir, preprocessed_dir = get_viz_dirs(config_path)

    # OCR motoruna giden görseli (preprocessing kapalıysa ham hali, kanal
    # güvenliği sonrası 3 kanallı hali) her zaman kaydediyoruz.
    save_preprocessed_image(ocr_input, preprocessed_dir, os.path.basename(image_path))

    # Highlight/mask çizimleri orijinal RENKLİ görsel üzerine yapılır —
    # preprocessing sadece OCR motoruna giden girdiyi etkiler.
    overlay = img.copy()
    pil_img = Image.new("RGB", (img.shape[1], img.shape[0]), (255, 255, 255))
    draw = ImageDraw.Draw(pil_img)

    font_path = resolve_font_path(font_path_config)

    # detail=0 verilirse results sadece string listesi döner, bbox/prob olmaz.
    # paragraph=true verilirse bbox formatı bazı EasyOCR sürümlerinde
    # değişebilir (satırlar birleştirilmiş "paragraf" kutuları olur) ve
    # bu format burada test edilmedi. Her iki durumda da görselleştirmeyi
    # GÜVENLİ şekilde atlıyoruz — format uyumsuzluğunda kodun hata vererek
    # tüm pipeline'ı durdurmasındansa, sadece highlight/mask görseli
    # üretilmemesini tercih ediyoruz (metin/confidence sonucu etkilenmez).
    detail = valid_readtext_kwargs.get("detail", 1)
    paragraph = valid_readtext_kwargs.get("paragraph", False)
    can_visualize = (detail != 0) and (not paragraph)

    confidences = []

    if can_visualize:
        for (bbox, text, prob) in results:
            (top_left, top_right, bottom_right, bottom_left) = bbox
            x, y = int(top_left[0]), int(top_left[1])
            w = int(bottom_right[0] - top_left[0])
            h = int(bottom_right[1] - top_left[1])

            confidences.append(prob)

            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), -1)

            # --- Masked (gerçek font, Türkçe karakterleri destekler) ---
            font_size = max(int(h * 0.8), 1)
            try:
                font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
            except Exception:
                font = ImageFont.load_default()

            draw.text((x, y), text, font=font, fill=(0, 0, 0))

        recognized_text = " ".join([r[1] for r in results])
    elif detail == 0:
        # detail=0 -> results zaten düz string listesi, prob bilgisi yok
        recognized_text = " ".join(results)
    else:
        # paragraph=true -> bbox formatı test edilmedi, güvenli tarafta
        # kalıp metni de aynı yapıdan (bbox, text, prob) çıkarıyoruz ama
        # highlight/mask ÇİZMİYORUZ. Confidence'ı yine topluyoruz çünkü
        # prob alanı paragraph modunda da genelde mevcut.
        recognized_words = []
        for item in results:
            # paragraph modunda format (bbox, text) olabilir, prob
            # gelmeyebilir — esnek şekilde ele alıyoruz.
            if len(item) == 3:
                _, text, prob = item
                confidences.append(prob)
            else:
                _, text = item
            recognized_words.append(text)
        recognized_text = " ".join(recognized_words)

    # --- avg_confidence ---
    # EasyOCR her kelime için 0-1 arası bir 'prob' döndürür. Tesseract'taki
    # avg_confidence (0-100 skala) ile aynı eksende kıyaslanabilsin diye
    # burada da 0-100'e çeviriyoruz.
    #
    # detail=0 kullanılırsa EasyOCR hiç prob bilgisi vermez — bu durumda
    # avg_confidence None olur (0.0 DEĞİL): "model %0 güvenle okudu" (gerçek
    # kötü sonuç) ile "bu ayarda confidence hiç ölçülmedi" (bilgi eksikliği)
    # birbirine karışmasın diye. Görselde gerçekten hiç kelime bulunamadıysa
    # (detail=1 ama results boşsa) bu GERÇEK bir "0 kelime" durumu, prob
    # eksikliği değil — o yüzden ayrı ele alınır.
    if detail == 0:
        avg_confidence = None  # bilgi hiç ölçülmedi
    elif not results:
        avg_confidence = 0.0  # gerçekten hiç kelime bulunamadı
    elif not confidences:
        # results var ama prob toplanamadı (örn. paragraph modunda format
        # farklı çıktı) -> bilgi eksikliği, gerçek 0 değil
        avg_confidence = None
    else:
        avg_confidence = round((sum(confidences) / len(confidences)) * 100, 2)

    # Kayıtlar
    img_name = os.path.basename(image_path)
    cv2.imwrite(
        os.path.join(highlighted_dir, f"highlighted_{img_name}"),
        cv2.addWeighted(overlay, 0.3, img, 0.7, 0),
    )

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
        # kendisi. load_time_seconds (model yükleme) DAHIL DEĞİL — o, motor
        # için bir kez ödenen, main.py'de paylaşılan bir maliyet.
        "total_time_seconds": total_time,
        "avg_confidence": avg_confidence,
        # Hangi dillerle çalıştığı artık model_used'da görünüyor, örn.
        # "easyocr-tr+en" — Tesseract'taki model_used formatıyla tutarlı.
        "model_used": f"easyocr-{'+'.join(languages)}",
        # GPU mu CPU mu kullanıldığı ayrı bir alanda — "eski donanım"
        # senaryonuzda süre rakamlarının hangi donanımda ölçüldüğünü
        # bilmek kritik. Bu olmadan "EasyOCR X saniye sürdü" başka bir
        # makinede (özellikle GPU'su olmayan eski bir bilgisayarda) hiç
        # anlamlı olmaz.
        "device_used": "gpu" if gpu else "cpu",
        "settings_used": valid_readtext_kwargs,
        "preprocessing_used": preprocessing_settings,
    }


if __name__ == "__main__":
    # main.py'nin çağırdığı gibi: config_path = f'configurations/{engine}/{model_name}.yaml'
    result = run_easyocr(
        image_path="inputs/images/ornek.png",
        config_path="configurations/easyocr/model_v1.yaml",
    )
    print(result)

    # --- ÖNEMLİ: main.py entegrasyonu için ---
    # Şu anki main.py, her görsel için run_easyocr(img, config) çağırırken
    # `reader` parametresini HİÇ vermiyor. Bu durumda yukarıdaki örnekte
    # olduğu gibi her çağrıda reader=None olur ve easyocr.Reader(...) HER
    # GÖRSEL İÇİN YENİDEN YÜKLENİR — 20 görsellik bir klasörde model 20 kez
    # diskten/internetten yüklenir. Bu hem gerçek dünyada anlamsız bir
    # yavaşlıktır hem de raporunuzdaki "EasyOCR süresi" rakamını yapay
    # olarak şişirir (asıl suçlu tekrar tekrar yükleme, OCR'ın kendisi
    # değil).
    #
    # Doğru kullanım — reader'ı BİR KEZ oluşturup tüm görseller için
    # paylaşmak:
    #
    #   import easyocr
    #   shared_reader = easyocr.Reader(["tr", "en"], gpu=False)
    #   for img_path in tüm_görseller:
    #       result = run_easyocr(img_path, config_path, reader=shared_reader)
    #
    # main.py'yi güncellediğimizde bu mantığı process_pipeline() içine
    # ekleyeceğiz — engine == 'easyocr' (ve aynı sorunu yaşayan 'trocr',
    # 'doctr' için de) ilk görselden önce reader/model bir kez kurulup
    # döngü boyunca yeniden kullanılacak.