"""
runners/_common.py

Dört OCR runner'ı (tesseract, easyocr, trocr, doctr) arasında BİREBİR aynı
şekilde tekrar eden fonksiyonlar burada tek bir yerde tanımlanır.

Neden bu dosya var:
- Önceden her runner kendi içinde load_config / resolve_font_path gibi
  fonksiyonları kopyalıyordu. Bir davranışı düzeltmek (örn. font bulma
  mantığı) istediğinde 4 dosyayı aynı anda güncellemen gerekiyordu.
- Artık her runner buradan import eder, tek yerde değişiklik 4 runner'a
  birden yansır.

Bu dosya hiçbir OCR motoruna özel kod İÇERMEZ — sadece motordan bağımsız,
ortak yardımcı fonksiyonlar burada yaşar.
"""

import os
import inspect
import platform
import yaml
import cv2
import numpy as np


def load_config(config_path):
    """YAML config dosyasını okuyup dict olarak döndürür."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_font_path(config_font_path=None):
    """
    Masked görselde metni yazdırmak için kullanılacak gerçek bir TrueType
    font dosyasının yolunu bulur. ImageFont.load_default()'ın bitmap fontu
    Türkçe karakterleri (ş, ğ, ı, ö, ü, ç) düzgün render ETMEZ — kutucuk
    veya eksik karakter olarak görünür, bu yüzden mümkünse gerçek bir font
    tercih edilir.

    Öncelik sırası:
    1. Config'te elle belirtilen font_path (varsa ve dosya gerçekten varsa)
    2. İşletim sistemine göre bilinen, Türkçe karakterleri destekleyen fontlar
    3. Hiçbiri bulunamazsa None döner (çağıran taraf ImageFont.load_default()'a düşer)
    """
    if config_font_path and os.path.exists(config_font_path):
        return config_font_path

    system = platform.system()
    candidates = []

    if system == "Darwin":
        candidates = [
            "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        ]
    elif system == "Linux":
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        ]
    elif system == "Windows":
        candidates = [
            "C:\\Windows\\Fonts\\arial.ttf",
            "C:\\Windows\\Fonts\\segoeui.ttf",
        ]

    for path in candidates:
        if os.path.exists(path):
            return path

    return None


def filter_valid_kwargs(func, kwargs):
    """
    Config'ten gelen kwargs sözlüğünü, hedef fonksiyonun (örn. readtext,
    model.generate) GERÇEKTEN kabul ettiği parametrelerle sınırlar.

    Bu sayede:
    - Config'e yazım hatası içeren bir anahtar eklenirse kod patlamaz,
      sadece o anahtar sessizce yok sayılır.
    - Kütüphanenin yeni bir sürümünde fonksiyonun imzası değişirse kod
      otomatik olarak buna uyum sağlar.

    Not: model.generate(**model_kwargs) gibi bazı HuggingFace fonksiyonları
    **kwargs ile serbest argüman da kabul edebiliyor (VAR_KEYWORD). Bu
    durumda inspect.signature her şeyi geçerli sayar — bu istenen davranış,
    çünkü generate() çoğu zaman generation_config üzerinden de parametre
    kabul eder.
    """
    sig = inspect.signature(func)
    params = sig.parameters

    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)

    valid_params = set(params.keys())
    return {k: v for k, v in kwargs.items() if k in valid_params}


def _deskew(gray):
    """
    Görseldeki metin satırlarının açısını tahmin edip görseli düzeltir.
    Telefonla çekilmiş, hafif eğri belgelerde (ehliyet/dekont fotoğrafı gibi)
    faydalıdır — satırlar yatay olmadığında OCR doğruluğu ciddi düşer.

    Yöntem: görseldeki (yazı olması muhtemel) koyu piksellerin minimum
    alanlı dikdörtgenini (minAreaRect) bulup açısını hesaplar, sonra tüm
    görseli ters açıda döndürür.
    """
    # Threshold ile yazı bölgelerini kabaca ayır (sadece açı tahmini için,
    # nihai OCR girdisini etkilemez)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))

    if coords.shape[0] < 10:
        # Yeterli yazı pikseli yoksa (boş/çok temiz görsel) açı tahmini
        # güvenilir olmaz, görseli olduğu gibi bırak.
        return gray

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = gray.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        gray, M, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated


def _autocrop_border(gray, margin_ratio=0.01):
    """
    Görselin kenarlarındaki boş/tek renk şeritleri (tarayıcı kenarlığı,
    fazla beyaz boşluk vb.) kırpar. İçerik tespiti için yazı/şekil
    piksellerinin sınırlayıcı kutusunu (bounding box) kullanır.

    margin_ratio: kırpılan kutunun etrafına bırakılacak küçük pay
    (görüntü boyutuna oranla), yazıyı tam kenardan kesmemek için.
    """
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    coords = cv2.findNonZero(thresh)

    if coords is None:
        # İçerik bulunamadıysa (tamamen boş görsel) kırpma yapma
        return gray

    x, y, w, h = cv2.boundingRect(coords)
    img_h, img_w = gray.shape[:2]

    margin_x = int(img_w * margin_ratio)
    margin_y = int(img_h * margin_ratio)

    x0 = max(0, x - margin_x)
    y0 = max(0, y - margin_y)
    x1 = min(img_w, x + w + margin_x)
    y1 = min(img_h, y + h + margin_y)

    return gray[y0:y1, x0:x1]


def _correct_illumination(gray, kernel_size=51):
    """
    Görselin bir kısmı diğerinden daha karanlık/aydınlık olduğunda (örn.
    telefon kamerasının gölgesi, eğik ışık) düzeltir. Threshold tek başına
    yetmediğinde (çok büyük/yumuşak aydınlatma farkları) faydalıdır.

    Yöntem: büyük bir Gaussian blur ile görselin "arka plan aydınlatma
    haritasını" tahmin eder, orijinalden bu haritayı çıkararak aydınlatma
    farkını giderir.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1  # GaussianBlur tek sayı kernel ister

    background = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    corrected = cv2.divide(gray, background, scale=255)
    return corrected


def preprocess_image(img, preprocessing_settings):
    """
    Config'teki preprocessing bloğuna göre görseli OCR'dan ÖNCE işler.
    Tüm motorlar (tesseract, easyocr, trocr, doctr) bu fonksiyonu çağırabilir
    — motora özel hiçbir şey içermez, girdi her zaman bir OpenCV (BGR) görsel,
    çıktı da öyle (ya da grayscale/binary, adımlara bağlı).

    TÜM adımlar varsayılan olarak KAPALI (false). Hangisinin gerçekten
    faydalı olduğu, baseline ölçüm + hata analizi yapıldıktan SONRA karar
    verilmeli — körlemesine hepsini açmak önerilmez. Kod burada hazır
    duruyor, ne zaman/hangisini açacağınız sizin kararınız.

    Adımlar SABİT bir sırada uygulanır (her biri kendinden öncekinin
    çıktısı üzerinde çalışır), çünkü sıra sonucu etkiler:

      1. resize                -> çok küçük görseli büyütür
      2. deskew                -> eğik metni düzeltir
      3. autocrop_border       -> kenar boşluklarını kırpar
      4. illumination_correct  -> aydınlatma/gölge farkını düzeltir
      5. grayscale             -> renkli -> gri tonlama
      6. clahe                 -> bölgesel kontrast artırma
      7. denoise               -> küçük gürültü/leke temizleme
      8. sharpen               -> kenarları belirginleştirme (bulanıklık için)
      9. morphology            -> ince/kopuk karakterleri birleştirme (dilate/erode)
      10. threshold            -> gri tonları siyah/beyaza indirgeme (binarization)

    Config örneği (hepsi opsiyonel, vermezsen o adım atlanır):

      preprocessing:
        resize: true
        resize_scale: 2.0                # büyütme katsayısı

        deskew: true                     # eğiklik düzeltme

        autocrop_border: true            # kenar boşluğu kırpma
        autocrop_margin_ratio: 0.01

        illumination_correct: true       # gölge/aydınlatma düzeltme
        illumination_kernel_size: 51

        grayscale: true

        clahe: true                      # kontrast artırma (CLAHE)
        clahe_clip_limit: 2.0
        clahe_grid_size: 8

        denoise: true

        sharpen: true                    # keskinleştirme

        morphology: true                 # dilate veya erode
        morphology_op: "dilate"          # "dilate" | "erode"
        morphology_kernel_size: 2

        threshold: true
        threshold_block_size: 35
        threshold_c: 11

    Hiçbir ayar verilmezse (preprocessing bloğu yoksa ya da hepsi false ise)
    görsel HAM haliyle döner — davranış öncekiyle birebir aynı kalır.

    Önemli: Tesseract gibi klasik motorlar bu adımlardan ciddi fayda görür.
    EasyOCR/doctr/TrOCR gibi derin öğrenme tabanlı motorlar ham görsele karşı
    daha dayanıklıdır, ama onlarda da fark gözlemlenebilir — bu yüzden
    fonksiyon motordan bağımsız tutuldu, her runner kendi config'inde
    preprocessing'i açıp kapatabilir.
    """
    if not preprocessing_settings:
        return img

    processed = img

    # --- 1. Resize: çok küçük çözünürlüklü görselleri büyütür ---
    if preprocessing_settings.get("resize", False):
        scale = preprocessing_settings.get("resize_scale", 2.0)
        processed = cv2.resize(
            processed, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )

    # --- 2. Deskew: eğik metni düzeltir (gri görsel üzerinde çalışır) ---
    if preprocessing_settings.get("deskew", False):
        if len(processed.shape) == 3:
            gray_for_deskew = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        else:
            gray_for_deskew = processed
        processed = _deskew(gray_for_deskew)

    # --- 3. Autocrop: kenar boşluklarını kırpar ---
    if preprocessing_settings.get("autocrop_border", False):
        if len(processed.shape) == 3:
            gray_for_crop = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        else:
            gray_for_crop = processed
        margin_ratio = preprocessing_settings.get("autocrop_margin_ratio", 0.01)
        processed = _autocrop_border(gray_for_crop, margin_ratio)

    # --- 4. Illumination correction: gölge/aydınlatma farkını giderir ---
    if preprocessing_settings.get("illumination_correct", False):
        if len(processed.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        kernel_size = preprocessing_settings.get("illumination_kernel_size", 51)
        processed = _correct_illumination(processed, kernel_size)

    # --- 5. Grayscale: renkli -> gri tonlama ---
    if preprocessing_settings.get("grayscale", False):
        if len(processed.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

    # --- 6. CLAHE: bölgesel kontrast artırma (histogram eşitleme) ---
    # Soluk mürekkep / düşük kontrastlı taramalarda yazı-arka plan farkını
    # güçlendirir. Threshold'dan farklı olarak gri tonları KORUR, yok etmez.
    if preprocessing_settings.get("clahe", False):
        if len(processed.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        clip_limit = preprocessing_settings.get("clahe_clip_limit", 2.0)
        grid_size = preprocessing_settings.get("clahe_grid_size", 8)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
        processed = clahe.apply(processed)

    # --- 7. Denoise: küçük gürültü/leke temizleme ---
    if preprocessing_settings.get("denoise", False):
        # fastNlMeansDenoising gri görsel ister; renkliyse önce gri yapılır.
        if len(processed.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        processed = cv2.fastNlMeansDenoising(processed, h=10)

    # --- 8. Sharpen: kenarları belirginleştirme (bulanık/motion-blur görsellerde) ---
    if preprocessing_settings.get("sharpen", False):
        if len(processed.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        # Unsharp mask: orijinalden bulanıklaştırılmış halini çıkarıp
        # orijinale geri ekleyerek kenarları güçlendirir.
        blurred = cv2.GaussianBlur(processed, (0, 0), 3)
        processed = cv2.addWeighted(processed, 1.5, blurred, -0.5, 0)

    # --- 9. Morphology: ince/kopuk karakterleri birleştirme ya da küçük
    # gürültü noktalarını silme ---
    if preprocessing_settings.get("morphology", False):
        if len(processed.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        op = preprocessing_settings.get("morphology_op", "dilate")
        kernel_size = preprocessing_settings.get("morphology_kernel_size", 2)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        if op == "erode":
            processed = cv2.erode(processed, kernel, iterations=1)
        else:  # "dilate" varsayılan
            processed = cv2.dilate(processed, kernel, iterations=1)

    # --- 10. Threshold: gri tonları siyah/beyaza indirgeme (binarization) ---
    if preprocessing_settings.get("threshold", False):
        if len(processed.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        block_size = preprocessing_settings.get("threshold_block_size", 35)
        c_value = preprocessing_settings.get("threshold_c", 11)
        # Adaptive threshold: sabit eşik yerine bölgesel ortalama kullanır,
        # böylece görselin bir kısmı diğerinden daha gölgeli/aydınlık olsa
        # bile (örn. telefonla çekilmiş bir dekont) her bölge kendi yerel
        # kontrastına göre siyah/beyaza ayrılır.
        processed = cv2.adaptiveThreshold(
            processed, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            block_size, c_value,
        )

    return processed


def save_preprocessed_image(ocr_input, preprocessed_dir, img_name):
    """
    OCR motoruna verilmeden önceki (preprocess_image() sonrası) görseli
    diske kaydeder. Preprocessing kapalıysa bu, görselin ham/orijinal hali
    olur — her durumda "motora ne gitti" diske yazılır.

    cv2.imwrite tek kanallı (gri/binary) ve 3 kanallı (BGR) görselleri
    sorunsuz kaydeder, bu yüzden burada ekstra bir dönüşüm gerekmiyor —
    fonksiyon sadece dosya adı/yol mantığını tek yerde tutmak için var,
    4 runner'da aynı isimlendirme kuralını (prefix "preprocessed_") garantiler.
    """
    out_path = os.path.join(preprocessed_dir, f"preprocessed_{img_name}")
    cv2.imwrite(out_path, ocr_input)
    return out_path


def get_viz_dirs(config_path, base_output_dir="outputs"):
    """
    config_path'ten (örn. configurations/tesseract/model_v1.yaml) viz
    klasörlerini türetir ve oluşturur:
        outputs/{engine}/{model_name}/viz/highlighted
        outputs/{engine}/{model_name}/viz/masked
        outputs/{engine}/{model_name}/viz/preprocessed

    main.py'deki process_pipeline(engine, model_name) ile birebir aynı klasör
    yapısına yazılır; farklı model_v1/model_v2 viz çıktıları birbirinin
    üstüne yazmaz.

    preprocessed_dir: OCR motoruna verilmeden önce (preprocess_image()
    sonrası) görselin kendisi buraya kaydedilir — preprocessing kapalı
    olsa bile (o zaman görsel hâliyle/ham olarak) kaydedilir, böylece her
    görsel için "motora ne gitti" her zaman görülebilir.

    Döndürür: (highlighted_dir, masked_dir, preprocessed_dir)
    """
    from pathlib import Path

    config_p = Path(config_path)
    engine = config_p.parent.name
    model_name = config_p.stem

    viz_dir = os.path.join(base_output_dir, engine, model_name, "viz")
    highlighted_dir = os.path.join(viz_dir, "highlighted")
    masked_dir = os.path.join(viz_dir, "masked")
    preprocessed_dir = os.path.join(viz_dir, "preprocessed")

    os.makedirs(highlighted_dir, exist_ok=True)
    os.makedirs(masked_dir, exist_ok=True)
    os.makedirs(preprocessed_dir, exist_ok=True)

    return highlighted_dir, masked_dir, preprocessed_dir