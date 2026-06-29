import os
import time
import uuid
import cv2
import torch
from doctr.models import ocr_predictor
from doctr.io import DocumentFile

from runners._common import preprocess_image, get_viz_dirs, load_config, save_preprocessed_image


def run_doctr(image_path, config_path, model=None):
    config = load_config(config_path)
    model_settings = config.get("model_settings", {})
    preprocessing_settings = config.get("preprocessing", {}) or {}

    det_arch = model_settings.get("det_arch", "db_resnet50")
    reco_arch = model_settings.get("reco_arch", "crnn_vgg16_bn")
    pretrained = model_settings.get("pretrained", True)

    if not pretrained:
        # pretrained=False -> doctr RASTGELE BAŞLATILMIŞ (eğitilmemiş) bir
        # model kurar. Bu, geçerli bir doctr özelliği (örn. kendi
        # verinizle sıfırdan eğitim yapacaksanız kullanılır) ama OCR
        # KARŞILAŞTIRMASI için kullanılırsa tamamen anlamsız/çöp metin
        # üretir ve kod HATA VERMEZ — sessizce kötü sonuç döner. Burada
        # konsola net bir uyarı basıyoruz ki yanlışlıkla bu ayarla bir
        # rapor karşılaştırması yapılıp "doctr çok kötü" diye hatalı bir
        # sonuca ulaşılmasın.
        print(
            f"[UYARI] doctr config'inde pretrained=false ayarlı "
            f"({config_path}). Model EĞİTİLMEMİŞ ağırlıklarla başlatılacak "
            f"ve OCR sonuçları anlamsız çıkacaktır. Bu kasıtlı değilse "
            f"config'te pretrained: true yapın."
        )

    # --- GPU seçimi ---
    # EasyOCR'daki ile aynı prensip: "auto" -> otomatik tespit, true/false
    # -> zorlama. doctr modelleri PyTorch tabanlı olduğu için aynı
    # torch.cuda.is_available() kontrolü kullanılıyor.
    gpu_setting = model_settings.get("gpu", "auto")
    if gpu_setting == "auto":
        gpu = torch.cuda.is_available()
    else:
        gpu = bool(gpu_setting)

    # --- load_time_seconds ---
    # EasyOCR/TrOCR'daki gibi burada da GERÇEK bir maliyet var:
    # ocr_predictor(...) detection + recognition modellerini diskten/
    # internetten yükler. model dışarıdan enjekte edilmemişse (main.py
    # her görsel için modeli YENİDEN oluşturuyorsa) bu maliyet her çağrıda
    # tekrar tekrar ödenir.
    if model is None:
        load_start = time.time()
        try:
            model = ocr_predictor(det_arch=det_arch, reco_arch=reco_arch, pretrained=pretrained)
            if gpu:
                model = model.cuda()
        except Exception as e:
            raise RuntimeError(
                f"doctr modeli oluşturulamadı (det_arch={det_arch}, "
                f"reco_arch={reco_arch}, config: {config_path}). "
                f"Orijinal hata: {e}"
            ) from e
        load_time = round(time.time() - load_start, 4)
    else:
        # Model dışarıdan (main.py'de bir kez oluşturulup tüm görseller
        # için tekrar kullanılıyorsa) geldiyse, bu çağrıda yükleme yok.
        load_time = 0.0

    # --- image_load_time_seconds: görseli diskten okuma süresi ---
    image_load_start = time.time()
    img = cv2.imread(image_path)
    image_load_time = round(time.time() - image_load_start, 4)

    if img is None:
        # cv2.imread bulunamayan/bozuk dosyada sessizce None döner; burada
        # bunu net bir hataya çeviriyoruz ki toplu işlemde hangi dosyanın
        # sorunlu olduğu hemen görülsün. (Bu kontrol sadece görselleştirme
        # için kullanılan img'i koruyor; asıl OCR girdisi DocumentFile ile
        # ayrıca okunuyor, aşağıda onu da kontrol ediyoruz.)
        raise ValueError(
            f"Görsel okunamadı (bozuk dosya, desteklenmeyen format ya da "
            f"yol hatalı olabilir): {image_path}"
        )

    # --- preprocessing_time_seconds ---
    # Sadece preprocess_image() çağrısının kendisi ölçülüyor. Geçici dosyaya
    # yazma (aşağıda, preprocessing açıkken) ayrı bir disk I/O maliyeti —
    # onu da execution_time'a değil, total_time'ın genel toplamına dahil
    # ediyoruz (aşağıda doc_write_time olarak).
    preprocessing_start = time.time()
    ocr_input = preprocess_image(img, preprocessing_settings)
    preprocessing_time = round(time.time() - preprocessing_start, 4)

    # --- Kanal güvenliği ---
    # preprocess_image bazı adımlardan sonra tek kanallı (2D) görsel
    # döndürebilir. doctr'ın DocumentFile.from_images() bir dosya yolu
    # bekliyor, numpy array değil — bu yüzden işlenmiş görseli geçici bir
    # dosyaya yazıp doctr'a o yolu veriyoruz. Tek kanallı bir görseli
    # diske PNG olarak yazmadan önce 3 kanala çeviriyoruz, çünkü doctr'ın
    # iç pipeline'ı RGB bekler.
    if len(ocr_input.shape) == 2:
        ocr_input = cv2.cvtColor(ocr_input, cv2.COLOR_GRAY2BGR)

    preprocessing_applied = bool(preprocessing_settings) and any(
        v is True for k, v in preprocessing_settings.items() if isinstance(v, bool)
    )

    doc_write_time = 0.0
    if preprocessing_applied:
        # İşlenmiş görseli geçici bir dosyaya yazıyoruz çünkü
        # DocumentFile.from_images() bir dosya YOLU bekler, ham numpy
        # array kabul etmez.
        #
        # Dosya adına bir uuid ekliyoruz: şu an sıralı (sequential)
        # çalıştığınız için risk yok, ama ileride bu fonksiyonu paralel
        # (multiprocessing/threading) çağırırsanız ve iki farklı klasörden
        # gelen görseller aynı isme sahipse (örn. iki ayrı "ornek.png"),
        # uuid olmadan bu geçici dosyalar birbirinin üzerine yazılabilir.
        # uuid ile her çağrı kendi benzersiz dosyasını kullanır.
        doc_write_start = time.time()
        tmp_dir = os.path.join("outputs", "_tmp_preprocessed")
        os.makedirs(tmp_dir, exist_ok=True)
        unique_id = uuid.uuid4().hex[:8]
        tmp_path = os.path.join(tmp_dir, f"pre_{unique_id}_{os.path.basename(image_path)}")
        cv2.imwrite(tmp_path, ocr_input)
        doc_input_path = tmp_path
        doc_write_time = round(time.time() - doc_write_start, 4)
    else:
        doc_input_path = image_path

    doc_load_start = time.time()
    doc = DocumentFile.from_images(doc_input_path)
    doc_load_time = round(time.time() - doc_load_start, 4)

    actual_start = time.time()  # Süre BURADA başlar — sadece OCR motorunun
    # kendisi (model çağrısı + render) ölçülüyor. Görsel okuma, preprocessing,
    # geçici dosya yazma ve DocumentFile.from_images (disk I/O) hepsi ayrı
    # alanlarda (image_load_time_seconds, preprocessing_time_seconds,
    # doc_write_time, doc_load_time) tutuluyor — diğer 3 motordaki ile
    # tutarlı prensip: "motorun kendisi ne kadar sürdü" net ölçülüyor.
    result = model(doc)
    raw_text = result.render()
    execution_time = round(time.time() - actual_start, 4)  # Süre BURADA donar

    # --- Metin normalizasyonu ---
    # result.render(), doctr'ın sayfa/blok/satır yapısını '\n' (satır sonu)
    # karakterleriyle ayrılmış çok satırlı bir metne çevirir. Tesseract ve
    # EasyOCR ise " ".join(...) ile TEK SATIRLIK metin üretiyor. Eğer ground
    # truth karşılaştırması (CER/WER) yaparken bu farkı normalize etmezsek,
    # doctr'ın çıktısı sadece biçim farkından dolayı (içerik aynı olsa bile)
    # yapay olarak daha "yanlış" görünebilir. Burada hem ham (çok satırlı,
    # render() çıktısının orijinali) hem normalize edilmiş (tek satır,
    # diğer motorlarla aynı formatta) halini saklıyoruz — raporlama/
    # accuracy hesaplaması normalize edilmiş alanı kullanmalı, ama ham
    # halini de kaybetmiyoruz (örn. layout'un kendisi ilginizi çekerse).
    text = " ".join(raw_text.split())

    # Geçici preprocessing dosyasını temizle
    if preprocessing_applied:
        try:
            os.remove(doc_input_path)
        except OSError:
            pass

    # --- Buradan sonrası süreye dahil değil: görselleştirme/raporlama ---
    highlighted_dir, masked_dir, preprocessed_dir = get_viz_dirs(config_path)

    img_name = os.path.basename(image_path)
    h, w, _ = img.shape

    # OCR motoruna giden görseli (preprocessing kapalıysa ham hali) her
    # zaman kaydediyoruz. Not: doctr'da ocr_input zaten ayrıca geçici bir
    # dosyaya yazılıp silinmişti (DocumentFile.from_images için) — bu,
    # ondan AYRI ve kalıcı bir kopya, viz/preprocessed altında durur.
    save_preprocessed_image(ocr_input, preprocessed_dir, img_name)

    # Highlight/mask çizimleri orijinal RENKLİ görsel üzerine yapılır —
    # preprocessing sadece OCR motoruna giden girdiyi etkiler.
    overlay = img.copy()

    confidences = []

    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    geom = word.geometry
                    if isinstance(geom[0], tuple):
                        x1, y1 = geom[0]
                        x2, y2 = geom[1]
                    else:
                        x1, y1, x2, y2 = geom

                    x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)

                    # doctr her kelime için word.confidence (0-1 arası) döndürür.
                    if hasattr(word, "confidence") and word.confidence is not None:
                        confidences.append(word.confidence)

    # --- avg_confidence ---
    # doctr'ın her kelime için verdiği confidence'ı (0-1) Tesseract/EasyOCR
    # ile aynı eksende (0-100) kıyaslanabilsin diye burada da çeviriyoruz.
    # Hiç kelime tespit edilemediyse (gerçek bir "0 sonuç" durumu) 0.0;
    # kelimeler var ama confidence hiç gelmediyse (beklenmeyen bir doctr
    # sürümü/durumu) None — "ölçülmedi" ile "gerçekten kötü" karışmasın.
    total_words = sum(len(line.words) for page in result.pages for block in page.blocks for line in block.lines)
    if total_words == 0:
        avg_confidence = 0.0
    elif not confidences:
        avg_confidence = None
    else:
        avg_confidence = round((sum(confidences) / len(confidences)) * 100, 2)

    alpha = 0.3
    highlighted_img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    cv2.imwrite(os.path.join(highlighted_dir, f"highlighted_{img_name}"), highlighted_img)

    synthetic_pages = result.synthesize()
    cv2.imwrite(os.path.join(masked_dir, f"masked_{img_name}"), synthetic_pages[0])

    total_time = round(
        image_load_time + preprocessing_time + doc_write_time + doc_load_time + execution_time, 4
    )

    return {
        "text": text,
        # raw_text: result.render()'ın ORİJİNAL çok satırlı hali (satır
        # sonları korunmuş). "text" alanı diğer motorlarla (Tesseract,
        # EasyOCR) tutarlı tek-satır formatında — accuracy/CER/WER
        # hesaplaması "text" alanını kullanmalı. raw_text sadece referans/
        # debug amaçlı, layout bilgisini görmek isterseniz işe yarar.
        "raw_text": raw_text,
        "load_time_seconds": load_time,
        "image_load_time_seconds": image_load_time,
        "preprocessing_time_seconds": preprocessing_time,
        # doctr'a özgü iki ek süre: geçici dosya yazma (sadece preprocessing
        # açıkken > 0) ve DocumentFile.from_images (her zaman ölçülür).
        # Diğer 3 motorda bu adımlar yok, bu yüzden sadece doctr'da var.
        "doc_write_time_seconds": doc_write_time,
        "doc_load_time_seconds": doc_load_time,
        "execution_time_seconds": execution_time,
        # total_time_seconds: görsel okuma + preprocessing + geçici dosya
        # yazma + DocumentFile yükleme + OCR motorunun kendisi. Yani
        # kullanıcının uçtan uca gerçekte beklediği toplam süre.
        # load_time_seconds (model yükleme) DAHIL DEĞİL — o, motor için bir
        # kez ödenen, main.py'de paylaşılan bir maliyet.
        "total_time_seconds": total_time,
        "avg_confidence": avg_confidence,
        # Tesseract/EasyOCR ile tutarlı format: motor adı + mimari bilgisi.
        "model_used": f"doctr-{det_arch}-{reco_arch}",
        # "eski donanım" senaryonuzda süre rakamlarının hangi donanımda
        # ölçüldüğünü bilmek kritik (EasyOCR'daki ile aynı gerekçe).
        "device_used": "gpu" if gpu else "cpu",
        "settings_used": {
            "det_arch": det_arch,
            "reco_arch": reco_arch,
            "pretrained": pretrained,
        },
        "preprocessing_used": preprocessing_settings,
    }


if __name__ == "__main__":
    # main.py'nin çağırdığı gibi: config_path = f'configurations/{engine}/{model_name}.yaml'
    result = run_doctr(
        image_path="inputs/images/ornek.png",
        config_path="configurations/doctr/model_v1.yaml",
    )
    print(result)

    # --- ÖNEMLİ: main.py entegrasyonu için ---
    # EasyOCR'daki ile AYNI sorun burada da var: main.py her görsel için
    # run_doctr(img, config) çağırırken `model` parametresini hiç
    # vermiyorsa, ocr_predictor(...) HER GÖRSEL İÇİN YENİDEN YÜKLENİR.
    # doctr'ın detection+recognition modelleri EasyOCR'dan bile daha ağır
    # olabilir — bu yüzden main.py güncellemesinde model'i BİR KEZ kurup
    # tüm görseller için paylaşmamız gerekiyor:
    #
    #   from doctr.models import ocr_predictor
    #   shared_model = ocr_predictor(det_arch="db_resnet50", reco_arch="crnn_vgg16_bn", pretrained=True)
    #   for img_path in tüm_görseller:
    #       result = run_doctr(img_path, config_path, model=shared_model)