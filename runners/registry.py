"""
runners/registry.py

OCR motorlarının TEK kayıt noktası. main.py bu dosyadaki ENGINES sözlüğünü
okuyarak hangi motorların var olduğunu, her birinin runner fonksiyonunu ve
modelinin nasıl (bir kez) yükleneceğini öğrenir.

=== YENİ BİR MOTOR EKLEMEK İÇİN ===
main.py'ye HİÇ DOKUNMANIZ GEREKMİYOR. Sadece:

  1. runners/run_<motor_adi>.py dosyanızı diğerleriyle aynı sözleşmeye göre
     yazın: run_<motor_adi>(image_path, config_path, **opsiyonel_model_kwarglari)
     ve {"text", "load_time_seconds", "execution_time_seconds", ...} içeren
     bir dict döndürsün.

  2. Aşağıdaki ENGINES sözlüğüne YENİ BİR SATIR ekleyin (örnek aşağıda).

main.py, process_pipeline() çağrıldığında otomatik olarak:
  - registry'den doğru run_function'ı bulur
  - varsa loader()'ı BİR KEZ çağırıp modeli/reader'ı kurar
  - bu yüklenmiş nesneyi her görsel için run_function'a doğru kwarg
    adlarıyla (shared_kwargs) geçirir

=== ALANLARIN ANLAMI ===

run_function:
    Runner modülündeki ana fonksiyon (örn. run_tesseract).

loader:
    Modeli/reader'ı/processor'ı BİR KEZ kuran fonksiyon. config_path alır,
    yüklenen nesne(leri) bir dict olarak döndürür — dict'in anahtarları
    shared_kwargs ile eşleşmeli. Gerçek bir model yükleme maliyeti yoksa
    (örn. Tesseract) None bırakılır; main.py bu durumda hiçbir şey
    yüklemeden, run_function'ı sadece (image_path, config_path) ile çağırır.

shared_kwargs:
    loader()'ın döndürdüğü dict'teki anahtarların, run_function'a hangi
    parametre adlarıyla geçirileceği. Örn. EasyOCR için {"reader": "reader"}
    -> loader {"reader": <Reader nesnesi>} döndürür, main.py bunu
    run_easyocr(img, config, reader=<Reader nesnesi>) olarak çağırır.
"""

from runners.run_tesseract import run_tesseract
from runners.run_easyocr import run_easyocr
from runners.run_trocr import run_trocr
from runners.run_doctr import run_doctr
from runners.run_rapidocr import run_rapidocr
from runners.run_paddleocr import run_paddleocr

from runners._common import load_config


def _load_easyocr_reader(config_path):
    """EasyOCR Reader'ını config'teki dil/gpu ayarlarına göre BİR KEZ kurar."""
    import easyocr
    import torch

    config = load_config(config_path)
    reader_settings = config.get("reader_settings", {})
    languages = reader_settings.get("languages", ["tr", "en"])
    gpu_setting = reader_settings.get("gpu", "auto")
    gpu = torch.cuda.is_available() if gpu_setting == "auto" else bool(gpu_setting)

    return {"reader": easyocr.Reader(languages, gpu=gpu)}


def _load_doctr_model(config_path):
    """doctr modelini config'teki mimari ayarlarına göre BİR KEZ kurar."""
    from doctr.models import ocr_predictor
    import torch

    config = load_config(config_path)
    model_settings = config.get("model_settings", {})
    det_arch = model_settings.get("det_arch", "db_resnet50")
    reco_arch = model_settings.get("reco_arch", "crnn_vgg16_bn")
    pretrained = model_settings.get("pretrained", True)
    gpu_setting = model_settings.get("gpu", "auto")
    gpu = torch.cuda.is_available() if gpu_setting == "auto" else bool(gpu_setting)

    model = ocr_predictor(det_arch=det_arch, reco_arch=reco_arch, pretrained=pretrained)
    if gpu:
        model = model.cuda()

    return {"model": model}


def _load_trocr_model(config_path):
    """TrOCR processor+model'ini config'teki model adına göre BİR KEZ kurar."""
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    import torch

    config = load_config(config_path)
    settings = config.get("ocr_settings", {})
    model_name = settings.get("model_name", "microsoft/trocr-base-handwritten")
    gpu_setting = settings.get("gpu", "auto")
    gpu = torch.cuda.is_available() if gpu_setting == "auto" else bool(gpu_setting)
    device = "cuda" if gpu else "cpu"

    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    model.to(device)

    return {"processor": processor, "model": model}


def _load_rapidocr_engine(config_path):
    """RapidOCR motorunu config'teki dil/model_selection ayarlarına göre BİR KEZ kurar."""
    from rapidocr import RapidOCR
    from runners.run_rapidocr import build_rapidocr_params

    config = load_config(config_path)
    settings = config.get("ocr_settings", {})
    lang_type = settings.get("lang_type", "tr")

    # build_rapidocr_params, run_rapidocr.py ile AYNI fonksiyon — burada
    # tekrar tanımlamak yerine import ediyoruz, böylece main.py üzerinden
    # (registry ile) çalıştırdığınızda da run_rapidocr.py'yi DOĞRUDAN
    # çağırdığınızdaki BİREBİR AYNI model_type/model_path seçimi uygulanır.
    rapidocr_params = {"Rec.lang_type": lang_type}
    rapidocr_params.update(build_rapidocr_params(settings))

    engine = RapidOCR(params=rapidocr_params)

    return {"engine": engine}


def _load_paddleocr_engine(config_path):
    """
    PaddleOCR motorunu config'teki dil ayarına göre BİR KEZ kurar.

    NOT: PaddleOCR'ın model dosyaları RapidOCR'daki gibi pakete gömülü
    DEĞİLDİR — ilk kullanımda internetten indirilir. Offline/kapalı bir
    ağda bu çağrı hata verir (network bağlantısı şart).
    """
    from paddleocr import PaddleOCR

    config = load_config(config_path)
    settings = config.get("ocr_settings", {})
    lang = settings.get("lang", "tr")
    use_doc_orientation_classify = settings.get("use_doc_orientation_classify", False)
    use_doc_unwarping = settings.get("use_doc_unwarping", False)
    use_textline_orientation = settings.get("use_textline_orientation", False)

    engine = PaddleOCR(
        lang=lang,
        use_doc_orientation_classify=use_doc_orientation_classify,
        use_doc_unwarping=use_doc_unwarping,
        use_textline_orientation=use_textline_orientation,
    )

    return {"engine": engine}


# =============================================================================
# TEK KAYIT NOKTASI — yeni motor eklerken sadece buraya bir satır ekleyin
# =============================================================================
ENGINES = {
    "tesseract": {
        "run_function": run_tesseract,
        "loader": None,  # Gerçek bir model yükleme maliyeti yok
        "shared_kwargs": {},
    },
    "easyocr": {
        "run_function": run_easyocr,
        "loader": _load_easyocr_reader,
        "shared_kwargs": {"reader": "reader"},
    },
    "doctr": {
        "run_function": run_doctr,
        "loader": _load_doctr_model,
        "shared_kwargs": {"model": "model"},
    },
    "trocr": {
        "run_function": run_trocr,
        "loader": _load_trocr_model,
        "shared_kwargs": {"processor": "processor", "model": "model"},
    },
    "rapidocr": {
        "run_function": run_rapidocr,
        "loader": _load_rapidocr_engine,
        "shared_kwargs": {"engine": "engine"},
    },
    "paddleocr": {
        "run_function": run_paddleocr,
        "loader": _load_paddleocr_engine,
        "shared_kwargs": {"engine": "engine"},
    },

    # --- Yeni bir motor eklemek için örnek satır (kopyalayıp düzenleyin) ---
    # "yeni_motor": {
    #     "run_function": run_yeni_motor,
    #     "loader": _load_yeni_motor_model,  # ya da gerçek yükleme yoksa None
    #     "shared_kwargs": {"model": "model"},
    # },
}