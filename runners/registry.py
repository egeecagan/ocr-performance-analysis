"""
runners/registry.py

The SINGLE registration point for OCR engines. main.py reads the ENGINES
dictionary in this file to learn which engines exist, each one's runner
function, and how its model should be loaded (once).

=== TO ADD A NEW ENGINE ===
You do NOT need to touch main.py. Just:

  1. Write your runners/run_<engine_name>.py file following the same
     contract as the others: run_<engine_name>(image_path, config_path,
     **optional_model_kwargs) and return a dict containing
     {"text", "load_time_seconds", "execution_time_seconds", ...}.

  2. Add a NEW ENTRY to the ENGINES dictionary below (example below).

When process_pipeline() is called, main.py automatically:
  - finds the correct run_function from the registry
  - if present, calls loader() ONCE to set up the model/reader
  - passes this loaded object into run_function for every image, using
    the correct kwarg names (shared_kwargs)

=== WHAT EACH FIELD MEANS ===

run_function:
    The main function in the runner module (e.g. run_tesseract).

loader:
    A function that sets up the model/reader/processor ONCE. Takes
    config_path, returns the loaded object(s) as a dict — the dict's keys
    must match shared_kwargs. Left as None if there's no real
    model-loading cost (e.g. Tesseract); in this case main.py loads
    nothing and calls run_function with just (image_path, config_path).

shared_kwargs:
    Maps the keys in the dict returned by loader() to the parameter names
    run_function expects them under. E.g. for EasyOCR, {"reader": "reader"}
    -> loader returns {"reader": <Reader object>}, and main.py calls this
    as run_easyocr(img, config, reader=<Reader object>).
"""

from runners.run_tesseract import run_tesseract
from runners.run_easyocr import run_easyocr
from runners.run_doctr import run_doctr
from runners.run_rapidocr import run_rapidocr

try:
    from runners.run_paddleocr import run_paddleocr
    HAS_PADDLEOCR = True
except (ImportError, ModuleNotFoundError):
    run_paddleocr = None
    HAS_PADDLEOCR = False

from runners._common import load_config


def _load_easyocr_reader(config_path):
    """Sets up the EasyOCR Reader ONCE, based on the language/gpu settings in the config."""
    import easyocr
    import torch
    from runners._common import filter_valid_kwargs

    config = load_config(config_path)
    reader_settings = config.get("reader_settings", {})
    languages = reader_settings.get("languages", ["tr", "en"])
    gpu_setting = reader_settings.get("gpu", "auto")
    gpu = torch.cuda.is_available() if gpu_setting == "auto" else bool(gpu_setting)

    # Same mechanism as run_easyocr.py: any extra key under reader_settings
    # (e.g. recog_network, detector, recognizer, download_enabled) that
    # easyocr.Reader actually accepts is passed through automatically — no
    # code change needed when you add new settings to the config.
    extra_reader_kwargs = filter_valid_kwargs(easyocr.Reader.__init__, reader_settings)
    extra_reader_kwargs.pop("lang_list", None)
    extra_reader_kwargs.pop("gpu", None)

    return {"reader": easyocr.Reader(languages, gpu=gpu, **extra_reader_kwargs)}


def _load_doctr_model(config_path):
    """Sets up the doctr model ONCE, based on the architecture settings in the config."""
    from doctr.models import ocr_predictor
    import torch
    from runners._common import filter_valid_kwargs

    config = load_config(config_path)
    model_settings = config.get("model_settings", {})
    det_arch = model_settings.get("det_arch", "db_resnet50")
    reco_arch = model_settings.get("reco_arch", "crnn_vgg16_bn")
    pretrained = model_settings.get("pretrained", True)
    gpu_setting = model_settings.get("gpu", "auto")
    gpu = torch.cuda.is_available() if gpu_setting == "auto" else bool(gpu_setting)

    # Same mechanism as run_doctr.py: any extra key under model_settings
    # (e.g. assume_straight_pages, preserve_aspect_ratio, detect_orientation,
    # detect_language) that ocr_predictor actually accepts is passed
    # through automatically — no code change needed when you add new
    # settings to the config. ocr_predictor accepts **kwargs, so we
    # explicitly remove our own config-only field (gpu) which would
    # otherwise be passed through unfiltered.
    extra_model_kwargs = filter_valid_kwargs(ocr_predictor, model_settings)
    extra_model_kwargs.pop("det_arch", None)
    extra_model_kwargs.pop("reco_arch", None)
    extra_model_kwargs.pop("pretrained", None)
    extra_model_kwargs.pop("gpu", None)
    extra_model_kwargs.pop("font_path", None)

    model = ocr_predictor(det_arch=det_arch, reco_arch=reco_arch, pretrained=pretrained, **extra_model_kwargs)
    if gpu:
        model = model.cuda()

    return {"model": model}


def _load_rapidocr_engine(config_path):
    """Sets up the RapidOCR engine ONCE, based on the language/model_selection settings in the config."""
    from rapidocr import RapidOCR
    from runners.run_rapidocr import build_rapidocr_params

    config = load_config(config_path)
    settings = config.get("ocr_settings", {})
    lang_type = settings.get("lang_type", "tr")

    # build_rapidocr_params is the SAME function as in run_rapidocr.py — we
    # import it here instead of redefining it, so that running via main.py
    # (through the registry) applies the EXACT SAME model_type/model_path
    # selection as when you call run_rapidocr.py DIRECTLY.
    rapidocr_params = {"Rec.lang_type": lang_type}
    rapidocr_params.update(build_rapidocr_params(settings))

    engine = RapidOCR(params=rapidocr_params)

    return {"engine": engine}


def _load_paddleocr_engine(config_path):
    """Sets up the PaddleOCR engine ONCE, based on the language/architecture settings in the config."""
    from paddleocr import PaddleOCR
    from runners._common import filter_valid_kwargs
    
    config = load_config(config_path)
    settings = config.get("ocr_settings", {})
    lang = settings.get("lang", "tr")
    use_doc_orientation_classify = settings.get("use_doc_orientation_classify", False)
    use_doc_unwarping = settings.get("use_doc_unwarping", False)
    use_textline_orientation = settings.get("use_textline_orientation", False)
    
    extra_init_kwargs = filter_valid_kwargs(PaddleOCR.__init__, settings)
    extra_init_kwargs.pop("lang", None)
    extra_init_kwargs.pop("use_doc_orientation_classify", None)
    extra_init_kwargs.pop("use_doc_unwarping", None)
    extra_init_kwargs.pop("use_textline_orientation", None)
    extra_init_kwargs.pop("font_path", None)
    
    engine = PaddleOCR(
        lang=lang,
        use_doc_orientation_classify=use_doc_orientation_classify,
        use_doc_unwarping=use_doc_unwarping,
        use_textline_orientation=use_textline_orientation,
        **extra_init_kwargs
    )
    return {"engine": engine}


# =============================================================================
# THE SINGLE REGISTRATION POINT — just add one entry here when adding a new engine
# =============================================================================
ENGINES = {
    "tesseract": {
        "run_function": run_tesseract,
        "loader": None,  # No real model-loading cost
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
    "rapidocr": {
        "run_function": run_rapidocr,
        "loader": _load_rapidocr_engine,
        "shared_kwargs": {"engine": "engine"},
    },
}

if HAS_PADDLEOCR:
    ENGINES["paddleocr"] = {
        "run_function": run_paddleocr,
        "loader": _load_paddleocr_engine,
        "shared_kwargs": {"engine": "engine"},
    }

    # --- Example entry for adding a new engine (copy and edit) ---
    # "new_engine": {
    #     "run_function": run_new_engine,
    #     "loader": _load_new_engine_model,  # or None if there's no real loading cost
    #     "shared_kwargs": {"model": "model"},
    # },