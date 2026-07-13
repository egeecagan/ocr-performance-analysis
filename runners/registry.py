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

  2. If the engine has a real model/reader/processor to set up, write a
     _load_<engine_name>(config_path) function in this file (see the
     existing _load_* functions below for the pattern) that returns
     {"model": <loaded object>}. Skip this step if there's nothing worth
     loading once (e.g. Tesseract) — use loader: None instead.

  3. Add a NEW ENTRY to the ENGINES dictionary below (example below).

When process_pipeline() is called, main.py automatically:
  - finds the correct run_function from the registry
  - if present, calls loader() ONCE to set up the model/reader
  - passes the loaded object into run_function for every image as the
    `model` kwarg

=== WHAT EACH FIELD MEANS ===

run_function:
    The main function in the runner module (e.g. run_tesseract). Not
    implemented here — each engine defines its own in runners/run_<engine>.py.
    It takes an image (plus config and, if any, the loaded model) and
    returns the OCR result dict.

loader:
    A function that sets up the model/reader/processor once. Takes
    config_path, validates the config against what the engine actually
    supports (all engines except Tesseract, which needs no such check),
    and returns {"model": <loaded object>}. Left as None if there's no
    real model-loading cost (e.g. Tesseract); in this case main.py loads
    nothing and calls run_function with just (image_path, config_path).

    By convention every loader returns its object under the key "model",
    and every run_<engine_name> accepts it as a `model` kwarg — this is
    what lets main.py wire loader -> run_function without any per-engine
    mapping.
"""

from runners.run_tesseract import run_tesseract
from runners.run_easyocr import run_easyocr
from runners.run_doctr import run_doctr
from runners.run_rapidocr import run_rapidocr

from runners._common import load_config


def _load_easyocr_reader(config_path):
    import easyocr
    import torch
    from runners._common import filter_valid_kwargs

    config = load_config(config_path)
    reader_settings = config.get("reader_settings", {})
    languages = reader_settings.get("languages", ["tr", "en"])
    gpu_setting = reader_settings.get("gpu", "auto")
    gpu = torch.cuda.is_available() if gpu_setting == "auto" else bool(gpu_setting)

    extra_reader_kwargs = filter_valid_kwargs(easyocr.Reader.__init__, reader_settings)
    # It checks the valid parameters for the constructor of the easyocr and if something
    # does not make sense in the reader_settings object it wont take that

    extra_reader_kwargs.pop("lang_list", None)
    extra_reader_kwargs.pop("gpu", None)

    return {"model": easyocr.Reader(languages, gpu=gpu, **extra_reader_kwargs)}


def _load_doctr_model(config_path):
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
    from rapidocr import RapidOCR
    from runners.run_rapidocr import build_rapidocr_params

    config = load_config(config_path)
    settings = config.get("ocr_settings", {})
    lang_type = settings.get("lang_type", "tr")

    rapidocr_params = {"Rec.lang_type": lang_type}
    rapidocr_params.update(build_rapidocr_params(settings))

    engine = RapidOCR(params=rapidocr_params)

    return {"model": engine}

ENGINES = {
    "tesseract": {
        "run_function": run_tesseract,
        "loader": None,  # No real model-loading cost
    },
    "easyocr": {
        "run_function": run_easyocr,
        "loader": _load_easyocr_reader,
    },
    "doctr": {
        "run_function": run_doctr,
        "loader": _load_doctr_model,
    },
    "rapidocr": {
        "run_function": run_rapidocr,
        "loader": _load_rapidocr_engine,
    },
}