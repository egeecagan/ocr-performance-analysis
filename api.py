"""
api.py — OCR Performance Analysis FastAPI Backend

Endpoint'ler:
    POST /process   : Gorsel yukle + engine/model_name sec -> OCR sonucu don
    GET  /report    : comparison_report.json icerigini don
    GET  /engines   : Kullanilabilir engine ve model listesini don

Calistirma:
    .\\venv\\Scripts\\uvicorn api:app --reload --port 8000
"""

import json
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# main.py'den OCR fonksiyonlarini ve yol sabitlerini aktar
from main import (
    process_single_image,
    BASE_DIR,
    OUTPUTS_DIR,
    CONFIGS_DIR,
)
from runners.registry import ENGINES

# =============================================================================
# Uygulama
# =============================================================================

app = FastAPI(
    title="OCR Performance Analysis API",
    description="Gorsel yukle, model sec, OCR sonuclarini al.",
    version="1.0.0",
)

# React dev server'inin (localhost:5173) isteklerine izin ver
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Endpoint: /engines — Kullanilabilir motor ve model listesi
# =============================================================================

@app.get("/engines")
def get_engines():
    """
    configurations/ klasorundeki her engine icin mevcut model versiyonlarini
    tarayarak {engine: [model_v1, ...]} sozlugu dondurur.
    """
    result = {}
    for engine_name in ENGINES.keys():
        engine_config_dir = CONFIGS_DIR / engine_name
        if engine_config_dir.is_dir():
            models = sorted(
                p.stem for p in engine_config_dir.glob("*.yaml")
            )
            if models:
                result[engine_name] = models
    return result


def compute_single_file_metrics(data: dict, filename: str) -> dict:
    """
    Tek bir görsel sonucu için generate_report.py'dekine benzer
    metrik hesaplamaları yapar ve döner.
    """
    from generate_report import (
        determine_doc_type,
        extract_common_fields_data,
        aggregate_common_fields,
        is_word_valid_surucubelgesi,
        safe_float,
        avg,
        true_ratio,
    )

    doc_type = determine_doc_type(filename)
    if not doc_type:
        doc_type = "surucubelgesi"  # varsayılan

    words = data.get("words", [])
    conf_list = []
    cer_list = []
    wer_list = []
    is_match_list = []

    for word in words:
        if not isinstance(word, dict):
            continue
        # Sürücü belgesinde filtreleme var
        if doc_type == "surucubelgesi":
            if not is_word_valid_surucubelgesi(word):
                continue

        c = safe_float(word.get("confidence"))
        e = safe_float(word.get("cer"))
        w = safe_float(word.get("wer"))
        m = word.get("is_match")

        if c is not None:
            conf_list.append(c)
        if e is not None:
            cer_list.append(e)
        if w is not None:
            wer_list.append(w)
        if isinstance(m, bool):
            is_match_list.append(m)

    common_data = extract_common_fields_data(data)
    common_metrics = aggregate_common_fields([common_data])

    total_time = safe_float(data.get("total_time_seconds")) or safe_float(data.get("execution_time_seconds"))

    metrics = {
        "doc_type": doc_type,
        "file_count": 1,
        "avg_total_time_seconds": total_time,
        "avg_confidence": avg(conf_list) if conf_list else None,
        "common_fields": {
            "avg_cer": common_metrics.get("avg_cer"),
            "avg_wer": common_metrics.get("avg_wer"),
            "avg_common_field_match_ratio": common_metrics.get("avg_common_field_match_ratio"),
            "avg_common_field_confidence": common_metrics.get("avg_common_field_confidence"),
            "found_true_ratio": common_metrics.get("found_true_ratio"),
        }
    }

    if doc_type == "surucubelgesi":
        metrics["valid_word_count"] = len(conf_list)
        metrics["avg_cer"] = avg(cer_list)
        metrics["avg_wer"] = avg(wer_list)
        metrics["avg_field_match_ratio"] = safe_float(data.get("field_match_ratio"))
        metrics["is_match_true_ratio"] = true_ratio(is_match_list)

    return metrics


# =============================================================================
# Endpoint: /process — Gorsel yukle + OCR calistir
# =============================================================================

@app.post("/process")
async def process_image(
    file: UploadFile = File(..., description="Taranacak gorsel dosyasi (png/jpg)"),
    engine: str      = Form(..., description="OCR motoru: tesseract, doctr, easyocr, rapidocr"),
    model_name: str  = Form(..., description="Model versiyonu: model_v1, model_upgraded ..."),
):
    """
    Yuklenen gorseli secilen OCR motoru ile isler.

    Doner:
        words                : Kelime listesi (text, bbox, confidence, matched_field ...)
        field_results        : Ground truth alan eslesmesi (cer, wer, found ...)
        common_field_results : Ortak alan kontrolu
        execution_time_seconds, fields_found, fields_total ...
        metrics              : generate_report formatındaki metrikler
    """
    # Desteklenen uzanti kontrolu
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}:
        raise HTTPException(
            status_code=400,
            detail=f"Desteklenmeyen dosya tipi: {suffix}. PNG veya JPG gonderin.",
        )

    # Gecici dosyaya kaydet (islem bittikten sonra otomatik silinir)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = process_single_image(
            img_path=tmp_path,
            engine=engine,
            model_name=model_name,
        )
    finally:
        # Gecici dosyayi temizle
        Path(tmp_path).unlink(missing_ok=True)

    # process_single_image hata dondurmuse HTTP hataya cevir
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])

    # Metrikleri hesapla ve ekle
    try:
        result["metrics"] = compute_single_file_metrics(result, file.filename)
    except Exception as e:
        print(f"Error computing single file metrics: {e}")

    return JSONResponse(content=result)


# =============================================================================
# Endpoint: /report — comparison_report.json icerigini don
# =============================================================================

@app.get("/report")
def get_report():
    """
    generate_report.py tarafindan olusturulan comparison_report.json
    dosyasini okur ve icerigini JSON olarak dondurur.
    Dashboard ekrani bu endpoint'i kullanir.
    """
    report_path = OUTPUTS_DIR / "comparison_report.json"
    if not report_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "comparison_report.json bulunamadi. "
                "Once main.py'i calistirarak raporu olusturun."
            ),
        )
    with open(report_path, encoding="utf-8") as f:
        data = json.load(f)
    return data


# =============================================================================
# Calistirma notlari (bu blok import edildiginde calisMAZ)
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
