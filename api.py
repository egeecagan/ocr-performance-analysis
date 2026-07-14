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

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# main.py'den OCR fonksiyonlarini ve yol sabitlerini aktar
from main import (
    process_single_image,
    BASE_DIR,
    OUTPUTS_DIR,
    CONFIGS_DIR,
)
from runners.registry import ENGINES

WEB_OUTPUTS_DIR = BASE_DIR / "web_outputs"
WEB_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

pipeline_status = {"status": "idle", "progress": "", "error": None}

def run_full_pipeline_task():
    global pipeline_status
    pipeline_status["status"] = "running"
    pipeline_status["progress"] = "Temizlik yapılıyor..."
    pipeline_status["error"] = None
    try:
        import main
        import shutil
        from generate_report import generate_report
        
        # 1. Clean outputs dir
        if main.OUTPUTS_DIR.exists():
            shutil.rmtree(main.OUTPUTS_DIR)
        main.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        
        # 2. Get all models to run
        configs = []
        for engine_name in main.ENGINES.keys():
            engine_config_dir = main.CONFIGS_DIR / engine_name
            if engine_config_dir.is_dir():
                for p in engine_config_dir.glob("*.yaml"):
                    configs.append((engine_name, p.stem))
                    
        # 3. Run process_pipeline for each model
        processed_models = []
        for i, (eng, mod) in enumerate(configs):
            pipeline_status["progress"] = f"Model çalıştırılıyor ({i+1}/{len(configs)}): {eng}/{mod}..."
            main.process_pipeline(eng, mod)
            processed_models.append((eng, mod))
            
        # 4. Generate the report
        pipeline_status["progress"] = "Karşılaştırma raporu oluşturuluyor..."
        generate_report(
            outputs_dir=str(main.OUTPUTS_DIR),
            common_fields_dir=str(main.COMMON_FIELDS_DIR),
            models_to_process=processed_models,
        )
        
        pipeline_status["status"] = "success"
        pipeline_status["progress"] = "Rapor başarıyla üretildi."
    except Exception as e:
        pipeline_status["status"] = "error"
        pipeline_status["progress"] = ""
        pipeline_status["error"] = str(e)

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
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve web_outputs directory statically
app.mount("/web_outputs", StaticFiles(directory=str(WEB_OUTPUTS_DIR)), name="web_outputs")

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

    doc_type = determine_doc_type(Path(filename).stem)
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
    Resmi ve ciktilari web_outputs klasorune kaydeder.
    """
    # Desteklenen uzanti kontrolu
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}:
        raise HTTPException(
            status_code=400,
            detail=f"Desteklenmeyen dosya tipi: {suffix}. PNG veya JPG gonderin.",
        )

    import time
    timestamp = int(time.time())
    safe_filename = Path(file.filename).name.replace(" ", "_")
    
    # Resmi web_outputs/<engine>/<model_name>/ altina kaydet
    saved_dir = WEB_OUTPUTS_DIR / engine / model_name
    saved_dir.mkdir(parents=True, exist_ok=True)
    
    saved_img_name = f"uploaded_{timestamp}_{safe_filename}"
    saved_img_path = saved_dir / saved_img_name

    with open(saved_img_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        result = process_single_image(
            img_path=str(saved_img_path),
            engine=engine,
            model_name=model_name,
            original_name=file.filename,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR islemi sirasinda hata: {e}")

    # process_single_image hata dondurmuse HTTP hataya cevir
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])

    # Metrikleri hesapla ve ekle
    try:
        result["metrics"] = compute_single_file_metrics(result, file.filename)
    except Exception as e:
        print(f"Error computing single file metrics: {e}")

    result["filename"] = file.filename

    # Sonucu JSON olarak web_outputs/<engine>/<model_name>/ klasorune kaydet
    saved_json_name = f"uploaded_{timestamp}_{Path(safe_filename).stem}.json"
    saved_json_path = saved_dir / saved_json_name
    with open(saved_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    return JSONResponse(content=result)


# =============================================================================
# Endpoint: /run-pipeline — Toplu analiz pipeline'ini tetikle
# =============================================================================

@app.post("/run-pipeline")
def run_pipeline(background_tasks: BackgroundTasks):
    global pipeline_status
    if pipeline_status["status"] == "running":
        raise HTTPException(status_code=400, detail="Toplu analiz zaten çalışıyor.")
    background_tasks.add_task(run_full_pipeline_task)
    return {"status": "started", "message": "Toplu analiz arka planda başlatıldı."}


# =============================================================================
# Endpoint: /pipeline-status — Toplu analiz durumunu sorgula
# =============================================================================

@app.get("/pipeline-status")
def get_pipeline_status():
    global pipeline_status
    return pipeline_status


# =============================================================================
# Endpoint: /run-custom-pipeline — Seçili görsel + model kombinasyonlarını çalıştır
# =============================================================================

custom_pipeline_status = {"status": "idle", "progress": "", "error": None, "result": None}


def run_custom_pipeline_task(image_paths: list, configs: list, run_id: str):
    """
    Kullanicinin sectigi gorseller ve model/config kombinasyonlariyla
    pipeline'i calistirir. Her (engine, model_name) x gorsel kombinasyonu
    ayri ayri islenir; sonuclar toplanip ozet rapor uretilir.
    """
    global custom_pipeline_status
    custom_pipeline_status["status"] = "running"
    custom_pipeline_status["progress"] = "Başlatılıyor..."
    custom_pipeline_status["error"] = None
    custom_pipeline_status["result"] = None

    try:
        import main
        import json as _json
        from generate_report import generate_report

        # Ciktilari custom run'a ozel alt klasore yaz
        output_subdir = f"custom_{run_id}"
        custom_out_dir = main.OUTPUTS_DIR / output_subdir
        custom_out_dir.mkdir(parents=True, exist_ok=True)

        total = len(configs)
        processed_models = []

        for i, cfg in enumerate(configs):
            eng = cfg["engine"]
            mod = cfg["model_name"]
            custom_pipeline_status["progress"] = (
                f"Model çalıştırılıyor ({i+1}/{total}): {eng}/{mod}..."
            )
            main.process_pipeline(
                engine=eng,
                model_name=mod,
                image_files=image_paths,
                output_subdir=output_subdir,
            )
            processed_models.append((eng, mod))

        # Rapor uret — generate_report'un report_path parametresini destekleyip
        # desteklemedigine gore cagri yap
        import inspect as _inspect
        custom_pipeline_status["progress"] = "Karşılaştırma raporu oluşturuluyor..."
        report_path = custom_out_dir / "comparison_report.json"

        gr_sig = _inspect.signature(generate_report)
        if "report_path" in gr_sig.parameters:
            generate_report(
                outputs_dir=str(custom_out_dir),
                common_fields_dir=str(main.COMMON_FIELDS_DIR),
                models_to_process=processed_models,
                report_path=str(report_path),
            )
        else:
            # Eski imza: report_path desteklenmiyor, OUTPUTS_DIR'a yazar
            # Sonucu oradan kopyala
            generate_report(
                outputs_dir=str(custom_out_dir),
                common_fields_dir=str(main.COMMON_FIELDS_DIR),
                models_to_process=processed_models,
            )

        # Raporu oku ve result olarak sakla
        if report_path.exists():
            with open(report_path, encoding="utf-8") as f:
                custom_pipeline_status["result"] = _json.load(f)

        custom_pipeline_status["status"] = "success"
        custom_pipeline_status["progress"] = "Tamamlandı."

    except Exception as e:
        custom_pipeline_status["status"] = "error"
        custom_pipeline_status["progress"] = ""
        custom_pipeline_status["error"] = str(e)


@app.post("/run-custom-pipeline")
async def run_custom_pipeline(
    background_tasks: BackgroundTasks,
    images: list[UploadFile] = File(..., description="Karşılaştırılacak görseller"),
    configs: str = Form(..., description='JSON: [{"engine":"tesseract","model_name":"model_v1"}, ...]'),
):
    """
    Kullanicinin sectigi gorseller + model/config kombinasyonlariyla
    ozel karsilastirma pipeline'ini arka planda baslatir.
    configs parametresi JSON string olarak gonderilir.
    """
    global custom_pipeline_status
    if custom_pipeline_status["status"] == "running":
        raise HTTPException(status_code=400, detail="Özel karşılaştırma zaten çalışıyor.")

    import json as _json
    import time as _time
    try:
        config_list = _json.loads(configs)
    except Exception:
        raise HTTPException(status_code=400, detail="configs geçerli bir JSON listesi değil.")

    if not config_list:
        raise HTTPException(status_code=400, detail="En az bir model/config seçmelisiniz.")
    if not images:
        raise HTTPException(status_code=400, detail="En az bir görsel yüklemelisiniz.")

    # Gorselleri gecici klasore kaydet
    run_id = str(int(_time.time()))
    upload_dir = WEB_OUTPUTS_DIR / f"custom_{run_id}" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for img in images:
        suffix = Path(img.filename).suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}:
            raise HTTPException(
                status_code=400,
                detail=f"Desteklenmeyen dosya tipi: {suffix}. PNG veya JPG gönderin.",
            )
        safe_name = Path(img.filename).name.replace(" ", "_")
        dest = upload_dir / safe_name
        with open(dest, "wb") as f:
            shutil.copyfileobj(img.file, f)
        saved_paths.append(str(dest))

    background_tasks.add_task(
        run_custom_pipeline_task, saved_paths, config_list, run_id
    )
    return {"status": "started", "run_id": run_id, "message": "Özel karşılaştırma başlatıldı."}


# =============================================================================
# Endpoint: /custom-pipeline-status — Özel pipeline durumunu sorgula
# =============================================================================

@app.get("/custom-pipeline-status")
def get_custom_pipeline_status():
    global custom_pipeline_status
    return custom_pipeline_status


# =============================================================================
# Endpoint: /clear-web-outputs — web_outputs klasorunu temizle
# =============================================================================

@app.post("/clear-web-outputs")
def clear_web_outputs():
    """
    web_outputs klasoru altindaki tum yuklemeleri ve sonuclari siler.
    """
    try:
        if WEB_OUTPUTS_DIR.exists():
            shutil.rmtree(WEB_OUTPUTS_DIR)
        WEB_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        return {"status": "success", "message": "web_outputs klasoru temizlendi."}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"web_outputs klasoru temizlenirken hata olustu: {e}",
        )


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
# Endpoint: /past-reports — Gecmis raporlari listele
# =============================================================================

@app.get("/past-reports")
def get_past_reports():
    reports = []
    if not WEB_OUTPUTS_DIR.exists():
        return []
        
    for json_path in sorted(WEB_OUTPUTS_DIR.glob("**/*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            
            rel_path = json_path.relative_to(WEB_OUTPUTS_DIR)
            if len(rel_path.parts) < 3:
                continue
                
            engine = rel_path.parts[0]
            model_name = rel_path.parts[1]
            
            # Ilgili gorsel dosyasini bulmaya calis
            parent_dir = json_path.parent
            img_file = None
            stem = json_path.stem
            img_extensions = [".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"]
            
            # 1. Tam eslesen dosya adini ara
            for ext in img_extensions:
                candidate = parent_dir / f"{stem}{ext}"
                if candidate.exists():
                    img_file = candidate.name
                    break
                    
            # 2. Eslesmediyse timestamp'e gore ara (uploaded_123456_...)
            if not img_file:
                parts = stem.split("_")
                if len(parts) >= 2 and parts[0] == "uploaded":
                    prefix = f"uploaded_{parts[1]}_"
                    for f_item in parent_dir.iterdir():
                        if f_item.is_file() and f_item.name.startswith(prefix) and f_item.suffix.lower() in img_extensions:
                            img_file = f_item.name
                            break
                            
            # 3. Hala yoksa dosya adina gore genel ara (legacy)
            if not img_file:
                for f_item in parent_dir.iterdir():
                    if f_item.is_file() and stem in f_item.name and f_item.suffix.lower() in img_extensions:
                        img_file = f_item.name
                        break
            
            metrics = data.get("metrics", {})
            filename = data.get("filename") or (img_file[20:] if img_file and img_file.startswith("uploaded_") else stem)
            
            reports.append({
                "id": stem,
                "engine": engine,
                "model_name": model_name,
                "filename": filename,
                "timestamp": json_path.stat().st_mtime,
                "image_url": f"/web_outputs/{engine}/{model_name}/{img_file}" if img_file else None,
                "metrics": metrics,
                "data": data
            })
        except Exception as e:
            print(f"Past report parsing error ({json_path}): {e}")
            continue
            
    return reports


# =============================================================================
# Endpoint: /past-reports/{engine}/{model_name}/{report_id} — Tekil rapor sil
# =============================================================================

@app.delete("/past-reports/{engine}/{model_name}/{report_id}")
def delete_past_report(engine: str, model_name: str, report_id: str):
    target_dir = WEB_OUTPUTS_DIR / engine / model_name
    if not target_dir.exists():
        raise HTTPException(status_code=404, detail="Gecmis rapor dizini bulunamadi.")
        
    json_file = target_dir / f"{report_id}.json"
    if not json_file.exists():
        raise HTTPException(status_code=404, detail="Rapor bulunamadi.")
        
    # JSON dosyasini sil
    try:
        json_file.unlink()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rapor JSON'i silinirken hata: {e}")
        
    # Gorsel dosyasini sil
    deleted_img = False
    img_extensions = [".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp"]
    for ext in img_extensions:
        img_file = target_dir / f"{report_id}{ext}"
        if img_file.exists():
            try:
                img_file.unlink()
                deleted_img = True
            except Exception as e:
                print(f"Gorsel silinirken hata ({img_file}): {e}")
                
    return {"status": "success", "message": "Rapor basariyla silindi.", "deleted_image": deleted_img}


# =============================================================================
# Calistirma notlari (bu blok import edildiginde calisMAZ)
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
