"""
OCR Karşılaştırmalı Rapor Üretici  (v2)
=========================================
Belge türüne göre farklı hesaplama kuralları uygular:

  surucubelgesi
    - Filtreleme AÇIK  : matched_field + matched_field_value + matched_substring
                         üçü birden null olan word objeleri hesaplamalara dahil edilmez.
    - Metrikler        : avg_confidence, avg_cer, avg_wer,
                         avg_field_match_ratio, is_match_true_ratio, valid_word_count
    - common_fields    : avg_cer, avg_wer, avg_common_field_match_ratio,
                         avg_common_field_confidence, found_true_ratio

  dekont
    - Filtreleme KAPALI: Tüm word objeleri hesaba katılır.
    - Metrikler        : Sadece avg_confidence  (diğerleri rapora yazılmaz)
    - common_fields    : Yukarıdakiyle aynı yapı

  Her iki tür için de:
    - avg_total_time_seconds  (dosya başına ortalama işlem süresi)
"""

import json
import os
from pathlib import Path
from collections import defaultdict


# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------

DOC_TYPE_KEYWORDS = {
    "surucubelgesi": "surucubelgesi",
    "dekont": "dekont",
}

# Her belge türü için aranacak özel kelimeler (ASCII karşılıkları kullanılır)
# Arama normalize edilmiş metin üzerinde yapıldığından Türkçe karakterler sorun çıkarmaz.
SPECIFIC_KEYWORDS = {
    "surucubelgesi": ["SURUCU BELGESI", "DRIVING LICENCE", "TURKIYE CUMHURIYETI"],
    "dekont":        ["VAKIFLAR BANKASI", "DEKONT", "SAYIN", "BUYUK MUKELLEFLER"],
}


# ---------------------------------------------------------------------------
# Genel yardımcı fonksiyonlar
# ---------------------------------------------------------------------------

def determine_doc_type(stem: str):
    """Dosya adından belge türünü saptar. Bilinmiyorsa None döner."""
    lower = stem.lower()
    for keyword, doc_type in DOC_TYPE_KEYWORDS.items():
        if keyword in lower:
            return doc_type
    return None


def safe_float(value):
    """None-safe float dönüşümü; dönüştürülemezse None döner."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def avg(values: list):
    """Sayı listesinin aritmetik ortalaması; boşsa None."""
    clean = [v for v in values if v is not None]
    return round(sum(clean) / len(clean), 6) if clean else None


def true_ratio(booleans: list):
    """Boolean listesinde True oranı (0.0–1.0); boşsa None."""
    if not booleans:
        return None
    return round(sum(1 for b in booleans if b is True) / len(booleans), 6)


# ---------------------------------------------------------------------------
# Özel kelime bulma başarısı (Hit Rate)
# ---------------------------------------------------------------------------

_TR_NORM_TABLE = str.maketrans(
    "ŞşĞğÜüÖöÇçİıÂâÎîÛû",
    "SsGgUuOoCcIiAaIiUu"
)

def normalize_text(text: str) -> str:
    """
    Metni küçük harfe çevirir ve Türkçe karakterleri
    karşılıklarına dönüştürür (ş→s, ğ→g, ü→u vb.).
    Bu sayede OCR çıktısındaki karakter farklılıklarına rağmen
    eşleşme yapılabilir.
    """
    return text.translate(_TR_NORM_TABLE).lower()


def compute_keyword_hit_rates(file_list: list, doc_type: str) -> dict:
    """
    Belge türüne ait özel kelimelerin dosyalar içindeki
    bulunma oranını hesaplar (0.0–100.0 yüzde olarak).

    Arama, hem 'text' hem 'raw_text' alanında yapılır;
    Türkçe karakter normalizasyonu uygulanır.

    Döner:
        { "KELIME": 75.0, ... }  (yüzde 0–100 arası float)
    """
    keywords = SPECIFIC_KEYWORDS.get(doc_type, [])
    if not keywords or not file_list:
        return {}

    # Normalize edilmiş anahtar kelimeler
    norm_keywords = {kw: normalize_text(kw) for kw in keywords}
    # Her kelime için kaç dosyada bulunduğunu say
    hit_counts = {kw: 0 for kw in keywords}

    for fp in file_list:
        try:
            with open(fp, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        # Arama kaynağı: 'text' + 'raw_text' birleştirilerek kullanılır
        source_text = normalize_text(
            str(data.get("text") or "") + " " + str(data.get("raw_text") or "")
        )

        for kw, norm_kw in norm_keywords.items():
            if norm_kw in source_text:
                hit_counts[kw] += 1

    total = len(file_list)
    return {
        kw: round(hit_counts[kw] / total * 100, 2)
        for kw in keywords
    }


# ---------------------------------------------------------------------------
# Ortak alan (common_field) hesaplamaları
# ---------------------------------------------------------------------------

def extract_common_fields_data(data: dict) -> dict:
    """
    Bir belge JSON'undan common_field_results bölümündeki
    ham verileri çıkarır.

    Dönen:
        cer_list              : common field başına CER değerleri
        wer_list              : common field başına WER değerleri
        found_list            : common field başına found bool değerleri
        common_fmr            : belge düzeyi common_field_match_ratio (ya float ya None)
        confidence_list       : common field'lara denk gelen word confidence'ları
    """
    words = data.get("words", [])
    cfr   = data.get("common_field_results", {})

    cer_list        = []
    wer_list        = []
    found_list      = []
    confidence_list = []

    if isinstance(cfr, dict):
        for field_data in cfr.values():
            if not isinstance(field_data, dict):
                continue

            c = safe_float(field_data.get("cer"))
            w = safe_float(field_data.get("wer"))
            f = field_data.get("found")

            if c is not None:
                cer_list.append(c)
            if w is not None:
                wer_list.append(w)
            if isinstance(f, bool):
                found_list.append(f)

            # Confidence: matched_word_indices üzerinden word listesinden çek
            indices = field_data.get("matched_word_indices")
            if isinstance(indices, list) and words:
                for idx in indices:
                    if isinstance(idx, int) and 0 <= idx < len(words):
                        conf = safe_float(words[idx].get("confidence"))
                        if conf is not None:
                            confidence_list.append(conf)

    common_fmr = safe_float(data.get("common_field_match_ratio"))

    return {
        "cer_list"        : cer_list,
        "wer_list"        : wer_list,
        "found_list"      : found_list,
        "common_fmr"      : common_fmr,
        "confidence_list" : confidence_list,
    }


# ---------------------------------------------------------------------------
# surucubelgesi dosyası işleyici
# ---------------------------------------------------------------------------

def is_word_valid_surucubelgesi(word: dict) -> bool:
    """
    Sürücü belgesi filtreleme kuralı:
    matched_field, matched_field_value, matched_substring
    üçü birden null ise geçersiz → False döner.
    """
    return not (
        word.get("matched_field")       is None and
        word.get("matched_field_value") is None and
        word.get("matched_substring")   is None
    )


def process_surucubelgesi(filepath: Path) -> dict:
    """
    Sürücü belgesi JSON'undan ham verileri çıkarır.
    Dönen anahtarlar:
        conf_list, cer_list, wer_list, is_match_list,
        field_match_ratio,
        total_time,
        common: { cer_list, wer_list, found_list, common_fmr, confidence_list }
    """
    empty = {
        "conf_list": [], "cer_list": [], "wer_list": [],
        "is_match_list": [], "field_match_ratio": None,
        "total_time": None,
        "common": {
            "cer_list": [], "wer_list": [], "found_list": [],
            "common_fmr": None, "confidence_list": [],
        },
    }

    try:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  [UYARI] Okunamadi: {filepath.name} -> {exc}")
        return empty

    words = data.get("words", [])
    conf_list     = []
    cer_list      = []
    wer_list      = []
    is_match_list = []

    for word in words:
        if not isinstance(word, dict):
            continue
        if not is_word_valid_surucubelgesi(word):
            continue  # Filtre: üçü birden null → atla

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

    return {
        "conf_list"         : conf_list,
        "cer_list"          : cer_list,
        "wer_list"          : wer_list,
        "is_match_list"     : is_match_list,
        "field_match_ratio" : safe_float(data.get("field_match_ratio")),
        "total_time"        : safe_float(data.get("total_time_seconds")),
        "common"            : extract_common_fields_data(data),
    }


# ---------------------------------------------------------------------------
# dekont dosyası işleyici
# ---------------------------------------------------------------------------

def process_dekont(filepath: Path) -> dict:
    """
    Dekont JSON'undan ham verileri çıkarır.
    Filtreleme UYGULANMAZ – tüm word'ler confidence hesabına dahil edilir.
    Dönen anahtarlar:
        conf_list, total_time,
        common: { cer_list, wer_list, found_list, common_fmr, confidence_list }
    """
    empty = {
        "conf_list"  : [],
        "total_time" : None,
        "common"     : {
            "cer_list": [], "wer_list": [], "found_list": [],
            "common_fmr": None, "confidence_list": [],
        },
    }

    try:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  [UYARI] Okunamadi: {filepath.name} -> {exc}")
        return empty

    words     = data.get("words", [])
    conf_list = []

    for word in words:
        if not isinstance(word, dict):
            continue
        # Filtreleme YOK: tüm word'ler hesaba katılır
        c = safe_float(word.get("confidence"))
        if c is not None:
            conf_list.append(c)

    return {
        "conf_list"  : conf_list,
        "total_time" : safe_float(data.get("total_time_seconds")),
        "common"     : extract_common_fields_data(data),
    }


# ---------------------------------------------------------------------------
# Ham veri birleştirici & metrik hesaplayıcı (belge türüne göre)
# ---------------------------------------------------------------------------

def aggregate_common_fields(common_parts: list) -> dict:
    """
    Birden fazla dosyadan gelen common field ham verilerini birleştirir
    ve 'common_fields' metrik objesini döner.
    """
    all_cer   = []
    all_wer   = []
    all_found = []
    all_fmr   = []
    all_conf  = []

    for cp in common_parts:
        all_cer.extend(cp.get("cer_list", []))
        all_wer.extend(cp.get("wer_list", []))
        all_found.extend(cp.get("found_list", []))
        all_conf.extend(cp.get("confidence_list", []))
        fmr = cp.get("common_fmr")
        if fmr is not None:
            all_fmr.append(fmr)

    return {
        "avg_cer"                    : avg(all_cer),
        "avg_wer"                    : avg(all_wer),
        "avg_common_field_match_ratio": avg(all_fmr),
        "avg_common_field_confidence": avg(all_conf),
        "found_true_ratio"           : true_ratio(all_found),
    }


def compute_surucubelgesi_metrics(file_list: list) -> dict:
    """
    Sürücü belgesi dosyaları için metrikleri hesaplar.
    """
    all_conf      = []
    all_cer       = []
    all_wer       = []
    all_is_match  = []
    all_fmr       = []
    all_time      = []
    common_parts  = []

    for fp in file_list:
        d = process_surucubelgesi(fp)
        all_conf.extend(d["conf_list"])
        all_cer.extend(d["cer_list"])
        all_wer.extend(d["wer_list"])
        all_is_match.extend(d["is_match_list"])
        common_parts.append(d["common"])
        if d["field_match_ratio"] is not None:
            all_fmr.append(d["field_match_ratio"])
        if d["total_time"] is not None:
            all_time.append(d["total_time"])

    return {
        "file_count"                   : len(file_list),
        "avg_total_time_seconds"       : avg(all_time),
        "valid_word_count"             : len(all_conf),
        "avg_confidence"               : avg(all_conf),
        "avg_cer"                      : avg(all_cer),
        "avg_wer"                      : avg(all_wer),
        "avg_field_match_ratio"        : avg(all_fmr),
        "is_match_true_ratio"          : true_ratio(all_is_match),
        "common_fields"                : aggregate_common_fields(common_parts),
        "specific_keyword_success_rates": compute_keyword_hit_rates(file_list, "surucubelgesi"),
    }


def compute_dekont_metrics(file_list: list) -> dict:
    """
    Dekont dosyaları için metrikleri hesaplar.
    Rapora yalnızca avg_confidence + common_fields eklenir.
    """
    all_conf     = []
    all_time     = []
    common_parts = []

    for fp in file_list:
        d = process_dekont(fp)
        all_conf.extend(d["conf_list"])
        common_parts.append(d["common"])
        if d["total_time"] is not None:
            all_time.append(d["total_time"])

    return {
        "file_count"                   : len(file_list),
        "avg_total_time_seconds"       : avg(all_time),
        "avg_confidence"               : avg(all_conf),
        "common_fields"                : aggregate_common_fields(common_parts),
        "specific_keyword_success_rates": compute_keyword_hit_rates(file_list, "dekont"),
    }


# ---------------------------------------------------------------------------
# Ana fonksiyon
# ---------------------------------------------------------------------------

def generate_report(outputs_dir: str = "outputs") -> None:
    """
    outputs/ dizinini tarar ve comparison_report.json üretir.

    Beklenen yapı:
        outputs/
            <model_adi>/
                <versiyon>/
                    dekont.json, dekont2.json, surucubelgesi1.json, ...
    """
    base = Path(outputs_dir)
    if not base.is_dir():
        raise FileNotFoundError(
            f"Dizin bulunamadi: {base.resolve()}\n"
            "Scripti 'outputs/' klasörünün bir üst dizininden çalıştırın."
        )

    # index[doc_type][model_label] = [Path, ...]
    index: dict = defaultdict(lambda: defaultdict(list))

    for model_dir in sorted(base.iterdir()):
        if not model_dir.is_dir():
            continue
        for version_dir in sorted(model_dir.iterdir()):
            if not version_dir.is_dir():
                continue
            model_label = f"{model_dir.name}/{version_dir.name}"
            for jf in sorted(version_dir.glob("*.json")):
                doc_type = determine_doc_type(jf.stem)
                if doc_type is None:
                    continue
                index[doc_type][model_label].append(jf)

    if not index:
        print("Hiç geçerli JSON bulunamadi.")
        return

    report: dict = {}

    for doc_type in sorted(index.keys()):
        report[doc_type] = {}
        print(f"\n[{doc_type.upper()}]")

        for model_label in sorted(index[doc_type].keys()):
            file_list = index[doc_type][model_label]
            print(f"  {model_label}  ({len(file_list)} dosya)")

            if doc_type == "surucubelgesi":
                metrics = compute_surucubelgesi_metrics(file_list)
            elif doc_type == "dekont":
                metrics = compute_dekont_metrics(file_list)
            else:
                # Gelecekte eklenebilecek türler için genel fallback
                metrics = {"file_count": len(file_list), "note": "hesaplama kurali tanimli degil"}

            report[doc_type][model_label] = metrics

            # Konsol özeti
            cf = metrics.get("common_fields", {})
            print(
                f"    avg_conf={metrics.get('avg_confidence')}  "
                f"avg_time={metrics.get('avg_total_time_seconds')}s  "
                f"cf_found%={cf.get('found_true_ratio')}"
            )

    out_path = base / "comparison_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=True, indent=4)

    print(f"\n Rapor kaydedildi: {out_path.resolve()}")


# ---------------------------------------------------------------------------
# Giriş noktası
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    outputs_dir = os.path.join(script_dir, "outputs")
    generate_report(outputs_dir=outputs_dir)
