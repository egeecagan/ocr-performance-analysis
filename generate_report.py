"""
generate_report.py

OCR Comparison Report Generator (v2)
=====================================
Reads the OCR output JSONs under `outputs/` (grouped by model), computes
per-document-type performance metrics, and collects everything into a
single `comparison_report.json` file.

Goal: let different OCR engines/models (and different versions of the
same engine) be compared side by side on the same document set
(confidence, CER, WER, field match ratio, etc.).

Calculation rules differ by document type, because each document type's
ground-truth data is labeled differently:

  surucubelgesi (driver's license)
    - Filtering ON   : A word object only counts if AT LEAST ONE of
                        matched_field, matched_field_value or
                        matched_substring is set (i.e. not all three are
                        null). Words where all three are null never
                        matched any expected field — they're treated as
                        "noise" and skipped.
    - Metrics        : avg_confidence, avg_cer, avg_wer,
                        avg_field_match_ratio, is_match_true_ratio,
                        valid_word_count
    - common_fields  : avg_cer, avg_wer, avg_common_field_match_ratio,
                        avg_common_field_confidence, found_true_ratio

  dekont (receipt)
    - Filtering OFF  : Unlike driver's licenses, ground-truth matching
                        here isn't done per word, so ALL word objects
                        are counted unfiltered.
    - Metrics        : avg_confidence only (CER/WER and similar metrics
                        aren't meaningful for this document type, so
                        they're left out of the report)
    - common_fields  : same structure as above

  Computed in common for both types:
    - avg_total_time_seconds (average processing time per file, for
                               comparing engine speed)

=== TO ADD A NEW DOCUMENT TYPE ===
Write a process_<doc_type>(filepath) function (see process_surucubelgesi /
process_dekont for the pattern) and a compute_<doc_type>_metrics(file_list,
common_fields_dir) aggregator, then branch to it in generate_report()'s
main loop below. Document type is inferred purely from the filename
prefix before the first underscore (see determine_doc_type) — no
registration list to update elsewhere.
"""

import json
import os
from pathlib import Path
from collections import defaultdict


# ---------------------------------------------------------------------------
# General helper functions
# ---------------------------------------------------------------------------

def determine_doc_type(stem: str):
    """
    Determines the document type from a filename (without extension,
    e.g. "dekont_003").

    Naming rule: the document type name itself never contains an
    underscore (_); a single underscore separates the type name from the
    file number (e.g. "dekont_003" -> "dekont", "surucubelgesi_012" ->
    "surucubelgesi", "fatura_7" -> "fatura"). So the document type is
    whatever comes BEFORE the first underscore in the filename.

    This is fully generic: no fixed keyword list (DOC_TYPE_KEYWORDS or
    similar) is needed — fatura, banka, dekont, surucubelgesi, or any new
    document type is recognized automatically. The same naming convention
    is also used to match the "<doc_type>_c.txt" files under
    common_fields (see load_specific_keywords).
    """
    return stem.split("_", 1)[0].lower()


def safe_float(value):
    """
    Converts a value to float; returns None instead of raising for None,
    missing, or unconvertible values (e.g. empty string, wrong type).
    This lets JSON fields with missing/malformed data be silently
    dropped by avg() without breaking the average calculation.
    """
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def avg(values: list):
    """
    Returns the arithmetic mean of a list of numbers, rounded to 6
    decimal places. None values are dropped before averaging; if no
    valid (non-None) value remains, returns None (not 0 — to keep "no
    data" distinct from "value is zero").
    """
    clean = [v for v in values if v is not None]
    return round(sum(clean) / len(clean), 6) if clean else None


def true_ratio(booleans: list):
    """
    Returns the fraction (0.0-1.0) of True values in a boolean list.
    Used for "how much of this succeeded/matched" metrics like
    is_match_true_ratio or found_true_ratio. Returns None if the list
    is empty.
    """
    if not booleans:
        return None
    return round(sum(1 for b in booleans if b is True) / len(booleans), 6)


# ---------------------------------------------------------------------------
# Dynamic specific-keyword reader
# ---------------------------------------------------------------------------

def load_specific_keywords(common_fields_dir, doc_type: str) -> list:
    """
    Reads the document-type-specific keywords to search for from the
    "<doc_type>_c.txt" file under common_fields_dir (e.g. for
    doc_type='dekont', common_fields_dir/dekont_c.txt). This naming
    matches exactly the document type determine_doc_type derives from
    the filename, so adding support for a new document type just means
    adding a correctly-named .txt file — no extra code.

    File format:
        - Each line is treated as one keyword.
        - Lines starting with '#' are treated as comments and skipped.
        - Blank lines are skipped.

    These .txt files are optional: if no file exists for the given
    doc_type, this silently returns an empty list (no error/warning),
    meaning no specific-keyword hit-rate is computed for that type.
    """
    txt_path = Path(common_fields_dir) / f"{doc_type}_c.txt"
    if not txt_path.exists():
        return []  # No txt -> no keyword search for this doc type

    keywords = []
    with open(txt_path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            # Skip comments ('#...') and blank lines
            if not stripped or stripped.startswith("#"):
                continue
            keywords.append(stripped)
    return keywords


# ---------------------------------------------------------------------------
# Specific-keyword hit rate
# ---------------------------------------------------------------------------

# Maps Turkish-specific characters to their ASCII equivalents.
# OCR engines sometimes recognize Turkish characters like "İ/I", "Ş/S"
# incorrectly or inconsistently, so before comparing we normalize both the
# expected keyword and the OCR text through this table to eliminate "false
# differences" (e.g. treat "üçüncü" and "ucuncu" as the same).
_TR_NORM_TABLE = str.maketrans(
    "ŞşĞğÜüÖöÇçİıÂâÎîÛû",
    "SsGgUuOoCcIiAaIiUu"
)

def normalize_text(text: str) -> str:
    """
    Lowercases the text and converts Turkish characters to their ASCII
    equivalents (ş->s, ğ->g, ü->u, İ->i, etc.). This lets keyword
    matching stay reliable despite character-encoding / casing
    differences in OCR output.
    """
    return text.translate(_TR_NORM_TABLE).lower()


def compute_keyword_hit_rates(file_list: list, doc_type: str, common_fields_dir) -> dict:
    """
    Computes, as a percentage, how many of the model's outputs correctly
    recognized document-type-specific keywords (e.g. "EHLIYET", "BANKA
    ADI" — phrases expected to appear in every document of that type).

    How it works:
        1. Read the expected keyword list for doc_type from its .txt file
           (see load_specific_keywords).
        2. For each JSON file in file_list, build a search source by
           concatenating both the 'text' and 'raw_text' fields and
           applying Turkish character normalization.
        3. For each keyword, compute the fraction of files it appears in
           out of the total file count.

    Returns:
        { "KEYWORD": 75.0, ... }
        Meaning "KEYWORD" was found in 75% of the files.
        Returns {} if there's no .txt file for the document type, or if
        file_list is empty.
    """
    keywords = load_specific_keywords(common_fields_dir, doc_type)
    if not keywords or not file_list:
        return {}

    # Normalized keywords
    norm_keywords = {kw: normalize_text(kw) for kw in keywords}
    # Count how many files each keyword was found in
    hit_counts = {kw: 0 for kw in keywords}

    for fp in file_list:
        try:
            with open(fp, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        # Search source: 'text' + 'raw_text' concatenated
        source_text = normalize_text(
            str(data.get("text") or "") + " " + str(data.get("raw_text") or "")
        )

        for kw, norm_kw in norm_keywords.items():
            if norm_kw in source_text:
                hit_counts[kw] += 1

    total = len(file_list)
    return {
        # The normalized (ASCII, uppercase) version is used as the report
        # key, so the output JSON shows easy-to-read ASCII keys instead of
        # Turkish characters like 'Ü'.
        normalize_text(kw).upper(): round(hit_counts[kw] / total * 100, 2)
        for kw in keywords
    }


# ---------------------------------------------------------------------------
# Common field calculations
# ---------------------------------------------------------------------------

def extract_common_fields_data(data: dict) -> dict:
    """
    "common_field"s are fields shared across every document regardless of
    document type (e.g. date, document number). This function extracts
    the raw (not yet averaged) performance data for these fields from a
    single document JSON's common_field_results section — the actual
    averaging happens in aggregate_common_fields.

    Returns:
        cer_list        : CER value per common field
        wer_list        : WER value per common field
        found_list      : found bool value per common field (was the
                           field found in the document?)
        common_fmr      : document-level common_field_match_ratio (float
                           or None)
        confidence_list : confidence values of the words corresponding to
                           these common fields, via matched_word_indices
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

            # Which words correspond to this common field is tracked in
            # matched_word_indices; we collect the confidence values of
            # the words at those indices.
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
# surucubelgesi (driver's license) file processor
# ---------------------------------------------------------------------------

def is_word_valid_surucubelgesi(word: dict) -> bool:
    """
    Driver's license filtering rule.

    For a word to count, at least one of matched_field,
    matched_field_value or matched_substring must be set. If all three
    are null, this word never matched any expected field in the
    ground truth (e.g. unrelated/noise text on the page) and is excluded
    from the metrics.
    """
    return not (
        word.get("matched_field")       is None and
        word.get("matched_field_value") is None and
        word.get("matched_substring")   is None
    )


def process_surucubelgesi(filepath: Path) -> dict:
    """
    Reads a single driver's license JSON and extracts raw (not yet
    averaged) performance data. Words filtered out by
    is_word_valid_surucubelgesi are never added to these lists.

    If the file can't be read/is malformed (JSON error, missing file,
    etc.), a warning is printed and an "empty" result (all lists empty)
    is returned — the script doesn't crash over one file, that file
    just contributes nothing to the report.

    Returns keys:
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
        print(f"  [WARNING] Could not read: {filepath.name} -> {exc}")
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
            continue  # Filter: all three null -> skip

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
# dekont (receipt) file processor
# ---------------------------------------------------------------------------

def process_dekont(filepath: Path) -> dict:
    """
    Reads a single receipt JSON and extracts raw performance data.

    Unlike surucubelgesi, there is NO per-word matching filter here — the
    confidence of every word on the page is counted. If the file can't be
    read, a warning is printed and an empty "empty" result is returned.

    Returns keys:
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
        print(f"  [WARNING] Could not read: {filepath.name} -> {exc}")
        return empty

    words     = data.get("words", [])
    conf_list = []

    for word in words:
        if not isinstance(word, dict):
            continue
        # No filtering: every word counts
        c = safe_float(word.get("confidence"))
        if c is not None:
            conf_list.append(c)

    return {
        "conf_list"  : conf_list,
        "total_time" : safe_float(data.get("total_time_seconds")),
        "common"     : extract_common_fields_data(data),
    }


# ---------------------------------------------------------------------------
# Raw data aggregator & metric calculator (by document type)
# ---------------------------------------------------------------------------

def aggregate_common_fields(common_parts: list) -> dict:
    """
    Merges the raw common-field lists (collected via
    extract_common_fields_data) from every file in a model/document-type
    group into a single pool, and produces the final 'common_fields'
    metrics object from that pool.

    No per-file averaging happens — all values are pooled first, and the
    average is computed once at the end (a micro-average). This means
    files with more common fields naturally carry more weight in the
    average.
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


def compute_surucubelgesi_metrics(file_list: list, common_fields_dir) -> dict:
    """
    Processes every driver's license file belonging to the same
    model/version (file_list) one by one via process_surucubelgesi,
    accumulates the raw data, and produces this model's final summary
    metrics (the report block). Cross-model comparison is done via the
    object this function returns.
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
        "specific_keyword_success_rates": compute_keyword_hit_rates(file_list, "surucubelgesi", common_fields_dir),
    }


def compute_dekont_metrics(file_list: list, common_fields_dir) -> dict:
    """
    Processes every receipt file belonging to the same model/version
    (file_list) one by one via process_dekont and produces the final
    summary metrics.

    Unlike surucubelgesi, metrics like CER/WER/field_match_ratio aren't
    tracked meaningfully in the ground truth for receipts, so they're left
    out of the report; only avg_confidence and common_fields are included.
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
        "specific_keyword_success_rates": compute_keyword_hit_rates(file_list, "dekont", common_fields_dir),
    }


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def generate_report(
    outputs_dir: str = "outputs",
    common_fields_dir: str = None,
    models_to_process: list = None,
) -> None:
    """
    Walks the outputs/ directory, computes performance metrics for every
    (model, version, document type) combination, and writes them all into
    a single comparison_report.json file.

    Parameters:
        outputs_dir        : Root folder containing the models' JSON
                              outputs.
        common_fields_dir  : Folder containing the .txt files with
                              document-type-specific keyword lists (see
                              load_specific_keywords). If None, no
                              keyword search is performed.
        models_to_process  : [(engine, model_name), ...] — restricts
                              processing to ONLY these models (e.g. to
                              re-run a single model and refresh the
                              report). If None, every model under
                              outputs/ is included.

    Expected folder structure:
        outputs/
            <model_name>/
                <version>/
                    dekont.json, dekont2.json, surucubelgesi1.json, ...

    Note: every time this function runs, any existing comparison_report.json
    is deleted first, so the report is always generated fresh and
    consistently (preventing stale/leftover data from leaking into it).
    """
    base = Path(outputs_dir)
    if not base.is_dir():
        raise FileNotFoundError(
            f"Directory not found: {base.resolve()}\n"
            "Run the script from the parent directory of the 'outputs/' folder."
        )

    # Delete the old report (start fresh every run)
    out_path = base / "comparison_report.json"
    if out_path.exists():
        out_path.unlink()
        print("[REPORT] Old report deleted, generating from scratch...")

    # Convert models_to_process into a set for fast membership checks:
    # {(engine, version), ...}
    allowed_models = None
    if models_to_process is not None:
        allowed_models = {(e, m) for e, m in models_to_process}

    # Intermediate structure grouping files by document type and model
    # while scanning: index[doc_type][model_label] = [Path, Path, ...]
    index: dict = defaultdict(lambda: defaultdict(list))

    for model_dir in sorted(base.iterdir()):
        if not model_dir.is_dir():
            continue
        for version_dir in sorted(model_dir.iterdir()):
            if not version_dir.is_dir():
                continue

            # If models_to_process is given, skip model/version
            # combinations not in the list
            if allowed_models is not None:
                if (model_dir.name, version_dir.name) not in allowed_models:
                    continue

            model_label = f"{model_dir.name}/{version_dir.name}"
            for jf in sorted(version_dir.glob("*.json")):
                doc_type = determine_doc_type(jf.stem)
                index[doc_type][model_label].append(jf)

    if not index:
        print("No valid JSON found.")
        return

    report: dict = {}

    for doc_type in sorted(index.keys()):
        report[doc_type] = {}
        print(f"\n[{doc_type.upper()}]")

        for model_label in sorted(index[doc_type].keys()):
            file_list = index[doc_type][model_label]
            print(f"  {model_label}  ({len(file_list)} files)")

            if doc_type == "surucubelgesi":
                metrics = compute_surucubelgesi_metrics(file_list, common_fields_dir)
            elif doc_type == "dekont":
                metrics = compute_dekont_metrics(file_list, common_fields_dir)
            else:
                metrics = {"file_count": len(file_list), "note": "no calculation rule defined"}

            report[doc_type][model_label] = metrics

            cf = metrics.get("common_fields", {})
            print(
                f"    avg_conf={metrics.get('avg_confidence')}  "
                f"avg_time={metrics.get('avg_total_time_seconds')}s  "
                f"cf_found%={cf.get('found_true_ratio')}"
            )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=True, indent=4)

    print(f"\n Report saved: {out_path.resolve()}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Paths are derived from the script's own location, not the directory
    # it's invoked from, so it finds the right outputs/ and common_fields/
    # folders even when called from elsewhere.
    script_dir        = os.path.dirname(os.path.abspath(__file__))
    outputs_dir       = os.path.join(script_dir, "outputs")
    common_fields_dir = os.path.join(script_dir, "inputs", "truths", "common_fields")
    generate_report(
        outputs_dir=outputs_dir,
        common_fields_dir=common_fields_dir,
        models_to_process=None,   # None = every model under outputs/
    )
