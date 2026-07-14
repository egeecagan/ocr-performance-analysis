"""
OCR Karşılaştırmalı Rapor Üretici  (v2)
=========================================
Bu script, `outputs/` klasörü altında modellere göre gruplanmış OCR
çıktı JSON'larını okur, belge türüne göre performans metriklerini
hesaplar ve tek bir `comparison_report.json` dosyasında toplar.

Amaç: farklı OCR modellerinin ve farklı versiyonlarının aynı belge seti
üzerindeki başarısını (confidence, CER, WER, alan eşleşme oranı vb.)
yan yana karşılaştırabilmek.

Belge türüne göre hesaplama kuralları birbirinden farklıdır, çünkü
her belge türünün ground-truth (doğruluk) verisi farklı şekilde
etiketlenmiştir:

  surucubelgesi  (sürücü belgesi)
    - Filtreleme AÇIK  : Bir word (tekil kelime) objesi ancak matched_field,
                         matched_field_value veya matched_substring
                         alanlarından EN AZ BİRİ doluysa (üçü birden
                         null DEĞİLSE) hesaba katılır. Bu üçü birden
                         null olan word'ler, beklenen alanlarla hiç
                         eşleşmemiş "gürültü" kabul edilir ve atlanır.
    - Metrikler        : avg_confidence, avg_cer, avg_wer,
                         avg_field_match_ratio, is_match_true_ratio,
                         valid_word_count
    - common_fields    : avg_cer, avg_wer, avg_common_field_match_ratio,
                         avg_common_field_confidence, found_true_ratio

  dekont
    - Filtreleme KAPALI: Sürücü belgesinden farklı olarak burada
                         ground-truth eşleştirmesi word bazlı
                         yapılmadığından, TÜM word objeleri filtresiz
                         şekilde hesaba katılır.
    - Metrikler        : Sadece avg_confidence  (CER/WER gibi diğer
                         metrikler bu belge türü için anlamlı
                         olmadığından rapora yazılmaz)
    - common_fields    : Yukarıdakiyle aynı yapı

  Her iki tür için de ortak olarak hesaplanır:
    - avg_total_time_seconds  (dosya başına ortalama işlem süresi,
                               modellerin hızını karşılaştırmak için)
"""

import json
import os
from pathlib import Path
from collections import defaultdict


# ---------------------------------------------------------------------------
# Genel yardımcı fonksiyonlar
# ---------------------------------------------------------------------------

def determine_doc_type(stem: str):
    """
    Dosya adından (uzantısız, örn. "dekont_003") belge türünü saptar.

    İsimlendirme kuralı: belge türü isminin kendisinde alt tire (_)
    BULUNMAZ; belge türü ismi ile dosya numarası arasına tek bir alt
    tire konur (örn. "dekont_003" -> "dekont", "surucubelgesi_012" ->
    "surucubelgesi", "fatura_7" -> "fatura"). Belge türü, dolayısıyla
    dosya adının ilk alt tireden ÖNCEKİ kısmıdır.

    Bu yapı tamamen geneldir: sabit bir anahtar kelime listesine
    (DOC_TYPE_KEYWORDS vb.) ihtiyaç yoktur — fatura, banka, dekont,
    surucubelgesi ya da yeni eklenecek herhangi bir belge türü otomatik
    olarak tanınır. Aynı isimlendirme, common_fields klasöründeki
    "<belge_turu>_c.txt" dosyalarıyla eşleştirme için de kullanılır
    (bkz. load_specific_keywords).
    """
    return stem.split('_')[0].lower()


def safe_float(value):
    """
    Bir değeri float'a çevirir; None, eksik ya da sayıya çevrilemeyen
    (örn. boş string, hatalı tip) değerlerde exception fırlatmak yerine
    None döner. Böylece eksik/bozuk veri içeren JSON alanları,
    ortalama hesaplarını bozmadan (avg fonksiyonunda) sessizce elenir.
    """
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def avg(values: list):
    """
    Bir sayı listesinin aritmetik ortalamasını 6 ondalık basamağa
    yuvarlayarak döner. Listedeki None değerler ortalamaya dahil
    edilmeden önce elenir; listede hiç geçerli (None olmayan) değer
    yoksa None döner (0 değil — "veri yok" ile "değer sıfır" ayrımını
    korumak için).
    """
    clean = [v for v in values if v is not None]
    return round(sum(clean) / len(clean), 6) if clean else None


def true_ratio(booleans: list):
    """
    Boolean listesindeki True değerlerin oranını (0.0–1.0 arası)
    döner. Örn. is_match_true_ratio veya found_true_ratio gibi
    "ne kadarı başarılı/eşleşti" metrikleri için kullanılır.
    Liste boşsa None döner.
    """
    if not booleans:
        return None
    return round(sum(1 for b in booleans if b is True) / len(booleans), 6)


# ---------------------------------------------------------------------------
# Dinamik spesifik kelime okuyucu
# ---------------------------------------------------------------------------

def load_specific_keywords(common_fields_dir, doc_type: str) -> list:
    """
    Belge türüne özel aranacak anahtar kelimeleri, common_fields_dir
    klasöründeki "<doc_type>_c.txt" dosyasından okur (örn. doc_type=
    'dekont' için common_fields_dir/dekont_c.txt). Bu isimlendirme,
    determine_doc_type'ın dosya adından çıkardığı belge türü ile
    birebir eşleşir, böylece herhangi bir belge türü için ek kod
    yazmadan sadece doğru adla bir .txt dosyası eklemek yeterlidir.

    Dosya formatı:
        - Her satır bir anahtar kelime olarak kabul edilir.
        - '#' ile başlayan satırlar yorum kabul edilip atlanır.
        - Boş satırlar atlanır.

    Bu .txt dosyaları opsiyoneldir: belirtilen doc_type için dosya
    yoksa hata/uyarı vermeden sessizce boş liste döner, yani bu belge
    türü için özel kelime arama (hit-rate) hesabı yapılmaz.
    """
    txt_path = Path(common_fields_dir) / f"{doc_type}_c.txt"
    if not txt_path.exists():
        return []  # txt yoksa bu tür için keyword arama yapılmaz

    keywords = []
    with open(txt_path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            # Yorum ('#...') ve boş satırları listeye ekleme
            if not stripped or stripped.startswith("#"):
                continue
            keywords.append(stripped)
    return keywords


# ---------------------------------------------------------------------------
# Özel kelime bulma başarısı (Hit Rate)
# ---------------------------------------------------------------------------

# Türkçe'ye özgü karakterlerin ASCII karşılıklarına eşlemesi.
# OCR motorları bazen "İ/I", "Ş/S" gibi Türkçe karakterleri yanlış ya da
# tutarsız tanıyabildiği için, karşılaştırma öncesi hem beklenen kelimeyi
# hem de OCR metnini bu tabloyla normalize ederek "sahte farkları" ortadan
# kaldırıyoruz (örn. "üçüncü" ile "ucuncu" aynı kabul edilsin).
_TR_NORM_TABLE = str.maketrans(
    "ŞşĞğÜüÖöÇçİıÂâÎîÛû",
    "SsGgUuOoCcIiAaIiUu"
)

def normalize_text(text: str) -> str:
    """
    Metni küçük harfe çevirir ve Türkçe karakterleri ASCII
    karşılıklarına dönüştürür (ş→s, ğ→g, ü→u, İ→i vb.).
    Bu sayede OCR çıktısındaki karakter kodlama / büyük-küçük harf
    farklılıklarına rağmen anahtar kelime eşleşmesi güvenilir şekilde
    yapılabilir.
    """
    return text.translate(_TR_NORM_TABLE).lower()


def compute_keyword_hit_rates(file_list: list, doc_type: str, common_fields_dir) -> dict:
    """
    Belge türüne özel önemli kelimelerin (örn. "EHLİYET", "BANKA ADI" gibi
    her belgede mutlaka geçmesi beklenen ifadelerin), model çıktılarının
    kaçında doğru şekilde tanındığını yüzde olarak hesaplar.

    Çalışma mantığı:
        1. doc_type için .txt dosyasından beklenen kelime listesi okunur
           (bkz. load_specific_keywords).
        2. file_list'teki her JSON dosyasında, hem 'text' hem 'raw_text'
           alanları birleştirilip Türkçe karakter normalizasyonu
           uygulanarak arama kaynağı oluşturulur.
        3. Her kelime için, o kelimenin geçtiği dosya sayısının toplam
           dosya sayısına oranı hesaplanır.

    Döner:
        { "KELIME": 75.0, ... }
        Yani "KELIME" dosyaların %75'inde bulunmuş demektir.
        Belge türü için .txt dosyası yoksa veya file_list boşsa: {}
    """
    keywords = load_specific_keywords(common_fields_dir, doc_type)
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
        # Rapordaki key olarak normalize edilmiş (ASCII, büyük harf)
        # versiyon kullanılır; böylece çıktı JSON'unda 'Ü' gibi Türkçe
        # karakterler yerine okunması kolay ASCII anahtarlar görünür.
        normalize_text(kw).upper(): round(hit_counts[kw] / total * 100, 2)
        for kw in keywords
    }


# ---------------------------------------------------------------------------
# Ortak alan (common_field) hesaplamaları
# ---------------------------------------------------------------------------

def extract_common_fields_data(data: dict) -> dict:
    """
    "common_field"lar, belge türünden bağımsız olarak her belgede ortak
    bulunan alanlardır (örn. tarih, belge no gibi). Bu fonksiyon, tek bir
    belge JSON'unun common_field_results bölümünden bu alanlara ait ham
    performans verilerini çıkarır (henüz ortalaması alınmamış, ham liste
    halinde) — asıl ortalama hesabı aggregate_common_fields'ta yapılır.

    Dönen:
        cer_list              : common field başına CER değerleri
        wer_list              : common field başına WER değerleri
        found_list            : common field başına found bool değerleri
                                 (alan belgede bulunabildi mi?)
        common_fmr            : belge düzeyi common_field_match_ratio (ya float ya None)
        confidence_list       : matched_word_indices üzerinden bu common
                                 field'lara karşılık gelen word'lerin
                                 confidence değerleri
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

            # Bu common field'ın hangi word'lere karşılık geldiği
            # matched_word_indices içinde tutulur; o index'lerdeki
            # word'lerin confidence değerlerini topluyoruz.
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
    Sürücü belgesi filtreleme kuralı.

    Bir word'ün geçerli sayılması için matched_field,
    matched_field_value veya matched_substring alanlarından en az
    biri dolu olmalıdır. Üçü de null ise, bu word ground-truth'taki
    hiçbir beklenen alanla eşleştirilememiş demektir (örn. sayfadaki
    alakasız/gürültü metin) ve metriklerden dışlanır.
    """
    return not (
        word.get("matched_field")       is None and
        word.get("matched_field_value") is None and
        word.get("matched_substring")   is None
    )


def process_surucubelgesi(filepath: Path) -> dict:
    """
    Tek bir sürücü belgesi JSON'unu okuyup ham (henüz ortalaması
    alınmamış) performans verilerini çıkarır. is_word_valid_surucubelgesi
    filtresine takılan word'ler bu listelere hiç eklenmez.

    Dosya okunamaz/bozuksa (JSON hatası, eksik dosya vb.) uyarı basılır
    ve tüm listeler boş olan bir "empty" sonuç döndürülür — script bu
    tek dosya yüzünden çökmez, o dosya sadece rapora katkı vermez.

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
    Tek bir dekont JSON'unu okuyup ham performans verilerini çıkarır.

    surucubelgesi'nden farklı olarak burada word bazlı bir eşleştirme
    filtresi YOKTUR — sayfadaki tüm word'lerin confidence değeri
    hesaba katılır. Dosya okunamazsa uyarı basılır ve boş bir "empty"
    sonuç döndürülür.

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
    Bir model/belge türü grubundaki tüm dosyalardan (extract_common_fields_data
    ile toplanmış) ham common-field listelerini tek bir havuzda birleştirir
    ve bu havuz üzerinden nihai 'common_fields' metrik objesini üretir.

    Her dosyanın kendi ortalaması alınmaz; tüm değerler önce
    birleştirilir, ortalama en sonda bir kez hesaplanır (mikro
    ortalama). Böylece daha çok common field içeren dosyalar
    ortalamada doğal olarak daha fazla ağırlığa sahip olur.
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
    Aynı model/versiyona ait tüm sürücü belgesi dosyalarını
    (file_list) tek tek process_surucubelgesi ile işler, ham verileri
    biriktirir ve bu modelin nihai özet metriklerini (rapor bloğunu)
    üretir. Modeller arası karşılaştırma bu fonksiyonun döndürdüğü
    obje üzerinden yapılır.
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


def compute_dekont_metrics(file_list: list, common_fields_dir, doc_type: str = "dekont") -> dict:
    """
    Aynı model/versiyona ait tüm dekont dosyalarını (file_list) tek tek
    process_dekont ile işler ve nihai özet metrikleri üretir.

    surucubelgesi'nin aksine CER/WER/field_match_ratio gibi metrikler
    dekont için ground-truth'ta anlamlı şekilde tutulmadığından rapora
    dahil edilmez; yalnızca avg_confidence ve common_fields eklenir.
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
        "specific_keyword_success_rates": compute_keyword_hit_rates(file_list, doc_type, common_fields_dir),
    }


# ---------------------------------------------------------------------------
# Ana fonksiyon
# ---------------------------------------------------------------------------

def generate_report(
    outputs_dir: str = "outputs",
    common_fields_dir: str = None,
    models_to_process: list = None,
    report_path: str = None,
) -> None:
    """
    outputs/ dizinini baştan sona tarar, her (model, versiyon, belge türü)
    kombinasyonu için performans metriklerini hesaplar ve hepsini tek bir
    comparison_report.json dosyasında toplayıp diske yazar.

    Parametreler:
        outputs_dir        : Modellerin JSON çıktılarının bulunduğu ana klasör.
        common_fields_dir  : Belge türü başına özel anahtar-kelime listelerini
                             içeren .txt dosyalarının bulunduğu klasör
                             (bkz. load_specific_keywords).
                             None geçilirse keyword arama yapılmaz.
        models_to_process  : [(engine, model_name), ...] şekilde SADECE
                             işlenecek modellerin listesi (örn. belirli bir
                             modeli yeniden çalıştırıp raporu güncellemek
                             için). None ise outputs/ altındaki tüm
                             modeller dahil edilir.

    Beklenen klasör yapısı:
        outputs/
            <model_adi>/
                <versiyon>/
                    dekont.json, dekont2.json, surucubelgesi1.json, ...

    Not: Fonksiyon her çalıştığında önce varsa eski comparison_report.json
    silinir, böylece rapor her seferinde sıfırdan ve tutarlı şekilde
    üretilir (eski/artık verinin rapora karışması engellenir).
    """
    base = Path(outputs_dir)
    if not base.is_dir():
        raise FileNotFoundError(
            f"Dizin bulunamadi: {base.resolve()}\n"
            "Scripti 'outputs/' klasörünün bir üst dizininden çalıştırın."
        )

    # Rapor çıktı yolu: report_path verilmisse onu kullan, yoksa outputs_dir altı
    out_path = Path(report_path) if report_path else base / "comparison_report.json"
    if out_path.exists():
        out_path.unlink()
        print("[REPORT] Eski rapor silindi, sifirdan uretiliyor...")

    # models_to_process listesini hızlı üyelik kontrolü için set'e çevir:
    # {(engine, version), ...}
    allowed_models = None
    if models_to_process is not None:
        allowed_models = {(e, m) for e, m in models_to_process}

    # Dosyaları tarama sırasında belge türüne ve modele göre gruplandığı
    # ara veri yapısı: index[doc_type][model_label] = [Path, Path, ...]
    index: dict = defaultdict(lambda: defaultdict(list))

    for model_dir in sorted(base.iterdir()):
        if not model_dir.is_dir():
            continue
        for version_dir in sorted(model_dir.iterdir()):
            if not version_dir.is_dir():
                continue

            # models_to_process belirtildiyse, listede olmayan
            # model/versiyon kombinasyonlarını atla
            if allowed_models is not None:
                if (model_dir.name, version_dir.name) not in allowed_models:
                    continue

            model_label = f"{model_dir.name}/{version_dir.name}"
            for jf in sorted(version_dir.glob("*.json")):
                doc_type = None
                try:
                    with open(jf, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        cf_source = data.get("common_fields_source")
                        if cf_source:
                            doc_type = Path(cf_source).stem.split('_')[0].lower()
                except Exception:
                    pass

                if not doc_type:
                    doc_type = determine_doc_type(jf.stem)

                if doc_type is None:
                    continue
                index[doc_type][model_label].append(jf)

    if not index:
        print("Hic gecerli JSON bulunamadi.")
        return

    report: dict = {}

    for doc_type in sorted(index.keys()):
        report[doc_type] = {}
        print(f"\n[{doc_type.upper()}]")

        for model_label in sorted(index[doc_type].keys()):
            file_list = index[doc_type][model_label]
            print(f"  {model_label}  ({len(file_list)} dosya)")

            if doc_type == "surucubelgesi":
                metrics = compute_surucubelgesi_metrics(file_list, common_fields_dir)
            elif doc_type == "dekont":
                metrics = compute_dekont_metrics(file_list, common_fields_dir, doc_type="dekont")
            else:
                # Sürücü belgesi dışındaki diğer tüm belgeler (kimlik, fatura vb.)
                # için dekont hesaplama mantığıyla ortak alan ve güven metriklerini hesapla
                metrics = compute_dekont_metrics(file_list, common_fields_dir, doc_type=doc_type)

            report[doc_type][model_label] = metrics

            cf = metrics.get("common_fields", {})
            print(
                f"    avg_conf={metrics.get('avg_confidence')}  "
                f"avg_time={metrics.get('avg_total_time_seconds')}s  "
                f"cf_found%={cf.get('found_true_ratio')}"
            )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=True, indent=4)

    print(f"\n Rapor kaydedildi: {out_path.resolve()}")


# ---------------------------------------------------------------------------
# Giriş noktası
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Yollar, script'in çalıştırıldığı dizinden değil script'in kendi
    # konumundan türetilir; böylece script başka bir dizinden
    # çağrılsa bile doğru outputs/ ve common_fields/ klasörlerini bulur.
    script_dir        = os.path.dirname(os.path.abspath(__file__))
    outputs_dir       = os.path.join(script_dir, "outputs")
    common_fields_dir = os.path.join(script_dir, "inputs", "truths", "common_fields")
    generate_report(
        outputs_dir=outputs_dir,
        common_fields_dir=common_fields_dir,
        models_to_process=None,   # None = outputs/ altındaki tüm modeller
    )
