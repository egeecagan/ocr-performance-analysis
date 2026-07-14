"""
lcs_cer.py

accuracy.py'den BAĞIMSIZ, AYRI bir modül. accuracy.py'deki
check_field_in_text/find_best_matching_field, RapidFuzz'ın fuzz.ratio +
kayan pencere + process.extractOne yöntemini kullanıyordu (karakter bazlı
GENEL benzerlik skoru, 0-100). Bu modül FARKLI, DAHA BASİT bir yöntem
kullanıyor:

  1. inputs/truths/common_fields/*.txt dosyasındaki HER SATIRI (örn.
     "SÜRÜCÜ BELGESİ", "TÜRKİYE CUMHURİYETİ") bir REFERANS olarak alır.
  2. O referans ile OCR çıktısının (JSON'daki "text" alanı — motorun
     ürettiği HAM, tek satırlık, sırasız metin) TAMAMI arasında, Python'un
     yerleşik difflib.SequenceMatcher.find_longest_match() fonksiyonuyla
     EN UZUN ORTAK ARDIŞIK ALT-DİZEYİ (longest common substring) bulur.
     Örnek: referans "TÜRKİYE CUMHURİYETİ" ise ve OCR metninde bir yerde
     "...TR CUMHURIYETI..." geçiyorsa, bulunan alt-dize " cumhuriyeti"
     olur (OCR "TÜRKİYE"yi hiç üretmemiş olsa bile).
  3. Referans ile BULUNAN BU ALT-DİZENİN KENDİSİ (uzatılmadan, pencere
     ile tahmin edilmeden — OLDUĞU GİBİ) arasında, KLASİK Levenshtein
     (düzenleme mesafesi) tabanlı CER (Character Error Rate) ve WER
     (Word Error Rate) hesaplar.

=== NEDEN "pencereyi referans uzunluğuna UZATMA" YAKLAŞIMI TERK EDİLDİ ===
Önceki bir sürümde, bulunan alt-dizeden başlayarak referansla AYNI
uzunlukta bir pencere kesip TÜM referansı bu pencereyle kıyaslıyorduk —
amaç, referansın eşleşmeyen kısmının (örn. OCR'ın atladığı bir kelimenin)
karakter hatası olarak da hesaba yansımasıydı. Ama bu yöntem YANLIŞ bir
varsayıma dayanıyordu: "referansın eşleşmeyen kısmı, OCR metninde AYNI
GÖRECELİ KONUMDA duruyor olmalı". OCR bir kelimeyi TAMAMEN ATLADIĞINDA
(yanlış okumak değil, hiç üretmemek), bu varsayım çöküyordu — pencere o
boşluğa hangi ALAKASIZ kelime rastgele denk geliyorsa onu yakalıyordu
(örn. "TÜRKİYE" hiç okunmamışken, pencerenin rastgele "LICENCE" gibi
tamamen alakasız bir kelimeyi yakalaması gibi).

Bunun yerine artık HİÇBİR TAHMİN/UZATMA yapmıyoruz: sadece GERÇEKTEN
bulunan alt-diziyi (" cumhuriyeti" gibi, olduğu gibi, kısa da olsa) alıp
TAM referansla (örn. "turkiye cumhuriyeti") Levenshtein ile kıyaslıyoruz.
Referansın eşleşmeyen kısmı ("turkiye ") otomatik olarak Levenshtein
mesafesine "eksik karakter" (deletion) olarak yansır — ayrıca tahmin
etmemize GEREK YOK, Levenshtein algoritması bunu zaten hesaba katıyor.
Bu hem DAHA BASİT hem DE daha doğru bir yöntem.

=== accuracy.py'deki yöntemle İLİŞKİSİ ===
Bu modül accuracy.py'nin YERİNE değil, YANINDA çalışır. accuracy.py'deki
check_all_fields (ground truth "fields" bloğu için) ve common_fields
kontrolü OLDUĞU GİBİ kalır. Bu modül, common_fields kelimeleri için AYRI,
EK bir bakış açısı (klasik CER/WER + piksel konumu) sunar.
"""

import re
from difflib import SequenceMatcher

from rapidfuzz.distance import Levenshtein

# accuracy.py'deki Türkçe-uyumlu normalizasyonu (İ/I çevirimi, noktalama
# temizliği, boşluk sadeleştirme, opsiyonel ASCII çevirimi) AYNEN yeniden
# kullanıyoruz — iki modülün normalizasyon mantığı TUTARSIZ olursa aynı
# metin için farklı modüllerden çelişkili sonuçlar çıkabilir. Bu yüzden
# KENDİ normalizasyonumuzu YENİDEN YAZMIYORUZ, accuracy.py'dekini import
# ediyoruz. MIN_WORD_LENGTH_FOR_FIELD_MATCH de aynı gerekçeyle: "çok kısa
# bir kutu güvenilir eşleştirilemez" kararı, hangi modülde olursak olalım
# AYNI olmalı.
from runners.accuracy import (
    normalize_text,
    turkish_lower,
    turkish_to_ascii,
    MIN_WORD_LENGTH_FOR_FIELD_MATCH,
)

# accuracy.py'deki FIELD_MATCH_THRESHOLD=80 (yani >=%80 benzerlik) ile
# TUTARLI bir eşik: hata oranı (error rate) cinsinden karşılığı
# 1 - 0.80 = 0.20. "found" alanı artık SADECE "en azından bir ardışık
# eşleşme var mı" (match.size > 0) sorusuna değil, "bu eşleşme YETERİNCE
# İYİ Mİ" sorusuna cevap veriyor — cer bu eşiğin ÜZERİNDEYSE, teknik
# olarak bir ardışık parça bulunmuş olsa bile "found=False" döner, çünkü
# pratikte kullanışlı/güvenilir bir eşleşme sayılmaz.
CER_FOUND_THRESHOLD = 0.20

# find_lcs_match ile AYNI noktalama deseni — accuracy.py'deki
# _PUNCTUATION_PATTERN'ın BİREBİR kopyası. Neden import ETMEK yerine
# burada tekrar tanımlandı: accuracy.py'deki orijinali TÜM STRING üzerinde
# regex.sub() ile çalışıyor, ama bu modülde (bbox versiyonunda) KARAKTER
# KARAKTER işlem yapıp her karakterin ham metindeki orijinal pozisyonunu
# takip etmemiz gerekiyor (bkz. _normalize_with_position_map) — bu yüzden
# aynı deseni tek-karakterlik eşleşme için burada ayrıca tanımlıyoruz.
# Deseni DEĞİŞTİRİRSENİZ, accuracy.py'deki orijinaliyle TUTARLI kalması
# için ikisini birlikte güncelleyin.
_PUNCTUATION_CHAR_PATTERN = re.compile(r"[^\w\s]", flags=re.UNICODE)


def _compute_cer_wer(norm_ref, matched_substring):
    """
    Referans (normalize edilmiş) ile bulunan alt-dize arasında, KLASİK
    Levenshtein tabanlı CER/WER hesaplar. Ortak, tek bir yerde tutulan
    hesaplama — find_lcs_match ve find_lcs_match_with_bbox AYNI mantığı
    kullanıyor, kopya kod olmasın diye buraya çıkarıldı.

    cer: karakter-bazlı Levenshtein mesafesi / referans uzunluğu
    wer: kelime-bazlı Levenshtein mesafesi / referanstaki kelime sayısı
      (RapidFuzz'ın Levenshtein.distance'ı kelime LİSTESİ üzerinde de
      çalışır — her kelimeyi tek bir "birim" olarak ele alır)
    """
    char_distance = Levenshtein.distance(norm_ref, matched_substring)
    cer = round(char_distance / len(norm_ref), 4)

    ref_words = norm_ref.split()
    matched_words = matched_substring.split()
    if ref_words:
        word_distance = Levenshtein.distance(ref_words, matched_words)
        wer = round(word_distance / len(ref_words), 4)
    else:
        wer = 1.0

    return cer, wer


def find_lcs_match(reference, predicted_text, ascii_normalize=True):
    """
    Bir referans string'i (örn. "TÜRKİYE CUMHURİYETİ"), OCR'ın ürettiği
    HAM metin (predicted_text) İÇİNDE, longest-common-substring yöntemiyle
    en uzun ortak ardışık parçayı bulur, sonra referans ile BULUNAN BU
    PARÇANIN KENDİSİ (uzatılmadan/tahmin edilmeden, olduğu gibi) arasında
    Levenshtein tabanlı CER/WER hesaplar.

    Parametreler:
      reference: aranan sabit/beklenen metin (örn. common_fields'taki bir
        satır)
      predicted_text: OCR motorunun ürettiği TÜM ham metin (JSON'daki
        "text" alanı)
      ascii_normalize: True ise Türkçe aksanlı karakterler ASCII
        karşılıklarına çevrilir karşılaştırmadan ÖNCE (varsayılan AÇIK)

    Döndürür (dict):
      reference: girdi olarak verilen referans
      matched_substring: OCR metninde bulunan, referansla ORTAK olan en
        uzun ardışık parça (normalize edilmiş hali, OLDUĞU GİBİ —
        referans uzunluğuna uzatılmaz/tahmin edilmez)
      match_length: matched_substring'in uzunluğu (karakter). 0 ise
        referans, metinde HİÇ ardışık bir parça halinde bulunamadı demektir.
      found: bool — cer, CER_FOUND_THRESHOLD'un (0.20) ALTINDA mı. Yani
        SADECE "bir eşleşme var mı" değil, "bu eşleşme YETERİNCE İYİ Mİ"
        sorusuna cevap veriyor — match_length > 0 olsa bile cer yüksekse
        (örn. referansın büyük bir kısmı OCR'da hiç yoksa) found=False
        döner.
      cer: float (0.0-1.0+) — Character Error Rate. Referansın
        eşleşmeyen kısımları da (OCR'ın atladığı kelimeler dahil)
        Levenshtein mesafesine "eksik karakter" olarak otomatik yansır.
      wer: float (0.0-1.0+) — Word Error Rate, aynı mantık kelime bazında.

    reference veya predicted_text boş/None ise, ya da hiç ardışık eşleşme
    yoksa: found=False, cer=1.0, wer=1.0 döner.
    """
    empty_result = {
        "reference": reference,
        "matched_substring": None,
        "match_length": 0,
        "found": False,
        "cer": 1.0,
        "wer": 1.0,
    }

    if not reference or not predicted_text:
        return empty_result

    norm_ref = normalize_text(reference, ascii_normalize=ascii_normalize)
    norm_pred = normalize_text(predicted_text, ascii_normalize=ascii_normalize)

    if not norm_ref or not norm_pred:
        return empty_result

    matcher = SequenceMatcher(None, norm_ref, norm_pred, autojunk=False)
    match = matcher.find_longest_match(0, len(norm_ref), 0, len(norm_pred))

    if match.size == 0:
        return empty_result

    matched_substring = norm_pred[match.b:match.b + match.size]
    cer, wer = _compute_cer_wer(norm_ref, matched_substring)

    return {
        "reference": reference,
        "matched_substring": matched_substring,
        "match_length": match.size,
        "found": cer <= CER_FOUND_THRESHOLD,
        "cer": cer,
        "wer": wer,
    }


def check_all_fields_lcs_cer(fields, predicted_text, ascii_normalize=True):
    """
    check_all_fields (accuracy.py) ile AYNI arayüz mantığında, ama
    find_lcs_match kullanan versiyonu. Genelde common_fields.txt'ten
    load_common_fields ile üretilen {"SÜRÜCÜ BELGESİ": "SÜRÜCÜ BELGESİ",
    ...} sözlüğüyle çağrılması beklenir.

    Döndürür (dict):
      field_results: dict — her alan için find_lcs_match çıktısı
      fields_total: int — toplam alan sayısı
      fields_found: int — found=True olan alan sayısı
      average_cer / average_wer: float | None — sadece found=True olan
        alanlar üzerinden ortalama. Hiç found=True alan yoksa None.
    """
    if not fields:
        return {
            "field_results": {},
            "fields_total": 0,
            "fields_found": 0,
            "average_cer": None,
            "average_wer": None,
        }

    field_results = {}
    for field_name, field_value in fields.items():
        field_results[field_name] = find_lcs_match(
            field_value, predicted_text, ascii_normalize=ascii_normalize
        )

    fields_total = len(field_results)
    measured = [r for r in field_results.values() if r["found"]]
    fields_found = len(measured)

    if measured:
        average_cer = round(sum(r["cer"] for r in measured) / len(measured), 4)
        average_wer = round(sum(r["wer"] for r in measured) / len(measured), 4)
    else:
        average_cer = None
        average_wer = None

    return {
        "field_results": field_results,
        "fields_total": fields_total,
        "fields_found": fields_found,
        "average_cer": average_cer,
        "average_wer": average_wer,
    }


def _normalize_with_position_map(text, ascii_normalize=True):
    """
    accuracy.py'deki normalize_text ile AYNI dönüşümleri (Türkçe-uyumlu
    küçük harfe çevirme, opsiyonel ASCII normalizasyonu, noktalama
    temizliği, boşluk sadeleştirme) uygular — AMA farkı: normalize
    edilmiş metindeki HER karakterin, ORİJİNAL (ham) metindeki hangi
    karaktere karşılık geldiğini gösteren bir POZİSYON HARİTASI da üretir.

    === Bu neden gerekli ===
    find_lcs_match_with_bbox, LCS eşleşmesini normalize edilmiş metin
    üzerinde buluyor — ama sonra bulunan konumun, "words" listesindeki
    HANGİ kutuya (piksele) denk geldiğini bulmamız gerekiyor. normalize_
    text tek başına normalize edilmiş STRING'i döndürür, pozisyon
    bilgisini KAYBEDER (noktalama silindiğinde/boşluklar sadeleştiril-
    diğinde hangi karakterin ham metindeki hangi indekse ait olduğu
    bilgisi gider). Bu fonksiyon normalizasyonu KARAKTER KARAKTER
    uygulayıp bu bilgiyi KORUYARAK ilerliyor.

    Döndürür: (normalized_text, position_map)
      normalized_text: normalize_text(text, ascii_normalize) ile
        (pratikte) AYNI sonucu üretir.
      position_map: normalized_text ile AYNI uzunlukta bir liste;
        position_map[i] = normalized_text[i] karakterinin, ORİJİNAL
        (ham) text içindeki karakter indeksi.
    """
    if not text:
        return "", []

    chars = []
    positions = []
    for idx, ch in enumerate(text):
        new_ch = turkish_lower(ch)
        if ascii_normalize:
            new_ch = turkish_to_ascii(new_ch)
        if not new_ch:
            continue
        # Türkçe alfabede pratikte gerçekleşmez ama teorik bir güvenlik:
        # .lower() bazı nadir Unicode karakterlerde birden fazla karakter
        # üretebilir — pozisyon eşlemesini bozmamak için sadece ilk
        # karakteri alıyoruz (1 girdi karakteri -> 1 çıktı karakteri kuralı).
        chars.append(new_ch[0])
        positions.append(idx)

    filtered_chars = []
    filtered_positions = []
    prev_was_space = False

    for ch, pos in zip(chars, positions):
        if _PUNCTUATION_CHAR_PATTERN.match(ch):
            continue
        if ch.isspace():
            if prev_was_space:
                continue
            ch = " "
            prev_was_space = True
        else:
            prev_was_space = False
        filtered_chars.append(ch)
        filtered_positions.append(pos)

    start = 0
    while start < len(filtered_chars) and filtered_chars[start] == " ":
        start += 1
    end = len(filtered_chars)
    while end > start and filtered_chars[end - 1] == " ":
        end -= 1

    normalized_text = "".join(filtered_chars[start:end])
    position_map = filtered_positions[start:end]

    return normalized_text, position_map


def find_lcs_match_with_bbox(reference, words, ascii_normalize=True):
    """
    find_lcs_match'in BBOX-FARKINDA versiyonu — CER/WER hesaplama mantığı
    TAMAMEN AYNI (tüm metin üzerinde longest-common-substring ile bulunan
    alt-dizeyi, OLDUĞU GİBİ, referansla Levenshtein kıyaslaması), ama EK
    olarak: bulunan alt-dizenin "words" listesindeki HANGİ kutu(lar)a
    denk geldiğini bulup, o kutuların pikselini (bbox) de döndürür.

    === Yöntem ===
    1. "words" listesinden HAM metni yeniden oluştur: " ".join(kelimeler)
       — runner'ların "text" alanını ürettiği yöntemin AYNISI. Aynı
       zamanda her kelimenin bu ham metin içindeki [başlangıç, bitiş)
       karakter aralığını da (word_spans_raw) kaydediyoruz.
    2. Ham metni _normalize_with_position_map ile normalize et — normalize
       edilmiş her karakterin ham metindeki karşılığını (position_map) da
       elde ediyoruz.
    3. find_lcs_match'teki AYNI longest-common-substring mantığıyla en iyi
       eşleşmeyi (match.a, match.b, match.size) bul, CER/WER hesapla —
       HİÇBİR pencere uzatma/tahmin YOK, sadece GERÇEKTEN bulunan kısım.
    4. position_map kullanarak match.b .. match.b+match.size aralığını
       HAM metin karakter aralığına çevir.
    5. Bu ham aralıkla ÖRTÜŞEN "words" kutularını bul — bunlar eşleşmeyi
       oluşturan GERÇEK OCR kutularıdır (rastgele/alakasız bir kelime
       ASLA karışmaz, çünkü sadece GERÇEKTEN eşleşen karakterlerin
       konumu kullanılıyor).

    Parametreler:
      reference: aranan sabit/beklenen metin (örn. "TÜRKİYE CUMHURİYETİ")
      words: runner'ın ürettiği "words" listesi — her elemanda EN AZ
        "text" ve "bbox": [x1, y1, x2, y2] alanları bulunmalı. Bu liste,
        runner'ın "text" alanını ürettiği SIRAYLA olmalı.
      ascii_normalize: bkz. find_lcs_match

    Döndürür (dict):
      reference, matched_substring, match_length, found, cer, wer: bkz.
        find_lcs_match — BİREBİR AYNI anlam ve hesaplama.
      matched_word_indices: eşleşen ham metin aralığıyla ÖRTÜŞEN "words"
        listesindeki orijinal indeksler (örn. [17] — tek kutu).
      bbox: [x1, y1, x2, y2] — matched_word_indices'teki TÜM kutuların
        dış sınırlayıcı kutusu, web arayüzünde tek bir dikdörtgen olarak
        vurgulamak için. Eşleşme bulunamadıysa None.
      bboxes: matched_word_indices'teki HER kutunun KENDİ (ayrı, dar)
        bbox'ı, liste halinde — "bbox" tek bir GENİŞ dikdörtgen olduğunda
        (örn. eşleşen kelimeler belgede birbirinden uzaksa) web arayüzü
        bunun yerine HER kelimeyi ayrı ayrı, dar bir kutuyla vurgulamak
        isteyebilir.

    reference veya words boş/None ise, ya da hiç ardışık eşleşme yoksa
    ya da bulunan eşleşme yeterince iyi değilse (cer > CER_FOUND_
    THRESHOLD): found=False döner. bbox/bboxes ise found=False'ta değil,
    "gerçekten hiçbir words kutusuyla örtüşmedi" durumunda None olur —
    yani found=False olsa bile bbox dolu gelebilir (kötü ama konumu belli
    bir eşleşme), bu ayrımı web arayüzü "düşük güven" göstergesi olarak
    kullanabilir.
    """
    empty_result = {
        "reference": reference,
        "matched_substring": None,
        "match_length": 0,
        "matched_word_indices": None,
        "bbox": None,
        "bboxes": None,
        "found": False,
        "cer": 1.0,
        "wer": 1.0,
    }

    if not reference or not words:
        return empty_result

    # --- "words" listesinden HAM metni ve kelime konumlarını yeniden oluştur ---
    raw_text_parts = [w.get("text", "") for w in words]
    raw_text = " ".join(raw_text_parts)

    if not raw_text.strip():
        return empty_result

    word_spans_raw = []
    cursor = 0
    for token in raw_text_parts:
        start = cursor
        end = start + len(token)
        word_spans_raw.append((start, end))
        cursor = end + 1  # +1: kelimeler arasına konan tek boşluk

    norm_ref = normalize_text(reference, ascii_normalize=ascii_normalize)
    norm_pred, position_map = _normalize_with_position_map(raw_text, ascii_normalize=ascii_normalize)

    if not norm_ref or not norm_pred:
        return empty_result

    # --- Longest common substring ile eşleşmeyi bul (find_lcs_match ile AYNI) ---
    matcher = SequenceMatcher(None, norm_ref, norm_pred, autojunk=False)
    match = matcher.find_longest_match(0, len(norm_ref), 0, len(norm_pred))

    if match.size == 0:
        return empty_result

    matched_substring = norm_pred[match.b:match.b + match.size]
    cer, wer = _compute_cer_wer(norm_ref, matched_substring)

    # --- Bulunan alt-dizenin (match.b .. match.b+match.size) konumunu
    # HAM metin karakter aralığına çevir ---
    if match.b >= len(position_map):
        return {
            "reference": reference,
            "matched_substring": matched_substring,
            "match_length": match.size,
            "matched_word_indices": None,
            "bbox": None,
            "bboxes": None,
            "found": cer <= CER_FOUND_THRESHOLD,
            "cer": cer,
            "wer": wer,
        }

    raw_start = position_map[match.b]
    last_idx = min(match.b + match.size, len(position_map)) - 1
    raw_end = position_map[last_idx] + 1 if last_idx >= match.b else raw_start + 1

    # --- Bu ham aralıkla ÖRTÜŞEN kelime kutularını bul ---
    matched_word_indices = [
        idx for idx, (w_start, w_end) in enumerate(word_spans_raw)
        if w_start < raw_end and w_end > raw_start
    ]

    if not matched_word_indices:
        return {
            "reference": reference,
            "matched_substring": matched_substring,
            "match_length": match.size,
            "matched_word_indices": None,
            "bbox": None,
            "bboxes": None,
            "found": cer <= CER_FOUND_THRESHOLD,
            "cer": cer,
            "wer": wer,
        }

    matched_words = [words[i] for i in matched_word_indices]
    xs1 = [w["bbox"][0] for w in matched_words]
    ys1 = [w["bbox"][1] for w in matched_words]
    xs2 = [w["bbox"][2] for w in matched_words]
    ys2 = [w["bbox"][3] for w in matched_words]
    bbox = [min(xs1), min(ys1), max(xs2), max(ys2)]
    bboxes = [w["bbox"] for w in matched_words]

    return {
        "reference": reference,
        "matched_substring": matched_substring,
        "match_length": match.size,
        "matched_word_indices": matched_word_indices,
        "bbox": bbox,
        "bboxes": bboxes,
        "found": cer <= CER_FOUND_THRESHOLD,
        "cer": cer,
        "wer": wer,
    }


def check_all_fields_lcs_cer_with_bbox(fields, words, ascii_normalize=True):
    """
    check_all_fields_lcs_cer ile AYNI arayüz mantığında, ama
    find_lcs_match_with_bbox kullanan (bbox döndüren) versiyonu.

    Parametreler:
      fields: dict — {alan_adı: referans_metin} (genelde
        load_common_fields'tan gelen common_fields sözlüğü)
      words: runner'ın ürettiği "words" listesi (bbox'lı kutular) —
        output_data["words"]
      ascii_normalize: bkz. find_lcs_match

    Döndürür (dict): check_all_fields_lcs_cer ile AYNI yapı
      (field_results, fields_total, fields_found, average_cer,
      average_wer) — field_results içindeki her elemanda EK olarak
      "matched_word_indices", "bbox" ve "bboxes" alanları da bulunur.

    fields veya words boş/None ise tüm sayılar 0, average_cer/
    average_wer None döner.
    """
    if not fields or not words:
        return {
            "field_results": {},
            "fields_total": 0,
            "fields_found": 0,
            "average_cer": None,
            "average_wer": None,
        }

    field_results = {}
    for field_name, field_value in fields.items():
        field_results[field_name] = find_lcs_match_with_bbox(
            field_value, words, ascii_normalize=ascii_normalize
        )

    fields_total = len(field_results)
    measured = [r for r in field_results.values() if r["found"]]
    fields_found = len(measured)

    if measured:
        average_cer = round(sum(r["cer"] for r in measured) / len(measured), 4)
        average_wer = round(sum(r["wer"] for r in measured) / len(measured), 4)
    else:
        average_cer = None
        average_wer = None

    return {
        "field_results": field_results,
        "fields_total": fields_total,
        "fields_found": fields_found,
        "average_cer": average_cer,
        "average_wer": average_wer,
    }


def find_best_matching_field_lcs(word_text, fields, ascii_normalize=True):
    """
    accuracy.py'deki find_best_matching_field'ın CER/WER tabanlı eşdeğeri.
    Orada RapidFuzz'ın fuzz.partial_ratio'su kullanılıyordu (0-100 benzerlik
    skoru); burada find_lcs_match'i (longest-common-substring + Levenshtein
    CER/WER) HER alan değeriyle tek tek çağırıp, EN DÜŞÜK cer'e sahip alanı
    kazanan ilan ediyoruz.

    Bu, check_field_in_text/find_lcs_match'in TERSİ yönünde çalışır: orada
    bir referans DEĞERİNİ uzun bir metnin içinde arıyorduk. Burada TEK BİR
    KELİMEYİ (bir OCR kutusunun metni), ground truth'taki TÜM ALANLAR
    arasında arıyoruz — "bu kelime en çok hangi alana benziyor" sorusuna
    cevap veriyor. find_lcs_match(word_text, field_value) çağrısı burada
    TAM istediğimiz işi yapıyor: word_text'i (kısa) field_value (uzun
    olabilir) İÇİNDE longest-common-substring ile konumlandırıp, CER'i
    word_text'e göre hesaplıyor — yani "bu kelime, bu alanın içinde ne
    kadar iyi buluyor" sorusuna cevap.

    Parametreler:
      word_text: tek bir OCR kutusunun metni (örn. "Cenglzhan")
      fields: dict — ground truth'un "fields" bloğu, örn.
        {"ad": "Cengizhan", "soy_ad": "Şahin", ...}
      ascii_normalize: bkz. find_lcs_match

    Döndürür (dict):
      matched_field: str | None — en iyi (en düşük cer'li) eşleşen alanın ADI
      matched_field_value: str | None — o alanın ground truth DEĞERİ
      matched_substring: str | None — o alan değerinin içinde, word_text'e
        en çok benzeyen ardışık parça
      cer, wer: float — bkz. find_lcs_match (en iyi alan için)
      is_match: bool — cer, CER_FOUND_THRESHOLD'un ALTINDA mı

    word_text çok kısaysa (MIN_WORD_LENGTH_FOR_FIELD_MATCH'ten az, bkz.
    accuracy.py'deki gerekçe: kısa kutular güvenilir eşleştirilemez) ya da
    word_text/fields boşsa: matched_field=None, cer=1.0, wer=1.0,
    is_match=False döner.
    """
    empty_result = {
        "matched_field": None,
        "matched_field_value": None,
        "matched_substring": None,
        "cer": 1.0,
        "wer": 1.0,
        "is_match": False,
    }

    if not word_text or not fields:
        return empty_result

    norm_word = normalize_text(word_text, ascii_normalize=ascii_normalize)
    if len(norm_word) < MIN_WORD_LENGTH_FOR_FIELD_MATCH:
        return empty_result

    best = None  # (cer, wer, field_name, field_value, matched_substring)
    for field_name, field_value in fields.items():
        result = find_lcs_match(word_text, field_value, ascii_normalize=ascii_normalize)
        if best is None or result["cer"] < best[0]:
            best = (result["cer"], result["wer"], field_name, field_value, result["matched_substring"])

    if best is None:
        return empty_result

    cer, wer, field_name, field_value, matched_substring = best

    return {
        "matched_field": field_name,
        "matched_field_value": field_value,
        "matched_substring": matched_substring,
        "cer": cer,
        "wer": wer,
        "is_match": cer <= CER_FOUND_THRESHOLD,
    }


def enrich_words_with_field_matches_lcs(words, fields, ascii_normalize=True):
    """
    accuracy.py'deki (kaldırılan) enrich_words_with_field_matches'ın
    CER/WER tabanlı eşdeğeri — "words" listesindeki HER kutuyu,
    find_best_matching_field_lcs ile zenginleştirir. main.py bunu, JSON'a
    yazmadan ÖNCE çağırır, böylece web arayüzü hiçbir hesaplama yapmadan
    hazır veriyi okur.

    === Komşu kutu birleştirme NEDEN KALDIRILDI ===
    Eski fuzzy yöntemde (RapidFuzz partial_ratio), bir kelimenin OCR
    tarafından birden fazla kutuya bölünmesi (örn. "İstanbul" -> "İstan" +
    "bul") skoru ciddi düşürebiliyordu — bu yüzden komşu kutuları
    birleştirip TEKRAR deniyorduk. CER/WER'e (bu modül) geçince bu gerek
    KALMADI: find_lcs_match zaten "bu kısa metin, referansın İÇİNDE bir
    yerde geçiyor mu" diye baktığı için, bölünmüş bir kelimenin HER PARÇASI
    TEK BAŞINA bile genelde referansın TAM bir alt-dizesi olarak bulunur ve
    cer=0.0 alır (örn. find_lcs_match("İstan", "İstanbul") -> cer=0.0,
    find_lcs_match("bul", "İstanbul") -> cer=0.0 — ikisi de zaten mükemmel).
    Birleştirmenin fayda sağlayacağı tek durum, MIN_WORD_LENGTH_FOR_FIELD_
    MATCH eşiğinin altında kalan (1-2 karakterlik) parçalardı — bu, pratikte
    ihmal edilebilir bir uç durum. Bu yüzden accuracy.py'deki
    _generate_merge_candidates ve komşuluk (_are_neighbors, _group_into_
    lines*) fonksiyonları da ARTIK KULLANILMIYOR.

    Parametreler:
      words: runner'ın ürettiği ham words listesi, örn.
        [{"text": "Cenglzhan", "bbox": [...], "confidence": 91.2}, ...]
      fields: ground truth'un "fields" bloğu (yoksa/boşsa hiçbir
        zenginleştirme yapılmaz, words OLDUĞU GİBİ döner)
      ascii_normalize: bkz. find_lcs_match

    Döndürür: words listesinin AYNISI, her elemana find_best_matching_
    field_lcs'in döndürdüğü alanlar (matched_field, matched_field_value,
    matched_substring, cer, wer, is_match) eklenmiş olarak.

    fields boşsa, bu alanlar None/1.0/False olarak eklenir.
    """
    if not fields:
        empty_extra = {
            "matched_field": None, "matched_field_value": None,
            "matched_substring": None, "cer": 1.0, "wer": 1.0,
            "is_match": False,
        }
        return [{**word, **empty_extra} for word in words]

    return [
        {**word, **find_best_matching_field_lcs(word.get("text", ""), fields, ascii_normalize=ascii_normalize)}
        for word in words
    ]


if __name__ == "__main__":
    # --- Bağımsız/manuel test kullanımı ---
    # main.py'ye eklemek için:
    #
    #   from runners.lcs_cer import check_all_fields_lcs_cer_with_bbox
    #   ...
    #   lcs_cer_check = check_all_fields_lcs_cer_with_bbox(common_fields, output_data.get("words", []))
    #   output_data["common_fields_lcs_cer"] = lcs_cer_check

    sample_text = (
        "13.0 Rh+ 10. 11. M 4 21.10.2022 12. SURUCU BELGESI TURKIYE "
        "21.10.2032 C* 14. A1 ave TR CUMHURIYETI A2 DRIVING LICENCE A A "
        "AKSOY B1 1. B E 21.10.2022 21. 0.2 2032 21. 10.2022 21. 10.2032 "
        "2. Volkan C1 C 3. 08.01.1997 Aydin D1 ga 4c.09 Efeler B a TT "
        "4a.21.10.2022 D 4d BE - 4b.21.10.2032 AB C1E - 5 CE . D1E MEA "
        "7. DE d 21.10.2022 21.1 .10.2032 a 9.B G O6 12."
    )

    sample_common_fields = {
        "SÜRÜCÜ BELGESİ": "SÜRÜCÜ BELGESİ",
        "DRIVING LICENCE": "DRIVING LICENCE",
        "TÜRKİYE CUMHURİYETİ": "TÜRKİYE CUMHURİYETİ",
    }

    result = check_all_fields_lcs_cer(sample_common_fields, sample_text)
    import json
    print("--- Sadece text (bbox'sız) yöntem ---")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    # bbox testi için "words" listesi (gerçek runner çıktısını simüle eder)
    fake_words = []
    x_cursor = 0
    for token in sample_text.split():
        token_width = len(token) * 10
        fake_words.append({
            "text": token,
            "bbox": [x_cursor, 100, x_cursor + token_width, 130],
        })
        x_cursor += token_width + 5

    result_bbox = check_all_fields_lcs_cer_with_bbox(sample_common_fields, fake_words)
    print("\n--- bbox-farkında yöntem ---")
    print(json.dumps(result_bbox, ensure_ascii=False, indent=2))