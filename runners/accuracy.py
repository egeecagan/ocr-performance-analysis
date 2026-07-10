"""
accuracy.py

ARTIK doğrudan doğruluk hesaplaması YAPMIYOR — tüm alan-bazlı CER/WER
karşılaştırması lcs_cer.py'ye taşındı (bkz. o dosyadaki modül docstring'i).
Bu dosya artık SADECE şu ortak yapı taşlarını sağlayan bir yardımcı
kütüphane:

  - normalize_text / turkish_lower / turkish_to_ascii: Türkçe-uyumlu metin
    normalizasyonu (lcs_cer.py bunları YENİDEN KULLANIYOR — kopyalamıyor).
  - load_common_fields / detect_common_fields_file: inputs/truths/
    common_fields/*.txt dosyalarını okuma ve görsel adına göre otomatik
    seçme — main.py bunları doğrudan çağırıyor.

Bu dosyanın kendi eski check_field_in_text / check_all_fields /
find_best_matching_field / enrich_words_with_field_matches fonksiyonları
(RapidFuzz fuzz.ratio/partial_ratio tabanlı, 0-100 benzerlik skoru
üreten) KALDIRILDI — main.py artık TÜM doğruluk kontrolünü lcs_cer.py'deki
LCS (longest-common-substring) + Levenshtein CER/WER tabanlı fonksiyonlarla
yapıyor (check_all_fields_lcs_cer_with_bbox, enrich_words_with_field_
matches_lcs). Bu değişikliğin gerekçesi: CER/WER hem daha basit hem daha
doğru sonuç veriyor (OCR'ın bir kelimeyi TAMAMEN atladığı durumlarda
fuzzy skorun ürettiği yanıltıcı/rastgele eşleşmeler ortadan kalkıyor) VE
aynı zamanda piksel konumunu (bbox) da hesaba katıyor.

Komşu-kutu-birleştirme fonksiyonları da (_are_neighbors,
_vertical_overlap_ratio, _group_into_lines*, _generate_merge_candidates)
AYNI GEREKÇEYLE KALDIRILDI: eski fuzzy yöntemde, bir kelimenin OCR
tarafından birden fazla kutuya bölünmesi (örn. "İstanbul" -> "İstan" +
"bul") skoru düşürebiliyordu, bu yüzden komşu kutuları birleştirip TEKRAR
deniyorduk. CER/WER'e geçince bu ihtiyaç ORTADAN KALKTI: find_lcs_match
zaten "bu kısa metin, referansın İÇİNDE bir yerde geçiyor mu" diye
baktığı için, bölünmüş bir kelimenin HER PARÇASI TEK BAŞINA bile genelde
referansın TAM bir alt-dizesi olarak bulunup cer=0.0 alıyor — ayrıca
birleştirmeye gerek kalmıyor (bkz. lcs_cer.py'deki enrich_words_with_
field_matches_lcs docstring'i, detaylı gerekçe orada).

=== Türkçe büyük/küçük harf çevirimi NEDEN ÖZEL ===
Python'un standart str.lower() Türkçe'de YANLIŞ sonuç verir:
  - "İ".lower() -> "i̇" (i + AYRI bir "combining dot above" karakteri,
    Unicode'da 2 karaktere ayrışır, görünüşte aynı ama karakter sayısı
    değişir)
  - "I".lower() -> "i" (ama Türkçe'de "I"nın küçüğü "ı" olmalı, "i" değil)

Bu fark benzerlik hesaplamasını BOZAR: karakter sayısı normalizasyon
sırasında sessizce değişirse, edit distance yanlış çıkar. Bu yüzden
standart İngilizce odaklı .lower() kullanmıyoruz; önce Türkçe'ye özel
İ->i ve I->ı dönüşümünü ELLE yapıp, ardından genel .lower() uyguluyoruz.
"""

import re
from pathlib import Path


def turkish_lower(text):
    """
    Python'un standart str.lower()'ından farklı olarak, Türkçe büyük/küçük
    harf kurallarına uyar:
      İ (U+0130, noktalı büyük İ) -> i
      I (U+0049, noktasız büyük I) -> ı

    Bu ikisini ELLE çevirdikten sonra, kalan tüm karakterler için standart
    .lower() güvenle uygulanabilir (İngilizce ve diğer Latin karakterler
    için zaten doğru çalışıyor).
    """
    text = text.replace("İ", "i").replace("I", "ı")
    return text.lower()


# Türkçe aksanlı karakterlerin ASCII karşılıkları. OCR motorları bazen
# "ş"yi "s", "ü"yü "u" gibi okur — bu KARAKTER TANIMA hatası değil,
# genelde fontun/modelin Türkçe karaktere tam destek vermemesinden
# kaynaklanır. RapidFuzz (karakter bazlı benzerlik) "SÜRÜCÜ" ile "SURUCU"
# arasında SADECE %50 benzerlik buluyor (test edip doğruladık) — aksan
# farkı küçük görünse de algoritma için öyle değil. Bu yüzden bu
# normalizasyon AYRI bir adım olarak gerekiyor, sadece fuzzy-match eşiğine
# güvenmek yetersiz kalıyor.
_TURKISH_TO_ASCII_MAP = {
    "ş": "s", 
    "Ş": "S",
    "ü": "u", 
    "Ü": "U",
    "ç": "c", 
    "Ç": "C",
    "ğ": "g", 
    "Ğ": "G",
    "ö": "o", 
    "Ö": "O",
    "ı": "i", 
    "İ": "I",
}


def turkish_to_ascii(text):
    """
    Türkçe aksanlı karakterleri ASCII karşılıklarına çevirir (ş->s,
    ü->u, ç->c, ğ->g, ö->o, ı/İ->i/I). Bu, "SÜRÜCÜ" ile "SURUCU"yu (veya
    OCR'ın aksanı kaçırdığı benzer durumları) AYNI saymak istediğinizde
    kullanılır.

    İSTEĞE BAĞLIDIR — varsayılan olarak kapalı, çünkü bazı senaryolarda
    (örn. bir motorun Türkçe karakter desteğini özellikle test etmek
    istiyorsanız) "Ş" ile "S"yi farklı saymak isteyebilirsiniz. Açmak
    için ilgili fonksiyonlara ascii_normalize=True geçirin.
    """
    for tr_char, ascii_char in _TURKISH_TO_ASCII_MAP.items():
        text = text.replace(tr_char, ascii_char)
    return text


# Normalizasyon: noktalama işaretlerini kaldırma + fazla boşlukları teke
# indirme + baştaki/sondaki boşlukları kırpma. Önceden jiwer.transforms
# kullanıyorduk (CER/WER hesaplaması jiwer'a bağımlıydı) — artık CER/WER
# kullanılmadığı için bu küçük adımı kendimiz, ek bir bağımlılık
# gerektirmeden yapıyoruz.
_PUNCTUATION_PATTERN = re.compile(r"[^\w\s]", flags=re.UNICODE)
_MULTIPLE_SPACES_PATTERN = re.compile(r"\s+")


def normalize_text(text, ascii_normalize=False):
    """
    Karşılaştırmadan ÖNCE her iki metne (ground truth ve tahmin)
    uygulanan ortak normalizasyon: Türkçe-uyumlu küçük harfe çevirme +
    noktalama temizliği + boşluk sadeleştirme.

    ascii_normalize=True ise, EK olarak Türkçe aksanlı karakterler ASCII
    karşılıklarına çevrilir (ş->s, ü->u, vb.) — "SÜRÜCÜ" ile "SURUCU"yu
    aynı saymak için.
    """
    if text is None:
        return ""
    text = turkish_lower(text)
    if ascii_normalize:
        text = turkish_to_ascii(text)
    text = _PUNCTUATION_PATTERN.sub("", text)
    text = _MULTIPLE_SPACES_PATTERN.sub(" ", text)
    return text.strip()


# find_best_matching_field_lcs'te (lcs_cer.py) kullanılan minimum kelime
# uzunluğu. Bundan KISA bir OCR kutusu (örn. "9", "TL", ".") güvenilir
# şekilde eşleştirilemez — çok kısa bir string, uzun bir alan değerinin
# HER ZAMAN bir yerinde tesadüfen bulunabilir (örn. "9" rakamı, içinde 9
# geçen herhangi bir tarihle %100 "eşleşir" ama bu anlamsız bir eşleşme).
# Bu eşiğin altındaki kutular için matched_field=None döner — "bu kutu
# çok kısa, güvenilir şekilde bir alana atfedilemedi" anlamına gelir, web
# arayüzünde böyle gösterilmelidir.
MIN_WORD_LENGTH_FOR_FIELD_MATCH = 3


def load_common_fields(common_fields_path):
    """
    Bir belge tipinin "ortak/sabit kelimeler" .txt dosyasını okuyup,
    check_all_fields/find_best_matching_field'ın beklediği {alan_adı:
    değer} sözlüğüne çevirir.

    Dosya formatı: her satırda BİR kelime/kelime grubu (bkz.
    inputs/truths/common_fields/dekont.txt örneği). "#" ile başlayan
    satırlar ve boş satırlar YORUM sayılır, atlanır.

    Dönen sözlükte alan ADI olarak kelimenin kendisi kullanılır (örn.
    {"DEKONT": "DEKONT", "TUTAR": "TUTAR", ...}) — çünkü bu kelimeler
    zaten kendi kendilerinin etiketi, ayrı bir isimlendirmeye gerek yok.

    Dosya yoksa boş sözlük döner (hata fırlatmaz) — main.py bu durumda
    "ortak kelime kontrolü atlandı" gibi davranabilsin diye.
    """
    common_fields_path = Path(common_fields_path)
    if not common_fields_path.exists():
        return {}

    fields = {}
    with open(common_fields_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields[line] = line

    return fields


def detect_common_fields_file(img_name, common_fields_dir):
    """
    Bir görsel dosya adına bakıp, hangi "ortak kelimeler" .txt dosyasının
    kullanılması gerektiğini OTOMATİK tespit eder.

    İsimlendirme kuralı: belge türü isminin kendisinde alt tire (_)
    BULUNMAZ; belge türü ismi ile dosya numarası arasına tek bir alt
    tire konur (örn. "dekont_4.png" -> "dekont", "surucubelgesi_1.png"
    -> "surucubelgesi", "fatura_7.png" -> "fatura"). Belge türü,
    dolayısıyla dosya adının ilk alt tireden ÖNCEKİ kısmıdır — bkz.
    generate_report.py'deki determine_doc_type ile aynı mantık.

    Bu belge türü adıyla birebir aynı adı taşıyan "<belge_turu>_c.txt"
    dosyası aranır (örn. "dekont" -> "dekont_c.txt"). Böylece herhangi
    bir belge türü (fatura, banka, dekont, surucubelgesi, ...) için ek
    kod yazmadan sadece doğru adla bir .txt dosyası eklemek yeterlidir.

    Dosya yoksa None döner — main.py bu durumda ortak kelime kontrolünü
    atlar (sorun değil, sadece o görsel için bu kontrol yapılmaz).
    """
    common_fields_dir = Path(common_fields_dir)
    if not common_fields_dir.exists():
        return None

    import re
    doc_type = re.sub(r'[_0-9]+$', '', Path(img_name).stem).lower()

    txt_file = common_fields_dir / f"{doc_type}_c.txt"
    return txt_file if txt_file.exists() else None