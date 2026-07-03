"""
accuracy.py

ARTIK doğrudan doğruluk hesaplaması YAPMIYOR — tüm alan-bazlı CER/WER
karşılaştırması lcs_cer.py'ye taşındı (bkz. o dosyadaki modül docstring'i).
Bu dosya artık SADECE şu ortak yapı taşlarını sağlayan bir yardımcı
kütüphane:

  - normalize_text / turkish_lower / turkish_to_ascii: Türkçe-uyumlu metin
    normalizasyonu (lcs_cer.py bunları YENİDEN KULLANIYOR — kopyalamıyor).
  - _are_neighbors / _vertical_overlap_ratio / _group_into_lines* /
    _generate_merge_candidates: OCR motorunun bir kelimeyi birden fazla
    kutuya bölmüş olma ihtimaline karşı komşu-kutu-birleştirme mantığı.
    Bu, bbox GEOMETRİSİNE dayalı olduğu için hangi skorlama yöntemini
    (fuzzy ya da CER) kullandığımızdan TAMAMEN BAĞIMSIZ — lcs_cer.py da
    bunu YENİDEN KULLANIYOR.
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
from collections import defaultdict
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


def _are_neighbors(bbox_a, bbox_b, max_horizontal_gap_ratio=0.5):
    """
    İki bbox'ın "komşu" sayılıp sayılmayacağına karar verir — yani
    aralarında metin birleştirme denemesi yapmaya değer mi.

    Komşu sayılma şartları (ikisi de sağlanmalı):
      1. AYNI SATIRDA: y-eksenindeki (dikey) aralıkları örtüşüyor olmalı
         — yani iki kutu da yaklaşık aynı yükseklikte.
      2. YATAYDA YAKIN: aralarındaki yatay boşluk, bbox_a'nın kendi
         yüksekliğine göre "makul" bir mesafede olmalı (çok uzaktaysa
         muhtemelen ayrı kelimeler/alanlar, birleştirmeye değmez).

    bbox formatı: [x1, y1, x2, y2] (tüm 5 motorda tutarlı format).

    NOT: Bu fonksiyon artık SADECE aynı satır grubu içindeki (bkz.
    _group_into_lines) ardışık kutular için çağrılıyor — yani "aynı
    satırda mı" sorusunun asıl/kaba cevabı zaten _group_into_lines
    tarafından veriliyor. Buradaki dikey örtüşme kontrolü, aynı satır
    grubuna düşmüş ama yine de dikey olarak hiç örtüşmeyen (örn. yanlış
    gruplanmış) uç durumlara karşı ek bir güvenlik katmanı.

    max_horizontal_gap_ratio: bbox_a'nın yüksekliğine göre, izin verilen
    maksimum yatay boşluk oranı. Örn. 0.5 ise, kutunun yüksekliğinin
    yarısı kadar (ya da daha az) boşluk varsa komşu sayılır — bu, normal
    kelime arası boşluğu tolere ederken, gerçekten uzak/alakasız
    kutuları elemeye yarar.
    """
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b

    # --- 1. Aynı satırda mı? (dikey örtüşme kontrolü) ---
    vertical_overlap = min(ay2, by2) - max(ay1, by1)
    if vertical_overlap <= 0:
        return False

    # --- 2. Yatayda yeterince yakın mı? ---
    a_height = ay2 - ay1
    if a_height <= 0:
        return False

    horizontal_gap = bx1 - ax2  # b, a'nın SAĞINDA olmalı (pozitif boşluk)
    if horizontal_gap < 0:
        # b, a'nın solunda veya üst üste biniyor — bu fonksiyon sadece
        # "a'nın sağındaki komşuyu" kontrol etmek için kullanılıyor.
        return False

    return horizontal_gap <= a_height * max_horizontal_gap_ratio


def _vertical_overlap_ratio(bbox_a, bbox_b):
    """
    İki bbox'ın dikey örtüşme ORANINI (0-1 arası) döndürür — mutlak piksel
    farkı değil, İKİ KUTUDAN KÜÇÜK OLANIN yüksekliğine göre oranlanmış bir
    değer. Bu, farklı font/kutu boyutlarında (örn. küçük bir etiket
    kelimesiyle büyük bir başlık) tutarlı bir "aynı satırda mı" ölçütü
    sağlar.

    _are_neighbors'taki "vertical_overlap <= 0" kontrolünden farklı olarak
    burada bir EŞİK KARŞILAŞTIRMASI yapılmıyor — sadece oran hesaplanıp
    döndürülüyor, eşik kararı çağıran tarafta (_group_into_lines_geometric)
    veriliyor. Bu ayrım, eşiği tek bir yerde (fonksiyon çağrısında) kolayca
    ayarlanabilir kılmak için.
    """
    ay1, ay2 = bbox_a[1], bbox_a[3]
    by1, by2 = bbox_b[1], bbox_b[3]

    overlap = min(ay2, by2) - max(ay1, by1)
    min_height = min(ay2 - ay1, by2 - by1)

    if min_height <= 0:
        return 0.0

    return max(0.0, overlap) / min_height


def _group_into_lines_geometric(words, overlap_threshold=0.3):
    """
    line_id bilgisi OLMAYAN motorlar (örn. EasyOCR, RapidOCR) için: hiçbir
    kesin satır/blok hiyerarşisi verilmediğinden, kutuları SADECE bbox
    dikey örtüşmelerine bakarak satır gruplarına ayırmaya ÇALIŞIR — bu bir
    TAHMİNDİR, Tesseract/doctr'daki gibi motorun kendi kesin segmentasyonu
    DEĞİLDİR.

    Yöntem: kutuları y1'e göre sıralayıp, her kutuyu şu ana kadar oluşmuş
    satır gruplarından biriyle (o gruptaki HERHANGİ bir kutuyla dikey
    örtüşme oranı eşiği geçiyorsa) eşleştiriyoruz; eşleşme yoksa yeni bir
    grup açıyoruz. Mutlak "overlap > 0" yerine ORANSAL bir eşik (varsayılan
    %30) kullanılıyor ki hafif eğik (skewed) taranmış/fotoğraflanmış
    belgelerde satırlar piksel bazında tam hizalı olmasa da doğru
    gruplansın.

    Dönen her grup, kendi içinde x1'e (soldan sağa) göre sıralanmış olarak
    döner — _generate_merge_candidates'ın komşuluk taramasının doğru
    sırada çalışması için gerekli.

    Döndürür: [[(orijinal_indeks, kelime_dict), ...], ...] — liste
    liste, dış liste satırları, iç liste o satırdaki (indeks, kelime)
    çiftlerini x1 sırasıyla tutar.
    """
    indexed = sorted(enumerate(words), key=lambda iw: iw[1]["bbox"][1])
    lines = []

    for idx, word in indexed:
        bbox = word["bbox"]
        placed = False
        for line in lines:
            if any(
                _vertical_overlap_ratio(bbox, existing_word["bbox"]) >= overlap_threshold
                for _, existing_word in line
            ):
                line.append((idx, word))
                placed = True
                break
        if not placed:
            lines.append([(idx, word)])

    return [sorted(line, key=lambda iw: iw[1]["bbox"][0]) for line in lines]


def _group_into_lines(words):
    """
    Kutuları satır gruplarına ayırmak için İKİ yoldan birini seçer:

      1. TÜM kutularda "line_id" alanı VARSA (örn. Tesseract'ın
         block_num-par_num-line_num'dan, doctr'ın page-block-line
         indekslerinden ürettiği kimlik): motorun KENDİ kesin satır
         segmentasyonuna güveniyoruz. Bu, bbox geometrisinden tahmin
         etmekten daha güvenilirdir — motorun kendi iç algoritması
         eğiklik/gürültüye karşı daha dayanıklı çalışır.

      2. Herhangi bir kutuda "line_id" YOKSA (örn. EasyOCR, RapidOCR —
         bu motorlar satır/blok hiyerarşisi vermez): bbox dikey
         örtüşmesine dayalı GEOMETRİK TAHMİNE düşülür
         (_group_into_lines_geometric).

    Neden "TÜM kutularda var mı" kontrolü (kısmi değil): bir motorun
    çıktısında bazı kutularda line_id olup bazılarında olmaması
    beklenmez (motor ya hep sağlar ya hiç sağlamaz) — ama karışık bir
    girdi ihtimaline karşı, tutarsız/eksik line_id ile yanlış gruplama
    yapmak yerine güvenli tarafta kalıp geometrik tahmine düşülüyor.

    Döndürür: _group_into_lines_geometric ile AYNI format — satır
    gruplarının listesi, her grup x1'e göre sıralı (indeks, kelime) çiftleri.
    """
    if words and all("line_id" in w and w["line_id"] is not None for w in words):
        lines_by_id = defaultdict(list)
        for idx, w in enumerate(words):
            lines_by_id[w["line_id"]].append((idx, w))
        return [
            sorted(group, key=lambda iw: iw[1]["bbox"][0])
            for group in lines_by_id.values()
        ]

    return _group_into_lines_geometric(words)


def _generate_merge_candidates(words):
    """
    "words" listesindeki her kutu için, SAĞINDAKİ 1-2 komşu kutuyla
    birleştirilmiş metin adayları üretir (2'li ve 3'lü gruplar).

    Bu, bazı motorların tek bir kelimeyi birden fazla küçük kutuya
    bölmesi durumunda (örn. "İstanbul" -> "İstan" + "bul" gibi iki ayrı
    kutu), kutuları birleştirerek doğru kelimeyi yeniden oluşturmayı
    dener.

    === Satır gruplama NEDEN önce yapılır ===
    Bu belge tek bir satırdan ibaret değil — genelde birden fazla satır
    içerir (örn. "Ad: Cengizhan" ve "Tarih: 25.08.2014" iki AYRI satır).
    Kutuları doğrudan x1 koordinatına göre GLOBAL sıralayıp ardışık
    çiftleri denersek, farklı satırlardan kutular birbirinin "komşusu"
    gibi test edilir (örn. "Ad:" ile "Tarih:" arası denenir, ama gerçek
    komşusu olan "Ad:" + "Cengizhan" hiç yan yana gelmeyebilir çünkü
    aralarına başka satırdan bir kutu girmiştir). Bunu önlemek için önce
    _group_into_lines ile kutuları satır gruplarına ayırıyoruz, SONRA
    komşuluk taramasını SADECE her grup İÇİNDE yapıyoruz.

    Döndürür: [(birlesik_metin, [orijinal_kutu_indeksleri]), ...] listesi
    — sadece GERÇEKTEN komşu olan (aynı satır grubu + yakın) kutu
    çiftleri/üçlüleri için, farklı satırlardan ya da rastgele uzak
    kutuları asla birleştirmez.
    """
    line_groups = _group_into_lines(words)

    candidates = []

    for group in line_groups:
        n = len(group)
        for i in range(n):
            idx_a, word_a = group[i]

            # --- 2'li grup: word_a + sağındaki 1 komşu (aynı satır grubu içinde) ---
            if i + 1 < n:
                idx_b, word_b = group[i + 1]
                if _are_neighbors(word_a["bbox"], word_b["bbox"]):
                    merged_text = f"{word_a['text']} {word_b['text']}".strip()
                    candidates.append((merged_text, [idx_a, idx_b]))

                    # --- 3'lü grup: word_a + sağındaki 2 komşu ---
                    if i + 2 < n:
                        idx_c, word_c = group[i + 2]
                        if _are_neighbors(word_b["bbox"], word_c["bbox"]):
                            merged_text_3 = f"{merged_text} {word_c['text']}".strip()
                            candidates.append((merged_text_3, [idx_a, idx_b, idx_c]))

    return candidates


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

    Yöntem: common_fields_dir altındaki HER .txt dosyasının adını (uzantısız,
    örn. "dekont", "surucu_belgesi"), görsel dosya adının (küçük harfe
    çevrilmiş, alt çizgiler kaldırılmış hali) İÇİNDE arar. Örnek:
      img_name = "dekont4.png"           -> "dekont" .txt adında geçiyor mu? EVET -> dekont.txt seçilir
      img_name = "surucubelgesi1.png"    -> "surucu_belgesi" -> alt çizgisiz "surucubelgesi" görsel adında geçiyor mu? EVET -> surucu_belgesi.txt

    Birden fazla dosya adı eşleşirse (nadir, isimlendirme çakışması), İLK
    eşleşen (alfabetik sırada) kullanılır — bu durumda dosya adlarınızı
    daha belirgin seçmeniz önerilir.

    Hiçbir dosya eşleşmezse None döner — main.py bu durumda ortak kelime
    kontrolünü atlar (sorun değil, sadece o görsel için bu kontrol
    yapılmaz).
    """
    common_fields_dir = Path(common_fields_dir)
    if not common_fields_dir.exists():
        return None

    img_stem_normalized = Path(img_name).stem.lower().replace("_", "")

    txt_files = sorted(common_fields_dir.glob("*.txt"))
    for txt_file in txt_files:
        file_key = txt_file.stem.lower().replace("_", "")
        if file_key in img_stem_normalized:
            return txt_file

    return None