# Tesseract OCR İyileştirme Planı: Gelişmiş Ön İşleme ve Otomatik Parametre Optimizasyonu (Auto-Tuner)

> [!IMPORTANT]
> **SIFIR YENİ BAĞIMLILIK (Zero New Dependencies) GARANTİSİ:** 
> Şube bilgisayarlarının kurulu olan sistem düzenini bozmamak adına, **projeye hiçbir yeni Python paketi/eklenti (scikit-image vb.) eklenmeyecektir.** 
> Sauvola, Niblack ve Otsu binarizasyon algoritmaları, projede halihazırda kurulu olan **NumPy** ve **OpenCV** kütüphaneleri kullanılarak doğrudan tarafımızdan sıfırdan yazılacaktır. Şubelerdeki bilgisayarlar sadece güncellenen `_common.py` dosyasını ve optimize edilmiş `tuner_best.yaml` ayar dosyasını kullanacaktır.

---

## Önerilen Terimlerin Detaylı Açıklamaları

### 1. Sauvola ve Niblack Eşikleme (Local Adaptive Thresholding)
* **Nedir?** Belgeleri siyah-beyaz yapmak için kullanılan yerel (lokal) eşikleme algoritmalarıdır. Resmin tamamı için tek bir gri tonu eşiği belirlemek yerine, resmin üzerinde küçük pencereler gezdirerek her bölgenin kendi kontrastına ve gölgesine göre yerel bir siyah-beyaz eşiği belirler.
* **Tesseract İçin Önemi:** Telefon kameralarıyla çekilen ehliyet ve dekontlarda sıkça gölgelenmeler ve parlama farkları olur. Standart binarizasyonlar gölgedeki yazıları tamamen siyaha boğup silerken; Sauvola ve Niblack, gölgede kalan yazıları bile arka plandan pürüzsüzce ayırıp okunabilir kılar.

### 2. Otsu Eşikleme (Global Thresholding)
* **Nedir?** Resmin renk histogramını analiz ederek, zemin ile yazıyı birbirinden ayıracak en ideal tek bir eşik değerini otomatik hesaplayan klasik bir yöntemdir.
* **Tesseract İçin Önemi:** Tesseract arka planda varsayılan olarak Otsu kullanır. Ancak bunu Tesseract'ın kendi insiyatifine bırakmak yerine, ön işleme adımında bizim yönetmemiz Tesseract'a giden görüntünün kalitesini önceden görmemizi ve optimize etmemizi sağlar.

### 3. Lanczos4 Büyütme/Netleştirme (Interpolation)
* **Nedir?** Resmin çözünürlüğünü artırırken (örneğin 2 katına çıkarırken) piksellerin arasını doldurarak harf kenarlarının yumuşamasını ve bozulmasını önleyen gelişmiş bir matematiksel formüldür.
* **Tesseract İçin Önemi:** Düşük çözünürlüklü taramalarda harfler büyütüldüğünde kenarları tırtıklı veya bulanık olur. Standart büyütme yöntemleri (bilinear, nearest) harfi bulanıklaştırıp Tesseract'ın kafasını karıştırırken; Lanczos4 kenarları jilet gibi keskin ve net tutarak Tesseract'ın harfleri doğru tanımasını sağlar.

### 4. Auto-Tuner (Grid Search / Parametre Arama) Scripti
* **Nedir?** Tesseract'ın ve ön işlemenin (resize scale, psm, dpi, sauvola k değeri, block size vb.) en iyi hangi ayarlarla okuma yaptığını bulan bir test aracıdır.
* **Tuner Performansı:** Bu arama işlemi yüzlerce farklı kombinasyonu tek tek deneyip CER/WER (karakter/kelime hata oranı) ölçeceğinden çalışma süresi uzun olabilir. Ancak bu işlem sadece geliştirici bilgisayarında **bir kereliğine** çalıştırılacaktır. Şube bilgisayarlarında bu arama işlemi kesinlikle çalışmayacak; şubeler sadece bu aramanın sonucunda çıkan tek satırlık YAML dosyasını kullanacaktır.

---

## Prosedür ve Yapılacak Değişiklikler

### 1. Ortak Ön İşleme Modülü (`runners/_common.py`)

#### [MODIFY] [runners/_common.py](file:///c:/Users/maraz/Desktop/ocr/runners/_common.py)
* **Yeni Binarizasyon Metotları:** 
  * `_sauvola_threshold(gray, window_size, k, R=128)`: NumPy ve OpenCV `cv2.boxFilter` yardımıyla gölgeli/düşük kontrastlı kağıtlar için son derece hızlı ve performanslı yerel binarizasyon.
  * `_niblack_threshold(gray, window_size, k)`: Yerel standart sapma tabanlı eşikleme.
  * `_otsu_threshold(gray)`: Global Otsu binarizasyonu.
* **Resizing Interpolasyonu:** 
  * `preprocess_image` fonksiyonuna `resize_interpolation` desteği eklenecektir. `cubic` (varsayılan) ve `lanczos` (`cv2.INTER_LANCZOS4`) seçenekleri sunulacaktır.
* **Eşikleme (Thresholding) Genişletmesi:** 
  * `threshold` adımında `threshold_method` (`adaptive`, `otsu`, `sauvola`, `niblack`) parametresi okunacak ve ilgili binarizasyon çalıştırılacaktır.

---

### 2. Backend API Modülü (`api.py`)

#### [MODIFY] [api.py](file:///c:/Users/maraz/Desktop/ocr/api.py)
* **`_PREPROCESSING_SCHEMA` Güncellemesi:**
  * `resize` parametreleri altına `resize_interpolation` (select: `cubic`, `lanczos`) eklenecektir.
  * `threshold` parametreleri altına `threshold_method` (select: `adaptive`, `otsu`, `sauvola`, `niblack`) seçeneği ve Sauvola/Niblack için `threshold_k` ile `threshold_k_niblack` float alanları eklenecektir.
  * Bu sayede, frontend arayüzündeki konfigürasyon oluşturma formu otomatik olarak bu yeni parametreleri destekleyecek ve kullanıcı yeni ön işlemleri UI'dan kaydedebilecektir.

---

### 3. Tesseract Auto-Tuner Geliştirilmesi

#### [NEW] [tune_tesseract.py](file:///c:/Users/maraz/Desktop/ocr/tune_tesseract.py)
* Proje kök dizininde çalışacak bir `tune_tesseract.py` script'i oluşturulacaktır.
* **Çalışma Mantığı:**
  1. `inputs/truths/` altındaki tüm ground-truth (ehliyet/dekont) YAML dosyalarını ve bunlarla eşleşen `inputs/images/` altındaki görselleri yükler.
  2. Belirlenen parametre uzayında (Tesseract PSM, binarizasyon türleri, resize çarpanları ve interpolasyon tipleri, CLAHE vb.) arama yapar.
  3. Tüm kombinasyonları verimli bir şekilde tarayabilmek için **aşamalı ızgara araması (Progressive Grid Search)** veya tam arama yaparak her kombinasyonun görüntüler üzerindeki **ortalama CER (Karakter Hata Oranı) ve WER (Kelime Hata Oranı)** değerlerini hesaplar.
  4. Performans ve doğruluk analizi sonuçlarına göre tüm parametre kombinasyonlarını en düşük ortalama CER değerine göre sıralar.
  5. En iyi sonuç veren **ilk 3 konfigürasyonu** sırasıyla `tuner_best_1.yaml`, `tuner_best_2.yaml` ve `tuner_best_3.yaml` isimleriyle `configurations/tesseract/` altına kaydeder.
  6. Ekrana en iyi 3 modelin tüm parametrelerini ve ortalama doğruluk/süre metriklerini gösteren bir özet tablosu yazdırır.

---

## Doğrulama Planı

### Otomatik Testler
* `tune_tesseract.py` çalıştırılarak hata almadan en iyi 3 konfigürasyon dosyasını (`tuner_best_1.yaml`, `tuner_best_2.yaml`, `tuner_best_3.yaml`) ürettiği doğrulanacaktır:
  ```powershell
  .\venv\Scripts\python tune_tesseract.py
  ```
* Üretilen bu 3 yeni konfigürasyon ile `main.py` çalıştırılarak (ör. `process_pipeline("tesseract", "tuner_best_1")`) sistemin sorunsuz çalıştığı ve doğruluk oranlarının eski baseline (`model_v1`) ile karşılaştırmalı olarak sisteme kaydedildiği doğrulanacaktır:
  ```powershell
  .\venv\Scripts\python main.py
  ```

### Manuel Doğrulama
* FastAPI sunucusu ayakta iken tarayıcıdan `/config-schema/tesseract` endpoint'ine istek atılarak yeni Sauvola/Otsu/Lanczos parametre şemalarının döndüğü ve React arayüzündeki "Yeni Konfigürasyon" modalında düzgün göründüğü teyit edilecektir.
* Yeni oluşturulan Sauvola/Otsu ön işlemleriyle arayüzden tekil belge taranarak görselleştirme çıktılarının (preprocessed görseli) kalitesi görsel olarak kontrol edilecektir.
