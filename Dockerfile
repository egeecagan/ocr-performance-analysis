# =============================================================================
# PaddleOCR Container — sadece PaddleOCR motoru için
# =============================================================================
# Neden bu container var:
# Mac'te (özellikle Apple Silicon) paddlepaddle'ın 'paddle_static' motoru
# desteklenmiyor / kararsız çalışıyor. Bu Dockerfile, Linux tabanlı bir
# ortamda PaddleOCR'ı çalıştırıp sonuçları Mac'teki proje klasörüyle
# (volume mount ile) paylaşmak için var. Diğer 5 motor (Tesseract, EasyOCR,
# doctr, TrOCR, RapidOCR) Mac'inizde doğrudan çalışmaya devam ediyor —
# SADECE PaddleOCR bu container içinde çalışır.
# =============================================================================

FROM python:3.11-slim

WORKDIR /app

# PaddleOCR/OpenCV'nin ihtiyaç duyduğu sistem kütüphaneleri (libgl1, glib
# gibi) — bunlar olmadan cv2 import hatası alırsınız.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Python bağımlılıkları — sadece PaddleOCR runner'ının ihtiyaç duyduğu
# paketler. paddlepaddle BURADA (Linux/container) sorunsuz kurulur.
RUN pip install --no-cache-dir \
    paddleocr \
    paddlepaddle \
    opencv-python-headless \
    numpy \
    pillow \
    pyyaml

# Sadece ihtiyaç duyulan kod kopyalanıyor (tüm proje değil) — runners/
# klasöründeki _common.py, run_paddleocr.py ve registry.py'nin
# import ettiği load_config fonksiyonu için gerekli.
COPY runners/_common.py /app/runners/_common.py
COPY runners/run_paddleocr.py /app/runners/run_paddleocr.py
COPY runners/__init__.py /app/runners/__init__.py
COPY docker_run_paddleocr.py /app/docker_run_paddleocr.py

ENTRYPOINT ["python", "docker_run_paddleocr.py"]