FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    paddleocr \
    paddlepaddle \
    opencv-python-headless \
    numpy \
    pillow \
    pyyaml

COPY runners/_common.py /app/runners/_common.py
COPY runners/run_paddleocr.py /app/runners/run_paddleocr.py
COPY runners/__init__.py /app/runners/__init__.py
COPY docker_run_paddleocr.py /app/docker_run_paddleocr.py

ENTRYPOINT ["python", "docker_run_paddleocr.py"]