# Use Ubuntu + CUDA base
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev libcudnn8-dev libcudnn8 \
    git wget curl ffmpeg libsndfile1 build-essential cmake libglib2.0-0 \ 
 && rm -rf /var/lib/apt/lists/*

# Python deps
RUN pip3 install --upgrade pip
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
RUN pip install --upgrade "cuda-python>=12.3,<13"
RUN pip3 install \
    librosa tqdm filetype imageio opencv-python-headless scikit-image cython \
    imageio-ffmpeg colored polygraphy onnxruntime-gpu tensorrt==8.6.1 \
    fastapi uvicorn runpod==1.4.0

# ------------------------------------------------------------
# 6. Copy source files 
# ------------------------------------------------------------

COPY ./core/ /app/core/
COPY ./inference.py /app/inference.py
COPY ./runpod_handler.py /app/runpod_handler.py
COPY ./stream_pipeline_offline.py /app/stream_pipeline_offline.py
COPY ./stream_pipeline_online.py /app/stream_pipeline_online.py
COPY ./static/ /app/static/
COPY ./example/ /app/example/
COPY ./scripts/ /app/scripts/

# Create outputs directory for generated videos
RUN mkdir -p /app/outputs

# ------------------------------------------------------------
# 7. Expose API port and define entrypoint
# ------------------------------------------------------------
EXPOSE 8000

CMD ["python3", "runpod_handler.py"]
