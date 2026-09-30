# Image OCR production: DeepSolo detection + PARSeq-VN recognition, CUDA 11.8.
#   docker build -t ocr-deepsolo-parseq layer_3/OCR
# GPU stack giống layer_1: base TF 2.13 (CUDA 11.8 + nvcc) + torch cu118.
FROM tensorflow/tensorflow:2.13.0-gpu

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# base image không kèm git (cần để clone detectron2/DeepSolo) và make (để build ext).
# Tag tensorflow:2.13.0-gpu bị re-push: bản mới bỏ hết header dev của thư viện
# CUDA (cublas/cusparse/... bản 24/9 vẫn có) nên cài thêm từ repo NVIDIA đã có
# sẵn trong image, khớp đúng 11.8.
RUN apt-get update && apt-get install -y --no-install-recommends git make \
        libcublas-dev-11-8 libcusparse-dev-11-8 libcusolver-dev-11-8 \
        libcurand-dev-11-8 libcufft-dev-11-8 cuda-nvrtc-dev-11-8 cuda-profiler-api-11-8 \
    && rm -rf /var/lib/apt/lists/*

# Build không có GPU nên torch tự dò arch được -> arch_list rỗng và crash.
# 7.5 (Turing, máy dev) + PTX để JIT trên GPU mới hơn.
ENV TORCH_CUDA_ARCH_LIST="7.5+PTX"

# torch cu118 phải cài trước detectron2/DeepSolo vì chúng build theo version torch này
RUN pip install --no-cache-dir \
        torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
        --index-url https://download.pytorch.org/whl/cu118

COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# detectron2: PyPI chỉ có wheel CPU. Tag v0.6 (2022) còn dùng Image.LINEAR mà
# Pillow 10 đã bỏ -- pin main @1e3e13b (có fix, version string vẫn '0.6' như
# bản production). Build không GPU nên phải ép arch, không thì arch_list rỗng
# và torch/cpp_extension crash; --no-deps để requirements.txt giữ các bản pin.
RUN git init /tmp/detectron2 \
    && cd /tmp/detectron2 \
    && git remote add origin https://github.com/facebookresearch/detectron2.git \
    && git fetch --depth 1 origin 1e3e13bbf607b54f62205c4c33922521822fb298 \
    && git checkout FETCH_HEAD \
    && FORCE_CUDA=1 MAX_JOBS=8 pip install --no-cache-dir --no-build-isolation --no-deps . \
    && rm -rf /tmp/detectron2

# DeepSolo (adet) @dbadae9 — build CUDA ext adet._C. --no-deps: Polygon3 của
# upstream không có trong ảnh production và adet không import nó.
RUN git init /tmp/deepsolo \
    && cd /tmp/deepsolo \
    && git remote add origin https://github.com/ViTAE-Transformer/DeepSolo.git \
    && git fetch --depth 1 origin dbadae995035246bad3376c7a44c015c69e9b313 \
    && git checkout FETCH_HEAD \
    && cp -r DeepSolo /workspace/DeepSolo \
    && cd /workspace/DeepSolo \
    && FORCE_CUDA=1 python setup.py build develop --no-deps \
    && rm -rf /tmp/deepsolo

# strhub (PARSeq-VN) vendored trong repo, đã vá cho Python 3.8
COPY parseq/ /parseq/

ENV PYTHONPATH=/workspace/DeepSolo:/parseq

# cv2 (cả headless 5.x) vẫn link libGL/libxcb; tag base bị re-push đã bỏ
# các libs này (bản 24/9 còn kèm) -- cài lại
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
