# Layer 3 (GPU): Image Embedding — SigLIP qua package keyframe_pipeline.
# Base: PyTorch GPU image, CUDA 11.8.
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

COPY pyproject.toml /workspace/pyproject.toml
COPY keyframe_pipeline /workspace/keyframe_pipeline
RUN pip install --no-cache-dir .

ENTRYPOINT ["python3", "-m", "keyframe_pipeline"]
