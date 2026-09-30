FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# Cài đặt các thư viện hệ thống cần thiết cho OpenCV (cv2)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Đặt thư mục làm việc trong container
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Đổi hiếm trước, đổi thường xuyên sau -> sửa src/ không phải copy lại checkpoint 1.4GB
COPY checkpoint/ checkpoint/
COPY beit3_src/ beit3_src/
COPY configs/ configs/
COPY cli.py .
COPY README.md .
COPY src/ src/

VOLUME ["/app/dataset", "/app/benchmark"]

ENTRYPOINT ["python", "cli.py"]
CMD ["--help"]
