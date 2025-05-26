FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3.11-dev curl git build-essential \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && python3.11 -m pip install --upgrade pip setuptools wheel \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY req.txt .

RUN python3.11 -m pip install \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

RUN python3.11 -m pip install \
    sentence-transformers transformers accelerate scipy einops scikit-learn

RUN python3.11 -m pip install -r req.txt

WORKDIR /app

COPY . .

EXPOSE 8000

CMD ["python3.11", "start.py"]
