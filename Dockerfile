############################
# 1) BUILDER STAGE
############################
FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        wget curl ca-certificates gnupg \
        build-essential git \
        libssl-dev libffi-dev \
        zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
        python3.11 python3.11-venv python3.11-dev \
        ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv /venv
ENV PATH="/venv/bin:$PATH"

RUN pip install --upgrade pip setuptools wheel
RUN pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126

WORKDIR /app
COPY extensions/ /app/extensions/

#Defined again in runtime stage
ENV TORCH_CUDA_ARCH_LIST="8.9"

RUN pip install --no-build-isolation ./extensions/chamfer_dist
#emd is never used, loss is chamfer distance but could be tested as loss
# RUN pip install ./extensions/emd 
RUN pip install --no-build-isolation ./extensions/Pointnet2_PyTorch/pointnet2_ops_lib
RUN pip install --no-build-isolation ./extensions/KNN_CUDA

############################
# 2) RUNTIME STAGE
############################
FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    libx11-6 libxcb1 libxext6 libgl1 libgl1-mesa-glx \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /venv /venv
ENV PATH="/venv/bin:$PATH"

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-deps -r /app/requirements.txt

COPY extensions/ /app/extensions/

ENV TORCH_CUDA_ARCH_LIST="8.9"
CMD ["bash"]
