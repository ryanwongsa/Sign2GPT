FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update

RUN apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    rsync \
    wget \
    ffmpeg \
    htop \
    nano \ 
    libatlas-base-dev \
    libboost-all-dev \
    libeigen3-dev \
    libhdf5-serial-dev \
    libleveldb-dev \
    liblmdb-dev \
    libopenblas-dev \
    libopenblas-base \
    libsm6 \
    libxext6 \
    libxrender-dev

RUN apt-get autoremove -y && \
    apt-get autoclean -y && \
    apt-get clean -y  && \
    rm -rf /var/lib/apt/lists/*

ENV WRKSPCE="/workspace"

RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p $WRKSPCE/miniconda3 \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

ENV PATH="$WRKSPCE/miniconda3/bin:${PATH}"
# ENV TORCH_CUDA_ARCH_LIST="8.0"
COPY environment.yml .

RUN conda install -n base python=3.9

RUN conda env update -n base --file environment.yml
RUN conda install -y -c fvcore -c iopath -c conda-forge fvcore iopath
RUN conda install -y -c bottler nvidiacub

RUN conda install -y pytorch=2.3.0 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

RUN pip install --upgrade setuptools pip wheel

RUN conda install -y xformers==0.0.26.post1 -c xformers

RUN conda update -y conda

RUN conda install -y av==8.1.0 -c conda-forge

RUN pip install numpy==1.26.4

RUN apt-get update && apt-get -y install --no-install-recommends g++ && \
    pip uninstall -y pillow && \
    CC="cc -mavx2" pip install --upgrade --no-cache-dir --force-reinstall pillow-simd && \
    apt-get remove -y g++ && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

RUN pip install tqdm

RUN mkdir /.cache /.config && \
    chmod 777 /.cache /.config && \
    chmod -R 777 /.cache /.config

RUN chmod -R 777 /workspace/miniconda3/bin/python
RUN chmod -R 777 /workspace/miniconda3/lib/python3.9
