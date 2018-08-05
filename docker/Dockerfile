FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
         build-essential \
         git \
         curl \
         vim \
         ca-certificates && \
     rm -rf /var/lib/apt/lists/*

RUN curl -o miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
     bash miniconda.sh -b -p /opt/conda && \
     rm miniconda.sh
ENV PATH /opt/conda/bin:$PATH

# install dependencies
COPY azalea/environment.yml environment.yml
RUN conda update -n base conda && \
    conda env update -n base -f environment.yml && \
    conda clean -ya && \
    rm environment.yml

COPY azalea /azalea
ENV PYTHONPATH /azalea

# needed for click to work under python3
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

WORKDIR /work
#CMD ["python", "/azalea/train.py"]
