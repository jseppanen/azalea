FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
         build-essential \
         git \
         curl \
         vim \
         ca-certificates && \
     rm -rf /var/lib/apt/lists/*

# install base environment
COPY environment.yml environment.yml
RUN curl -o miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash miniconda.sh -b -p /opt/conda \
    && /opt/conda/bin/conda update -n base -c defaults conda \
    && /opt/conda/bin/conda env update -n base -f environment.yml \
    && /opt/conda/bin/conda clean -ya \
    && rm miniconda.sh environment.yml
ENV PATH /opt/conda/bin:$PATH

# install azalea
ARG PACKAGE
COPY $PACKAGE .
RUN pip install --no-cache $PACKAGE \
    && rm -f $PACKAGE

# needed for click to work under python3
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

RUN mkdir /work
WORKDIR /work
