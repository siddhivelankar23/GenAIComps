
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

FROM vault.habana.ai/gaudi-docker/1.16.1/ubuntu22.04/habanalabs/pytorch-installer-2.2.2:latest

ENV LANG=C.UTF-8

RUN apt-get update -y && apt-get install -y --no-install-recommends --fix-missing \
    build-essential \
    libgl1-mesa-glx \
    libjemalloc-dev \
    default-jre \
    vim

# Install ffmpeg static build
RUN cd /root && wget https://johnvansickle.com/ffmpeg/builds/ffmpeg-git-amd64-static.tar.xz && \
    mkdir ffmpeg-git-amd64-static && tar -xvf ffmpeg-git-amd64-static.tar.xz -C ffmpeg-git-amd64-static --strip-components 1 && \
    export PATH=/root/ffmpeg-git-amd64-static:$PATH && \
    cp /root/ffmpeg-git-amd64-static/ffmpeg /usr/local/bin/

RUN useradd -m -s /bin/bash user && \
    mkdir -p /home/user && \
    chown -R user /home/user/

USER user

COPY comps /home/user/comps

RUN pip install --no-cache-dir --upgrade pip setuptools && \
    pip install --no-cache-dir -r /home/user/comps/dataprep/redis/langchain_multimodal/requirements.txt

ENV PYTHONPATH=$PYTHONPATH:/home/user

USER root

RUN mkdir -p /home/user/comps/dataprep/redis/langchain_multimodal/uploaded_files && chown -R user /home/user/comps/dataprep/redis/langchain_multimodal/uploaded_files

USER user

WORKDIR /home/user/comps/dataprep/redis/langchain_multimodal

ENTRYPOINT ["python", "prepare_doc_redis.py"]

