# FROM nvidia/cuda:11.6.2-base-ubuntu20.04
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV LD_LIBRARY_PATH=/usr/local/cuda/efa/lib:/usr/local/cuda/lib:/usr/local/cuda/lib64:/usr/local/cuda:/usr/lib/x86_64-linux-gnu

RUN apt update && apt install -qqy python3 python3-virtualenv python3-dev \
    build-essential python3-pip git vim nano wget curl git-lfs ffmpeg

# Create non-root user
RUN useradd -m -d /bark bark

# Run as new user
USER bark
WORKDIR /bark

# Clone git repo
RUN git clone https://github.com/C0untFloyd/bark-gui 

# Switch to git directory
WORKDIR /bark/bark-gui

# Append pip bin path to PATH
ENV PATH=$PATH:/bark/.local/bin

# Install dependancies
RUN pip install .
RUN pip install -r requirements.txt

# List on all addresses, since we are in a container.
RUN sed -i "s/server_name: ''/server_name: 0.0.0.0/g" ./config.yaml

# Suggested volumes
RUN mkdir -p /bark/bark-gui/assets/prompts/custom
RUN mkdir -p /bark/bark-gui/models
RUN mkdir -p /bark/.cache/huggingface/hub

VOLUME /bark/bark-gui/assets/prompts/custom
VOLUME /bark/bark-gui/models
VOLUME /bark/.cache/huggingface/hub

# Default port for web-ui
EXPOSE 7860/tcp

# Start script
CMD python3 webui.py
