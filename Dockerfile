FROM python:3.10-bookworm

# Install system packages
RUN DEBIAN_FRONTEND=noninteractive apt update && apt install -y \
	ffmpeg 


# Create non-root user
RUN useradd -m -d /bark bark 

# Copy project files
COPY ./ /bark/bark-gui

# Correct permission for non-root user
RUN chown bark:bark -R /bark/bark-gui

# Run as new user
USER bark
WORKDIR /bark/bark-gui

# Append pip bin path to PATH
ENV PATH=$PATH:/bark/.local/bin

# Install dependancies
RUN pip install --no-cache-dir . &&\
	pip install --no-cache-dir -r requirements.txt

# List on all addresses, since we are in a container.
RUN sed -i "s/server_name: ''/server_name: 0.0.0.0/g" ./config.yaml

# Suggested volumes
VOLUME /bark/bark-gui/assets/prompts/custom
VOLUME /bark/bark-gui/models
VOLUME /bark/.cache/huggingface/hub

# Default port for web-ui
EXPOSE 7860/tcp

# Start script
CMD python3 webui.py
