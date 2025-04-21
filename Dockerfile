# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Install Python and essential build tools
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install base requirements first (heavy ML libraries)
COPY requirements_base.txt .
RUN pip3 install -r requirements_base.txt

# Install app-specific requirements
COPY requirements_more.txt .
RUN pip3 install -r requirements_more.txt

# Install python-dotenv
RUN pip3 install python-dotenv

# Copy the source code and data files
COPY src/ ./src/

# Create directories for outputs
RUN mkdir -p /app/models /app/output /app/plots

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MODEL_CHECKPOINT_DIR=/app/models
ENV OUTPUT_DIR=/app/output
ENV PLOTS_DIR=/app/plots

# HuggingFace token will be passed at runtime
ARG HUGGINGFACE_TOKEN
ENV HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN}

# Set the default command to python
CMD ["python3"]

# Build command:
# docker build --build-arg HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN} -t maia-project .

# Run command:
# docker run --gpus all -v ${PWD}/src:/app/src -v ${PWD}/models:/app/models -v ${PWD}/output:/app/output -v ${PWD}/plots:/app/plots -e HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN} maia-project src/lstm_model.py