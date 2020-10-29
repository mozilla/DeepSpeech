FROM tensorflow/tensorflow:1.15.2-py3

ARG DEEPSPEECH_CONTAINER_DIR=/opt/deepspeech
ARG DEEPSPEECH_VERSION=0.8.2

# Install OS dependencies
RUN apt-get update && \
    apt-get install --no-install-recommends -y wget ffmpeg && \
    apt-get clean

# Create app directory
RUN mkdir -p ${DEEPSPEECH_CONTAINER_DIR}

# Get pre-trained model
RUN wget -q "https://github.com/mozilla/DeepSpeech/releases/download/v${DEEPSPEECH_VERSION}/deepspeech-${DEEPSPEECH_VERSION}-models.pbmm" \
         -O ${DEEPSPEECH_CONTAINER_DIR}/model.pbmm
RUN wget -q "https://github.com/mozilla/DeepSpeech/releases/download/v${DEEPSPEECH_VERSION}/deepspeech-${DEEPSPEECH_VERSION}-models.scorer" \
         -O ${DEEPSPEECH_CONTAINER_DIR}/scorer.scorer

# Install Python dependencies
RUN pip3 install --upgrade pip

COPY requirements.txt /tmp
RUN pip3 install -r /tmp/requirements.txt

# Copy code and configs
COPY deepspeech_server ${DEEPSPEECH_CONTAINER_DIR}/deepspeech_server
COPY application.conf ${DEEPSPEECH_CONTAINER_DIR}

WORKDIR ${DEEPSPEECH_CONTAINER_DIR}

ENTRYPOINT python -m deepspeech_server.app
