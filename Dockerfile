FROM nvcr.io/nvidia/pytorch:21.12-py3

RUN set -x && \
    echo "Acquire { HTTP { Proxy \"$HTTP_PROXY\"; }; };" | tee /etc/apt/apt.conf

ENV LC_ALL=C.UTF-8 LANG=C.UTF-8

RUN apt-get update || true && apt-get install -y build-essential \
    wget curl git git-lfs vim zip unzip tmux htop

RUN apt-get install -y libaio1 libaio-dev

WORKDIR /workspace

RUN pip install triton==1.0.0

RUN DS_BUILD_QUANTIZER=1 DS_BUILD_TRANSFORMER_INFERENCE=1 DS_BUILD_TRANSFORMER=1 DS_BUILD_CPU_ADAM=1 DS_BUILD_FUSED_ADAM=1 DS_BUILD_FUSED_LAMB=1 DS_BUILD_SPARSE_ATTN=1 DS_BUILD_UTILS=1 DS_BUILD_AIO=1  pip install deepspeed --global-option="build_ext" --global-option="-j8"
COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('wordnet_ic'); nltk.download('sentiwordnet'); nltk.download('omw-1.4')"
CMD sleep infinity