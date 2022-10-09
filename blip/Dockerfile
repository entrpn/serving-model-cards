FROM nvidia/cuda:11.3.1-base-ubuntu20.04

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt install -y python3.8 && \
    apt install -y python3-pip

RUN apt-get update && apt-get -y upgrade \
  && apt-get install -y --no-install-recommends \
    git \
    wget \
    g++ \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/salesforce/BLIP.git
WORKDIR "BLIP"
RUN pip install -r requirements.txt
RUN pip install uvicorn fastapi

COPY app/load_weights.py .
COPY load_weights.sh .
RUN ["./load_weights.sh"]

COPY app .
COPY entrypoint.sh .

EXPOSE 8080
ENTRYPOINT ["./entrypoint.sh"]