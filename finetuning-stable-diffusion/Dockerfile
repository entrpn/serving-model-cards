FROM us-docker.pkg.dev/vertex-ai/training/tf-tpu-pod-base-cp38:latest

RUN wget https://storage.googleapis.com/cloud-tpu-tpuvm-artifacts/libtpu/1.2.0/libtpu.so -O /lib/libtpu.so
RUN chmod 777 /lib/libtpu.so

RUN apt-get update && apt-get install -y curl
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | \
    tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
    tee /usr/share/keyrings/cloud.google.gpg && apt-get update -y && apt-get install google-cloud-sdk -y


RUN pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
RUN pip install tensorflow==2.9.2
RUN pip install transformers diffusers flax optax torch torchvision ftfy tensorboard modelcards datasets

# Like nvidia-smi for TPUs.
RUN apt-get update && \
    apt-get install -y golang
RUN pip install jax-smi

WORKDIR 'finetuning_sd'

COPY . .