FROM us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-11:latest

COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR 'image_segmentation'

COPY main.py .

ENTRYPOINT ["python","main.py"]