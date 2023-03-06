from google.cloud import storage
import os

bucket_name = os.getenv("BUCKET_ID",None)
prefix = os.getenv("PREFIX",None)
client = storage.Client()
iterator = client.list_blobs(bucket_name,prefix=prefix)
with open("dataset.txt","w") as f:
    for blob in iterator:
        if blob.name.endswith((".png",".jpg",".jpeg")):
            f.write(f"gs://{bucket_name}/{blob.name}\n")