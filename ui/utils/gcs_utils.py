from PIL import Image
import io

from google.cloud import storage

def get_blob(gcs_uri):
    storage_client = storage.Client()
    bucket_name = gcs_uri.replace("gs://",'').split('/')[0]
    blob_uri = gcs_uri.replace(f"gs://{bucket_name}/",'')
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_uri)
    return blob

def read_image(blob):
    return Image.open(io.BytesIO(blob.download_as_string()))