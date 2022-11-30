from os.path import exists

import gradio as gr
import json
from PIL import Image
import io

from google.cloud import storage

from utils import gcs_utils
from css_and_js import call_js

storage_client = storage.Client()

global_metadata_contents = []

def tmp(x):
    idx = x[1]
    selected = global_metadata_contents[idx]
    return (selected.get('prompt'), 
            selected.get('negative_prompt',''),
            selected.get('scale',7.5),
            selected.get('num_inference_steps',50),
            selected.get('scheduler','DDIMScheduler'),
            selected.get('model_id',''),
            selected.get('seed',-1))

def cache_and_return_images(image_uris):
    retval = []
    for image_uri in image_uris:
        image_filename = image_uri.split('/')[-1]
        local_image_uri = f'/tmp/{image_filename}'
        if exists(local_image_uri):
            retval.append(Image.open(local_image_uri))
        else:
            tmp_img = gcs_utils.read_image(gcs_utils.get_blob(image_uri))
            tmp_img.save(local_image_uri)
            retval.append(tmp_img)
    return retval

def load(metadata_file_uri):
    bucket_name = metadata_file_uri.replace("gs://",'').split('/')[0]
    metadata_uri = metadata_file_uri.replace(f"gs://{bucket_name}/",'')
    print(metadata_uri)
    bucket = storage_client.bucket(bucket_name)
    metadata_blob = bucket.blob(metadata_uri)
    if metadata_blob.exists():
        with metadata_blob.open("r") as f:
            metadata_contents = f.readlines()
            tmp = []
            for l in metadata_contents:
                metadata_content = json.loads(l)
                image_uris = metadata_content['image_uris']
                for image_uri in image_uris:
                    metadata_content['image_uris'] = image_uri
                    global_metadata_contents.append(metadata_content)
                tmp.extend(cache_and_return_images(image_uris))
                # for image in image_uris:
                #     tmp.append(gcs_utils.read_image(gcs_utils.get_blob(image)))
            metadata_contents = tmp
            print(metadata_contents)
    else:
        gr.Error("No file found")
    
    return metadata_contents

def view_creations():
    with gr.Blocks():
        gr.Markdown("""View your creations in this tab. If you haven't yet, create a job through the Batch predict tab.""")
        with gr.Row():
            metadata_file_uri = gr.Textbox(label='Metadata uri (Ex: gs://bucket-name/sd-predictions/results.jsonl)')
            load_metadata_file_btn = gr.Button("Load")
        with gr.Row():
            creations_prompt = gr.Textbox(label='Prompt')
            creations_neg_prompt = gr.Textbox(label='Negative prompt')
        with gr.Row():
            with gr.Column():
                creations_model_id = gr.Textbox(label='Model Id')
                creations_scheduler = gr.Textbox(label='Scheduler')
                creations_num_inference_steps = gr.Textbox(label='No. Inference Steps')
                creations_scale = gr.Textbox(label='Scale')
                creations_seed = gr.Textbox(label='Seed (-1 is random)')
            with gr.Column():
                creations_gallery = gr.Gallery(label='Images', elem_id='creations_gallery').style(grid=[4, 4])
                view_params_btn = gr.Button('View Parameters!')
        load_metadata_file_btn.click(
                load,
                [metadata_file_uri],
                outputs=creations_gallery
            )
        view_params_btn.click(
            tmp,
            creations_gallery,
            [creations_prompt, creations_neg_prompt, creations_scale, 
            creations_num_inference_steps, creations_scheduler, creations_model_id,
            creations_seed],
            _js=call_js("getGallerySelectedItem",element_id="creations_gallery")
        )