import os
from os.path import exists

import gradio as gr
import json
from PIL import Image
import io

from google.cloud import storage

from utils import gcs_utils
from css_and_js import call_js
from constants import LOCAL_SAVE_PATH

storage_client = storage.Client()

global_metadata_contents = []

batch_job_entries = []

LOCAL_METADATA_PATH=f'{LOCAL_SAVE_PATH}/metadata.jsonl'

def update_batch_job_entry_n_cache(prompt, neg_prompt, model_id, scheduler, scale, num_images, seed):
    entry = {
        "prompt" : prompt,
        "neg_prompt" : neg_prompt,
        "model_id" : model_id,
        "scheduler" : scheduler,
        "scale" : scale,
        "num_images" : num_images,
        "seed" : seed if seed != -1 else None
        }
    with open(LOCAL_METADATA_PATH,'a+') as f:
        f.write(f'{json.dumps(entry)}\n')
    batch_job_entries.append([prompt, neg_prompt, model_id, scheduler, scale, num_images, seed])

def update_batch_job_dataframe(x):
    print(x)
    batch_job_entries.clear()
    os.remove(LOCAL_METADATA_PATH)
    for _, row in x.iterrows():
        prompt = row['prompt']
        neg_prompt = row['negative_prompt']
        model_id = row['model_id']
        scheduler = row['scheduler']
        scale = row['scale']
        num_images = row['num. images']
        seed = row['seed']
        if num_images == 0:
            continue
        update_batch_job_entry_n_cache(prompt,
            neg_prompt, model_id, scheduler, scale, num_images, seed)
       

def add_batch_job_entry(prompt, neg_prompt, model_id,
            scheduler, scale, num_images, seed):
    seed = int(seed)
    scale = float(scale)
    num_images = int(num_images)
    update_batch_job_entry_n_cache(prompt,
            neg_prompt, model_id, scheduler, scale, num_images, seed)
    
    return batch_job_entries

def load_image_params(x):
    idx = x[1]
    selected = global_metadata_contents[idx]
    return (selected.get('prompt'), 
            selected.get('negative_prompt',''),
            selected.get('scale',7.5),
            selected.get('num_inference_steps',50),
            selected.get('scheduler','DDIMScheduler'),
            selected.get('model_id',''),
            selected.get('seed',-1),
            selected.get('image_uris',''))

def cache_and_return_images(image_uris):
    retval = []
    for image_uri in image_uris:
        image_filename = image_uri.split('/')[-1]
        local_image_uri = f'/tmp/{image_filename}'
        if exists(local_image_uri):
            retval.append(Image.open(local_image_uri))
        else:
            if gcs_utils.get_blob(image_uri).exists():
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
                    metadata_content = json.loads(l)
                    metadata_content['image_uris'] = image_uri
                    global_metadata_contents.append(metadata_content)
                tmp.extend(cache_and_return_images(image_uris))
            metadata_contents = tmp
            print(metadata_contents)
    else:
        gr.Error("No file found")
    
    return metadata_contents

def view_creations():
    with gr.Blocks():
        gr.Markdown("""View your creations in this tab. If you haven't yet, you can create a job below.""")
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
                creations_image_uri = gr.Textbox(label='Image uri')
            with gr.Column():
                creations_gallery = gr.Gallery(label='Images', elem_id='creations_gallery').style(grid=[4, 4])
                view_params_btn = gr.Button('View Parameters')
        
        with gr.Row():
            with gr.Accordion("Create a job"):
                gr.Markdown("""
                You can create a batch job with your favorite stable diffusion models here! 

                This uses GCP resources as follows:
                - Cloud build to build a docker image that will be used to run a training job.
                - Vertex AI to create a custom job where all predictions will be made.
                - Cloud storage to save all generated images and metadata file.
                """)
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("Project parameters")
                        batch_job_project_id = gr.Textbox(label='Enter your project id')
                        batch_job_region = gr.Textbox('us-central1', label='Enter your region')
                        batch_job_accelerator_type = gr.Dropdown(label='Accelerator Type', value='NVIDIA_TESLA_T4',
                            choices=['NVIDIA_TESLA_T4','NVIDIA_TESLA_P4','NVIDIA_TESLA_K80', 'NVIDIA_TESLA_V100', 'NVIDIA_TESLA_A100', 'NVIDIA_A100_80GB'])
                        batch_job_gcs_folder = gr.Textbox(label='Enter the gcs folder where results are stored (Ex: gs://bucket-name/sd-predictions/)')
                        batch_job_hugginface_token = gr.Textbox(label='Enter your huggingFace token (Needed to access diffusion models)')
                    with gr.Column():
                        gr.Markdown("Fill in the fields and press Add Entry")
                        batch_job_prompt = gr.Textbox(label="Prompt (Required)")
                        batch_job_neg_prompt = gr.Textbox(label="Negative prompt")
                        batch_job_model_id = gr.Textbox("runwayml/stable-diffusion-v1-5", label="Model id")
                        batch_job_scheduler = gr.Dropdown(['DDIMScheduler','DPMSolverMultistepScheduler', 'EulerAncestralDiscreteScheduler',
                            'EulerDiscreteScheduler','LMSDiscreteScheduler','PNDMScheduler'], value='DDIMScheduler', label="Scheduler")
                        batch_job_scale = gr.Textbox("7.5", label="Scale")
                        batch_job_num_images = gr.Textbox("4", label="Number of images to generate (batch). If too large, the GPU might run out of memory.")
                        batch_job_seed = gr.Textbox("-1", label="Seed (defaults to random)")
                        add_batch_job_entry_btn = gr.Button('Add entry')
                with gr.Row():
                    batch_job_dataframe = gr.Dataframe(headers=['prompt', 'negative_prompt','model_id','scheduler','scale','num. images', 'seed'],
                        datatype=["str", "str", "str", "str","number", "number", "number"], wrap=True, row_count=[1,'dynamic'], col_count=[7,'fixed'])
                    batch_job_dataframe.change(update_batch_job_dataframe,
                    batch_job_dataframe,
                    None
                    )
                submit_batch_prediction_job_btn = gr.Button('Submit job')

        load_metadata_file_btn.click(
                load,
                [metadata_file_uri],
                outputs=creations_gallery
            )
        view_params_btn.click(
            load_image_params,
            creations_gallery,
            [creations_prompt, creations_neg_prompt, creations_scale, 
            creations_num_inference_steps, creations_scheduler, creations_model_id,
            creations_seed, creations_image_uri],
            _js=call_js("getGallerySelectedItem",element_id="creations_gallery")
        )

        add_batch_job_entry_btn.click(
            add_batch_job_entry,
            [batch_job_prompt, batch_job_neg_prompt, batch_job_model_id,
            batch_job_scheduler, batch_job_scale, batch_job_num_images, batch_job_seed],
            batch_job_dataframe
        )

        submit_batch_prediction_job_btn.click(
            lambda x:x,
            [batch_job_dataframe],
            None
        )