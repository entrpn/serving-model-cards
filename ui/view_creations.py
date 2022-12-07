import os
from os.path import exists
import io
import time

import gradio as gr
import json
from PIL import Image

from utils import gcs_utils, cloud_build_utils, vertexai_utils, cache_utils
from css_and_js import call_js
from constants import LOCAL_SAVE_PATH, MODELS, MODEL_DICT

global_metadata_contents = []

global_batch_job_entries = []

LOCAL_METADATA_PATH=f'{LOCAL_SAVE_PATH}/metadata.jsonl'
UI_VALS_CACHE=f'{LOCAL_SAVE_PATH}/creations_ui_vals.json'
global_ui_values = {}

global_batch_job_dataset_idx = [0]

# saves values from ui so they don't have to be loaded again.
def load_ui_values():
    if exists(UI_VALS_CACHE):
        with open(UI_VALS_CACHE, 'r') as f:
            global_ui_values.update(json.loads(f.read()))

def get_batch_job_status_str(build_id, build_status, job_id=None, job_state=None, results_blob_uri=None, job_error=None):
    retval = ""
    retval = f"Cloud build job launched : {build_id}\n"
    retval += f"Cloud build job status................{build_status}\n"
    
    if job_id:
        retval += f"Vertex AI job id : {job_id}\n"
    
    if job_state:
        retval += f"Vertex AI job status..............{job_state}\n"
    else:
        retval += f"Deploying Vertex AI job...........QUEUED\n"
    
    if results_blob_uri:
        retval += f"Results are located in {results_blob_uri}. Paste it in the Metadata uri textbox and click Load to view them."

    if job_error is not None:
        retval += f"Error: {job_error}"
    return retval

def submit_batch_job(project_id, region,
    accelerator_type, gcs_folder, hf_token, ):
    cache_utils.save_ui_values(UI_VALS_CACHE,project_id=project_id, region=region, gcs_folder=gcs_folder, hf_token=hf_token)
    metadata_blob_uri = os.path.join(gcs_folder,'metadata.jsonl')
    print(metadata_blob_uri)
    metadata_blob = gcs_utils.get_blob(metadata_blob_uri)
    metadata_blob.upload_from_filename(LOCAL_METADATA_PATH)
    build_id = None
    progress = ""
    job_id = None
    results_blob_uri = None
    for _ in range(1000):
        if build_id is None:
            build_id, image_uri = cloud_build_utils.create_build(project_id, metadata_blob_uri)
        build = cloud_build_utils.get_build(project_id, build_id)
        build_status = cloud_build_utils.get_status_str(build.status)
        
        progress = get_batch_job_status_str(build_id=build_id, build_status=build_status)
        
        if build_status == 'SUCCESS':
            job_error = None
            if job_id is None:
                job_id, job_error = vertexai_utils.create_custom_job_sample(project_id, 
                    region, 
                    'stable-diffusion-batch-job', 
                    accelerator_type, 
                    gcs_folder, 
                    hf_token, image_uri)
                if job_error is not None:
                    progress = get_batch_job_status_str(build_id=build_id, build_status=build_status, job_error=job_error)
                    yield """<center><h3 style="background-color:red;">Something went wrong</h1></center>""", progress
                    return
            else:
                job_status = vertexai_utils.get_custom_job_sample(project_id, job_id, region)
                job_state = vertexai_utils.get_job_state_str(job_status.state.value)
                progress = get_batch_job_status_str(build_id=build_id, build_status=build_status, job_state=job_state)
                if job_state == 'SUCCEEDED':
                    results_blob_uri = os.path.join(gcs_folder,'results.jsonl')
                    progress = get_batch_job_status_str(build_id=build_id, 
                                                        build_status=build_status, 
                                                        job_state=job_state,
                                                        results_blob_uri=results_blob_uri)
                    yield """<center><h3 style="background-color:green;">Job completed</h1></center>""", progress
                    return
                elif (job_state == 'FAILED'
                    or job_state == 'CANCELLED'
                    or job_state == 'EXPIRED'):
                    yield """<center><h3 style="background-color:red;">Something went wrong</h1></center>""", progress
                    return

        elif (build_status == 'FAILURE' 
            or build_status == 'INTERNAL_ERROR' 
            or build_status == 'TIMEOUT'
            or build_status == 'CANCELLED'
            or build_status == 'EXPIRED'):
            yield """<center><h3 style="background-color:red;">Something went wrong</h1></center>""", progress
            return
        
        yield """<center><h3 style="background-color:orange;">Deploying job</h1></center>""", progress
        time.sleep(30)
    # Create image with cloud build
    build_id = cloud_build_utils.create_build(project_id, metadata_blob_uri)
    # Launch job
    yield """<center><h3 style="background-color:orange;">Deploying job</h1></center>""", progress

def load_last_job_entry():
    if exists(LOCAL_METADATA_PATH):
        global_batch_job_entries.clear()
        with open(LOCAL_METADATA_PATH,'r') as f:
            entries = f.readlines()
            for entry in entries:
                entry = json.loads(entry)
                
                enhance = entry.get("enhance")
                if enhance is None:
                    enhance = {'model_id' : "None", 'face_enhance' : False, 'outscale' : 2}

                prompt = entry.get('prompt')
                if prompt is None or prompt == '':
                    continue
                
                negative_prompt = entry.get('negative_prompt','')
                model_id = entry.get('model_id')
                scheduler = entry.get('scheduler')
                scale = entry.get('scale',7.5)
                steps = entry.get('steps',50)
                num_images = entry.get('num_images')
                seed = entry.get('seed',-1)
                enhance_model_id = enhance.get('model_id','None')
                face_enhance = enhance.get('face_enhance')
                outscale = enhance.get('outscale')

                global_batch_job_entries.append([prompt, negative_prompt, model_id, 
                    scheduler, scale, steps, num_images, seed, enhance_model_id, face_enhance, outscale])
    return global_batch_job_entries

def update_batch_job_entry_n_cache(global_batch_job_entries):
    if exists(LOCAL_METADATA_PATH):
        os.remove(LOCAL_METADATA_PATH)
    
    for global_batch_job_entry in global_batch_job_entries:
        seed = global_batch_job_entry[7]

        enhance = None
        if global_batch_job_entry[8] != "None":
            enhance = {
                "model_id" : global_batch_job_entry[8], 
                "face_enhance" : global_batch_job_entry[9],
                "outscale" : global_batch_job_entry[10]
            }
        entry = {
            "prompt" : global_batch_job_entry[0],
            "negative_prompt" : global_batch_job_entry[1],
            "model_id" : global_batch_job_entry[2],
            "scheduler" : global_batch_job_entry[3],
            "scale" : global_batch_job_entry[4],
            "steps" : global_batch_job_entry[5],
            "num_images" : global_batch_job_entry[6],
            "seed" : seed if seed != -1 else None,
            "enhance" : enhance
        }
        with open(LOCAL_METADATA_PATH,'a+') as f:
            f.write(f'{json.dumps(entry)}\n')
    
    return global_batch_job_entries
    

def append_batch_job_entry_n_cache(prompt, neg_prompt, model_id, scheduler, 
    scale, steps, num_images, seed, enhanced_model, enhanced_face_restore, enhanced_scale):
    enhance = None
    if enhanced_model != "None":
        enhance = {
            "model_id" : enhanced_model, 
            "face_enhance" : enhanced_face_restore,
            "outscale" : enhanced_scale
        }
    entry = {
        "prompt" : prompt,
        "negative_prompt" : neg_prompt,
        "model_id" : model_id,
        "scheduler" : scheduler,
        "scale" : scale,
        "steps" : steps,
        "num_images" : num_images,
        "seed" : seed,
        "enhance" : enhance
        }
    with open(LOCAL_METADATA_PATH,'a+') as f:
        f.write(f'{json.dumps(entry)}\n')
    global_batch_job_entries.append([prompt, neg_prompt, model_id, scheduler, scale, 
        steps, num_images, seed, enhanced_model, enhanced_face_restore, enhanced_scale])
    return global_batch_job_entries


def delete_batch_job_entry():
    del global_batch_job_entries[global_batch_job_dataset_idx[0]]
    return update_batch_job_entry_n_cache(global_batch_job_entries)

def edit_batch_job_entry(prompt, neg_prompt, model_id,
            scheduler, scale, steps, num_images, seed,
            enhanced_model, enhanced_face_restore,
            enhanced_scale):
    if seed is not None:
        seed = int(seed)
    else:
        seed = -1
    scale = float(scale)
    num_images = int(num_images)
    global_batch_job_entries[global_batch_job_dataset_idx[0]] = [prompt, neg_prompt, model_id, 
        scheduler, scale, steps, num_images, seed, enhanced_model, enhanced_face_restore,
        enhanced_scale]
    return update_batch_job_entry_n_cache(global_batch_job_entries)

def add_batch_job_entry(prompt, neg_prompt, model_id,
            scheduler, scale, steps, num_images, seed,
            enhanced_model, enhanced_face_restore,
            enhanced_scale):
    error_message = ""
    if prompt is not None and len(prompt) > 0:
        if seed is not None:
            seed = int(seed)
        else:
            seed = -1
        scale = float(scale)
        num_images = int(num_images)
        print('seed:',seed)
        append_batch_job_entry_n_cache(prompt,
                neg_prompt, model_id, scheduler, 
                scale, steps, num_images, seed,
                enhanced_model, enhanced_face_restore, enhanced_scale)
    else:
        error_message = """<center><h3 style="background-color:red;">Prompt cannot be empty</h1></center>"""
    
    return global_batch_job_entries, error_message

def load_image_params(x):
    idx = x[1]
    selected = global_metadata_contents[idx]
    enhanced = selected.get('enhance',{'model_id' : "None", 'face_enhance' : False, 'outscale' : 1})
    return (selected.get('prompt'), 
            selected.get('negative_prompt',''),
            selected.get('scale',7.5),
            selected.get('num_inference_steps',50),
            selected.get('scheduler','DDIMScheduler'),
            selected.get('model_id',''),
            selected.get('seed',-1),
            selected.get('image_uris',''),
            enhanced.get('model_id'),
            enhanced.get('face_enhance'),
            enhanced.get('outscale'))

def cache_and_return_images(image_uris):
    images = []
    # if an image is deleted in the bucket, it needs to be reflected in load method.
    ret_image_uris = []
    os.makedirs(f'{LOCAL_SAVE_PATH}/images/',exist_ok=True)
    for image_uri in image_uris:
        image_filename = image_uri.split('/')[-1]
        local_image_uri = f'{LOCAL_SAVE_PATH}/images/{image_filename}'
        if exists(local_image_uri):
            print(f"{local_image_uri} exists...")
            images.append(Image.open(local_image_uri))
            ret_image_uris.append(local_image_uri)
        else:
            if gcs_utils.get_blob(image_uri).exists():
                print(f"downloading {image_uri}")
                tmp_img = gcs_utils.read_image(gcs_utils.get_blob(image_uri))
                tmp_img.save(local_image_uri)
                images.append(tmp_img)
                ret_image_uris.append(local_image_uri)

    return images, ret_image_uris

def load(metadata_file_uri):
    metadata_blob = gcs_utils.get_blob(metadata_file_uri)
    if metadata_blob.exists():
        cache_utils.save_ui_values(UI_VALS_CACHE,metadata_file_uri=metadata_file_uri)
        with metadata_blob.open("r") as f:
            metadata_contents = f.readlines()
            tmp = []
            for l in metadata_contents:
                metadata_content = json.loads(l)
                image_uris = metadata_content['image_uris']
                images, image_uris = cache_and_return_images(image_uris)
                tmp.extend(images)
                for image_uri in image_uris:
                    metadata_content = json.loads(l)
                    metadata_content['image_uris'] = image_uri
                    global_metadata_contents.append(metadata_content)
            metadata_contents = tmp
    else:
        gr.Error("No file found")
    
    return metadata_contents

def view_creations():
    load_ui_values()
    with gr.Blocks():
        with gr.Accordion(""):
            gr.Markdown("""
            This GUI uses [Google Cloud Platform](https://cloud.google.com/) to deploy stable diffusion jobs to generate images. There are two sections, `View your job` and `Create a job`. 
            
            First you'll create a job through the `Create a job` section, which it will deploy it to GCP's AIML platform, Vertex AI. Once the job is complete, you'll receive a Google Cloud Storage url that you can paste in the `View your job` section to load your generations and view their parameters.

            **Features**
            - Generate images using various models uploaded to HuggingFace in the same job. Currently only diffusers models are supported (ckpt models coming soon).
            - Face enhancement.
            - Upscaling
            """)
        with gr.Row():
            with gr.Accordion(""):
                gr.Markdown("""

                ## Create a job

                You can create a batch job with your favorite stable diffusion models here! 

                This uses GCP resources as follows:
                - Cloud build to build a docker image that will be used to run a training job.
                - Vertex AI to create a custom job where all predictions will be made.
                - Cloud storage to save all generated images and metadata file.
                """)
                gr.Markdown("**Step 1. Enter the required job parameters**")
                batch_job_project_id = gr.Textbox(global_ui_values.get('project_id',''), label='Enter your GCP project id')
                batch_job_region = gr.Textbox('us-central1', label='Enter your region')
                batch_job_accelerator_type = gr.Dropdown(label='Accelerator Type', value='NVIDIA_TESLA_T4',
                    choices=['NVIDIA_TESLA_T4','NVIDIA_TESLA_P4','NVIDIA_TESLA_K80', 'NVIDIA_TESLA_V100', 'NVIDIA_TESLA_A100', 'NVIDIA_A100_80GB'])
                batch_job_gcs_folder = gr.Textbox(global_ui_values.get('gcs_folder',''), label='Enter the gcs folder where results are stored (Ex: gs://bucket-name/sd-predictions/)')
                batch_job_hugginface_token = gr.Textbox(global_ui_values.get('hf_token',''), label='Enter your huggingFace token (Needed to access diffusion models)')
                with gr.Row():
                    with gr.Column():
                        gr.Markdown(
                            """
                            **Step 2. To add an entry, fill in the fields and click the `Add Entry` button. You can add as many entries as you like which will be displayed in a table below before submitting a job.**

                                - You can click on an existing entry in the table, make changes and click the `Edit Entry` button to edit its parameters.
                                - You can click on an existing entry in the table and click the `Delete Entry` button to delete it from the list.
                                - If you've run a job before, you can click on `Load last job` to load the last job's entries.
                                - When you are ready to submit the job, click the `Submit` button. 

                            Submitting a job for the first time, might take longer as a new Docker image is built. On subsequent builds, the build steps are cached and will run much faster.

                            A training job might take longer based on the number of entries it contains. You can view the job in the GCP Vertex AI console.
                            

                            **Entry Form**
                            """)
                        with gr.Row():
                            batch_job_prompt = gr.Textbox(label="Prompt (Required)")
                            batch_job_neg_prompt = gr.Textbox(label="Negative prompt")
                        with gr.Row(label="Enhancements"):
                            batch_job_enhanced_model = gr.Dropdown(label='Enhancement model id',value='None',choices=['None','RealESRGAN_x4plus','RealESRGAN_x4plus_anime_6B'])
                            batch_job_enhanced_face_restore = gr.Dropdown(label='Enhancement face restore (Works if Enhancement model id is set)', value=False,choices=[True,False])
                            batch_job_enhanced_scale = gr.Dropdown(label='Image upscaling (Works if Enhancement model id is set)',value=2,choices=[2,4])
                        with gr.Row():
                            batch_job_model_id = gr.Textbox(label='Model id', value='CompVis/stable-diffusion-v1-4')
                            batch_job_model_id_dropdown = gr.Dropdown(label='You can select a model from this list to autopopulate model id.', value='CompVis/stable-diffusion-v1-4',choices=MODELS)
                            batch_job_model_id_description = gr.Markdown("Model description")
                        batch_job_scheduler = gr.Dropdown(['DDIMScheduler','DPMSolverMultistepScheduler', 'EulerAncestralDiscreteScheduler',
                            'EulerDiscreteScheduler','LMSDiscreteScheduler','PNDMScheduler'], value='DDIMScheduler', label="Scheduler")
                        batch_job_steps = gr.Textbox("50", label="Steps")
                        batch_job_scale = gr.Textbox("7.5", label="Scale")
                        batch_job_num_images = gr.Textbox("4", label="Number of images to generate (batch). If too large, the GPU might run out of memory.")
                        batch_job_seed = gr.Textbox("-1", label="Seed (defaults to random)")
                        with gr.Row():
                            add_batch_job_entry_btn = gr.Button('Add entry')
                            edit_batch_job_entry_btn = gr.Button('Edit entry')
                            delete_batch_job_entry_btn = gr.Button('Delete entry')
                        load_last_batch_job_entry_btn = gr.Button('Load last job')
                with gr.Row():
                    batch_job_dataset = gr.Dataset(samples=[[]], type="index",
                        components=[
                            batch_job_prompt, 
                            batch_job_neg_prompt, 
                            batch_job_model_id, 
                            batch_job_scheduler, 
                            batch_job_scale, 
                            batch_job_steps, 
                            batch_job_num_images,
                            batch_job_seed,
                            batch_job_enhanced_model,
                            batch_job_enhanced_face_restore,
                            batch_job_enhanced_scale
                        ],
                        headers = ['Prompt', 'Negative prompt', 'Model id', 
                            'Scheduler', 'Scale', 'Steps', 'Num. images', 'Seed',
                            'Enhance model','Enhance face restore','Enhance scale'])
                submit_batch_prediction_job_btn = gr.Button('Submit job')
                submit_batch_prediction_job_error_code = gr.HTML()
                submit_batch_prediction_job_id = gr.Textbox(visible=True,show_label=False)
        
        with gr.Accordion(""):
            gr.Markdown("""
            ## View your job

            Once you've run a job, you'll see a gcs url in the logs (shown below the `Submit Job`). Take that url and paste in in the `Metadata uri` field below and click the `Load` button. Depending on the number of generations, this can take from 1 to 10 minutes as the UI downloads and caches the images.

            Once the images are loaded, you can click on the gallery to zoom into each image and click the `View Parameters` button to view the parameters that were used to generate the images.

            """)
            with gr.Row():
                metadata_file_uri = gr.Textbox(global_ui_values.get('metadata_file_uri',''), label='Metadata uri',placeholder="gs://bucket-name/sd-predictions/results.jsonl")
                load_metadata_file_btn = gr.Button("Load")
            with gr.Row():
                creations_prompt = gr.Textbox(label='Prompt')
                creations_neg_prompt = gr.Textbox(label='Negative prompt')
            with gr.Row(label="Enhancements"):
                creations_enhanced_model = gr.Dropdown(label='Enhancement model id',value='None',choices=['None','RealESRGAN_x4plus','RealESRGAN_x4plus_anime_6B'])
                creations_enhanced_face_restore = gr.Dropdown(label='Enhancement face restore', value=False,choices=[True,False])
                creations_enhanced_scale = gr.Textbox(label='Image upscaling')
            with gr.Row():
                with gr.Column():
                    creations_model_id = gr.Textbox(label='Model id')
                    creations_scheduler = gr.Textbox(label='Scheduler')
                    creations_num_inference_steps = gr.Textbox(label='Steps')
                    creations_scale = gr.Textbox(label='Scale')
                    creations_seed = gr.Textbox(label='Seed')
                    creations_image_uri = gr.Textbox(label='Image uri')
                with gr.Column(scale=2):
                    creations_gallery = gr.Gallery(label='Images', elem_id='creations_gallery').style(grid=[4, 4])
                    view_params_btn = gr.Button('View Parameters')

        def batch_job_model_id_dropdown_change(x):
            print(x)
            return x, f'Model description : {MODEL_DICT[x]}'

        batch_job_model_id_dropdown.change(
            batch_job_model_id_dropdown_change,
            batch_job_model_id_dropdown,
            [batch_job_model_id, batch_job_model_id_description]
        )
    
        def batch_job_dataset_click(x):
            print("entry index: ",x)
            global_batch_job_dataset_idx[0] = x
            return global_batch_job_entries[x]

        batch_job_dataset.click(
            batch_job_dataset_click,
            inputs = batch_job_dataset,
            outputs = [batch_job_prompt, batch_job_neg_prompt, batch_job_model_id,
            batch_job_scheduler, batch_job_scale, batch_job_steps, batch_job_num_images,
            batch_job_seed, batch_job_enhanced_model, batch_job_enhanced_face_restore,
            batch_job_enhanced_scale])

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
            creations_seed, creations_image_uri, creations_enhanced_model, creations_enhanced_face_restore,
            creations_enhanced_scale],
            _js=call_js("getGallerySelectedItem",element_id="creations_gallery")
        )

        add_batch_job_entry_btn.click(
            add_batch_job_entry,
            [batch_job_prompt, batch_job_neg_prompt, batch_job_model_id,
            batch_job_scheduler, batch_job_scale, batch_job_steps, batch_job_num_images, batch_job_seed,
            batch_job_enhanced_model, batch_job_enhanced_face_restore, batch_job_enhanced_scale],
            [batch_job_dataset, submit_batch_prediction_job_error_code]
        )
        edit_batch_job_entry_btn.click(
            edit_batch_job_entry,
            [batch_job_prompt, batch_job_neg_prompt, batch_job_model_id,
            batch_job_scheduler, batch_job_scale, batch_job_steps, batch_job_num_images, batch_job_seed,
            batch_job_enhanced_model, batch_job_enhanced_face_restore, batch_job_enhanced_scale],
            batch_job_dataset
        )
        delete_batch_job_entry_btn.click(
            delete_batch_job_entry,
            inputs=None,
            outputs=batch_job_dataset
        )
        load_last_batch_job_entry_btn.click(
            load_last_job_entry,
            inputs=None,
            outputs=batch_job_dataset
        )

        submit_batch_prediction_job_btn.click(
            submit_batch_job,
            [batch_job_project_id, batch_job_region,
            batch_job_accelerator_type, batch_job_gcs_folder, batch_job_hugginface_token],
            [submit_batch_prediction_job_error_code, submit_batch_prediction_job_id]
        )