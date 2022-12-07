from google.cloud import aiplatform

def get_job_state_str(job_state):
    job_state_dict = {
        0 : 'UNSPECIFIED',
        1 : 'QUEUED',
        2 : 'PENDING',
        3 : 'RUNNING',
        4 : 'SUCCEEDED',
        5 : 'FAILED',
        6 : 'CANCELLING',
        7 : 'CANCELLED',
        8 : 'PAUSED',
        9 : 'EXPIRED',
        10 : 'UPDATING'
    }
    return job_state_dict.get(job_state,'UNSPECIFIED')

def get_custom_job_sample(
    project,
    custom_job_id,
    location
):
    api_endpoint = f"{location}-aiplatform.googleapis.com"
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.JobServiceClient(client_options=client_options)
    name = client.custom_job_path(
        project=project, location=location, custom_job=custom_job_id
    )
    response = client.get_custom_job(name=name)
    print("response:", response)
    return response

def create_custom_job_sample(
    project_id, location, display_name, gpu_type, 
    gcs_output_dir, hf_token, image_uri, accelerator_count=1
):

    # The AI Platform services require regional API endpoints.
    api_endpoint = f"{location}-aiplatform.googleapis.com"
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.JobServiceClient(client_options=client_options)
    custom_job = {
        "display_name": display_name,
        "job_spec": {
            "worker_pool_specs": [
                {
                    "machine_spec": {
                        "machine_type": "n1-standard-8",
                        "accelerator_type": gpu_type,
                        "accelerator_count": accelerator_count,
                    },
                    "replica_count": 1,
                    "disk_spec" : {
                        "boot_disk_type": "pd-ssd",
                        "boot_disk_size_gb" : 500
                    },
                    "container_spec": {
                        "image_uri": image_uri,
                        "command": [],
                        "args": [],
                        "env" : [
                            {"name" : "GCS_OUTPUT_DIR", "value" : gcs_output_dir},
                            {"name" : "HF_TOKEN", "value" : hf_token},
                        ]
                    },
                }
            ],
            "enable_web_access" : True
        },
    }
    parent = f"projects/{project_id}/locations/{location}"
    error = None
    try:
        response = client.create_custom_job(parent=parent, custom_job=custom_job)
        print("response:", response)
    except Exception as e:
        print('error:', e)
        error = e
    job_id = None
    if error is None:
        job_id = response.name.split('/')[-1]
    return job_id, error