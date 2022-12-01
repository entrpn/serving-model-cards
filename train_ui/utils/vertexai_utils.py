from google.cloud import aiplatform

def create_custom_job_sample(
    project_id, region, display_name, gpu_type, 
    gcs_output_dir, hf_token, image_uri, accelerator_count=1
):

    # The AI Platform services require regional API endpoints.
    api_endpoint = f"{region}-aiplatform.googleapis.com"
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
    parent = f"projects/{project_id}/locations/{region}"
    response = client.create_custom_job(parent=parent, custom_job=custom_job)
    print("response:", response)