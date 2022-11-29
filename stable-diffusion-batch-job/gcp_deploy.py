from google.cloud import aiplatform
import argparse

def create_custom_job_sample(
    args
):

    # The AI Platform services require regional API endpoints.
    api_endpoint = f"{args.region}-aiplatform.googleapis.com"
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.JobServiceClient(client_options=client_options)
    custom_job = {
        "display_name": args.display_name,
        "job_spec": {
            "worker_pool_specs": [
                {
                    "machine_spec": {
                        "machine_type": "n1-standard-8",
                        "accelerator_type": args.gpu_type,
                        "accelerator_count": args.accelerator_count,
                    },
                    "replica_count": 1,
                    "disk_spec" : {
                        "boot_disk_type": "pd-ssd",
                        "boot_disk_size_gb" : 500
                    },
                    "container_spec": {
                        "image_uri": args.image_uri,
                        "command": [],
                        "args": [],
                        "env" : [
                            {"name" : "GCS_OUTPUT_DIR", "value" : args.gcs_output_dir},
                            {"name" : "HF_TOKEN", "value" : args.hf_token},
                        ]
                    },
                }
            ],
            "enable_web_access" : True
        },
    }
    parent = f"projects/{args.project_id}/locations/{args.region}"
    response = client.create_custom_job(parent=parent, custom_job=custom_job)
    print("response:", response)

def parse_args():
    parser = argparse.ArgumentParser(description="Pass your deployment params")
    parser.add_argument(
        "--project-id",
        type=str,
        default=None,
        required=True,
        help="GCP project id"
    )
    parser.add_argument(
        "--region",
        type=str,
        default="us-central1",
        help="GCP region"
    )
    parser.add_argument(
        "--display-name",
        type=str,
        default="stable-diffusion-batch-job",
        help="Training job name"
    )
    parser.add_argument(
        "--gpu-type",
        type=str,
        default='NVIDIA_TESLA_T4',
        help="GPU type"
    )
    parser.add_argument(
        "--accelerator-count",
        type=str,
        default=1,
        help="Number of accelerators"
    )
    parser.add_argument(
        "--image-uri",
        type=str,
        default=None,
        required=True,
        help="GCR image. Ex: gcr.io/project_id/training-dreambooth:latest"
    )
    parser.add_argument(
        "--gcs-output-dir",
        type=str,
        required=True,
        default=None,
        help="A GCS bucket location. Ex: gs://my-bucket-name/sd-model/"
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugginface token to access model."
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    create_custom_job_sample(args)

if __name__ == "__main__":
    main()