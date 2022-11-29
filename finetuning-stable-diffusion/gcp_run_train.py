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
                        "machine_type": "cloud-tpu",
                        "accelerator_type": aiplatform.gapic.AcceleratorType.TPU_V3,
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
                            {"name" : "MODEL_NAME", "value" : args.model_name},
                            {"name" : "GCS_OUTPUT_DIR", "value" : args.gcs_output_dir},
                            {"name" : "RESOLUTION", "value" : args.resolution},
                            {"name" : "BATCH_SIZE", "value" : args.batch_size},
                            {"name" : "LEARNING_RATE", "value" : args.learning_rate},
                            {"name" : "MAX_TRAIN_STEPS", "value" : args.max_train_steps},
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
        default="finetuning-stable-diffusion",
        help="Training job name"
    )
    parser.add_argument(
        "--accelerator-count",
        type=str,
        default=8,
        help="Number of accelerators"
    )
    parser.add_argument(
        "--image-uri",
        type=str,
        default=None,
        required=True,
        help="GCR image. Ex: gcr.io/project_id/finetuning-stable-diffusion:latest"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        help="Hugginface model name"
    )
    parser.add_argument(
        "--gcs-output-dir",
        type=str,
        required=True,
        default=None,
        help="A GCS bucket location. Ex: gs://my-bucket-name/sd-model/"
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="512",
        help="Image resolution"
    )

    parser.add_argument(
        "--batch-size",
        type=str,
        default="1",
        help="Per device batch size. Ex: On a TPUv3-8, a value of 1 will have a global batch size of 8"
    )

    parser.add_argument(
        "--learning-rate",
        type=str,
        default="5e-6",
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--max-train-steps",
        type=str,
        default="3000",
        help="Total number of training steps to perform."
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