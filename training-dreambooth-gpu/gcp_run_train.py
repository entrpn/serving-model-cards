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
                        "machine_type": args.machine_type,
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
                            {"name" : "MODEL_NAME", "value" : f"{args.model_name}"},
                            {"name" : "MODEL_REVISION", "value" : args.model_revision},
                            {"name" : "INSTANCE_PROMPT", "value" : f"{args.instance_prompt}"},
                            {"name" : "GCS_OUTPUT_DIR", "value" : args.gcs_output_dir},
                            {"name" : "RESOLUTION", "value" : args.resolution},
                            {"name" : "BATCH_SIZE", "value" : args.batch_size},
                            {"name" : "LEARNING_RATE", "value" : args.learning_rate},
                            {"name" : "MAX_TRAIN_STEPS", "value" : args.max_train_steps},
                            {"name" : "CLASS_PROMPT", "value" : args.class_prompt},
                            {"name" : "NUM_CLASS_IMAGES", "value" : args.num_class_images},
                            {"name" : "PRIOR_LOSS_WEIGHT", "value" : args.prior_loss_weight},
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
        default="training-dreambooth",
        help="Training job name"
    )
    parser.add_argument(
        "--machine-type",
        type=str,
        default='n1-standard-8',
        help="Machine type. Ex: a2-highgpu-1g for A100"
    ),
    parser.add_argument(
        "--gpu-type",
        type=str,
        default='NVIDIA_TESLA_T4',
        help="GPU type. Ex: NVIDIA_TESLA_A100"
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
        "--model-name",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        help="Hugginface model name"
    )
    parser.add_argument(
        "--model-revision",
        type=str,
        default="fp16",
        help="Model revision"
    )
    parser.add_argument(
        "--gcs-output-dir",
        type=str,
        required=True,
        default=None,
        help="A GCS bucket location. Ex: gs://my-bucket-name/sd-model/"
    )
    parser.add_argument(
        "--instance-prompt",
        type=str,
        default=None,
        required=True,
        help="The instance prompt. Ex: A photo of sks dog"
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
        help="Batch size"
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
    parser.add_argument(
        "--class-prompt",
        type=str,
        default=None,
        help="Class prompt. Ex: a photo of a dog."
    )
    parser.add_argument(
        "--num-class-images",
        type=str,
        default="128",
        help="Minimal class images for prior preservation loss. If not have enough images, additional images will be"
            " sampled with class_prompt."
    )
    parser.add_argument(
        "--prior-loss-weight",
        type=str,
        default="0.5",
        help="The weight of prior preservation loss."
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    create_custom_job_sample(args)

if __name__ == "__main__":
    main()