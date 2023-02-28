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
                        "machine_type": "n1-standard-16",
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
                            {"name" : "DATA_DIR", "value" : args.gcs_datadir},
                            {"name" : "NUM_TRAIN_IMAGES", "value" : str(args.num_train_images)},
                            {"name" : "NUM_EVAL_IMAGES", "value" : str(args.num_eval_images)},
                            {"name" : "MODEL_OUTPUT_URI", "value" : args.model_output_dir},
                            {"name" : "BATCH_SIZE", "value" : str(args.batch_size)}
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
        default="semantic segmentation training job",
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
        type=int,
        default=1,
        help="Number of accelerators"
    )
    parser.add_argument(
        "--image-uri",
        type=str,
        default=None,
        required=True,
        help="GCR image. Ex: gcr.io/project_id/image_name:latest"
    )
    parser.add_argument(
        "--gcs-datadir",
        type=str,
        required=True,
        default=None,
        help="A GCS bucket where data is located."
    )
    parser.add_argument(
        "--num-train-images",
        type=int,
        default=None,
        help="Number of images to use for training"
    )
    parser.add_argument(
        "--num-eval-images",
        type=int,
        default=None,
        help="Number of images to use eval"
    )
    parser.add_argument(
        "--model-output-dir",
        type=str,
        default=None,
        help="GCS bucket to save the model after training"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size"
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    create_custom_job_sample(args)

if __name__ == "__main__":
    main()