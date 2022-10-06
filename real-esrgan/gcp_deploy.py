import argparse
from google.cloud import aiplatform

def main(opt):

    models = aiplatform.Model.list()
    model_found = False
    for model in models:
        
        if model.display_name == opt.model_name:
            print(model.resource_name)
            model_found = True
            break
    
    if not model_found:
        print("No model found, creating...")
        model = aiplatform.Model.upload(
            display_name=opt.model_name,
            serving_container_image_uri=opt.image_uri,
            serving_container_ports=[8080],
            serving_container_predict_route="/predict"
        )

    endpoints = aiplatform.Endpoint.list()
    endpoint_found = False
    for endpoint in endpoints:
        if endpoint.display_name == opt.endpoint_name:
            print(endpoint.resource_name)
            endpoint_found = True
            break    
    
    if not endpoint_found:
        print("No endpoint found, creating...")
        endpoint = aiplatform.Endpoint.create(
            display_name=opt.endpoint_name,
            create_request_timeout=300.0
        )
    print(vars(endpoint)) 
    models = endpoint.list_models()
    if len(models) > 0:
        print("undeploying previous models...")
        endpoint.undeploy_all()

    #NVIDIA_TESLA_V100
    #NVIDIA_TESLA_T4
    endpoint.deploy(
        model,
        deployed_model_display_name="stable_diffusion_endpoint_deployed",
        traffic_percentage=100,
        machine_type=opt.machine_type,
        min_replica_count=opt.min_replica_count,
        max_replica_count=opt.max_replica_count,
        accelerator_type=opt.gpu_type,
        accelerator_count=opt.accelerator_count,
        sync=True,
        deploy_request_timeout=300.0
    )

    print("done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--min-replica-count',
        type=int,
        default=1,
        help='Minimum number of replicas'
    )
    parser.add_argument(
        '--machine-type',
        type=str,
        default='n1-standard-8',
        help='Machine type'
    )
    parser.add_argument(
        '--max-replica-count',
        type=int,
        default=1,
        help='Maximum number of replicas'
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
        help="GPU count"
    )
    parser.add_argument(
        "--region",
        type=str,
        default="us-central1",
        help="gcp region"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="real-esrgan",
        help="name of model"        
    )
    parser.add_argument(
        "--endpoint-name",
        type=str,
        default="real-esrgan-endpoint",
        help="Name of endpoint"        
    )
    parser.add_argument(
        "--endpoint-deployed-name",
        type=str,
        default="real-esrgan_endpoint_deployed",
        help="Endpoint deployed name"
    )
    parser.add_argument(
        "--image-uri",
        type=str,
        required=True,
        help="name of image in gcr. Ex: gcr.io/project-name/real-esrgan:latest"
    )

    opt = parser.parse_args()
    main(opt)