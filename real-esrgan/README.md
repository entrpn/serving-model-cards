# serving real-esrgan

<center>
    <image src="../images/real_esrgan.jpeg" width="256px">
</center>

## Intro

This repo containerizes [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) into a serving container using [fastapi](https://fastapi.tiangolo.com/). 

**The model license can be found [here.](https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE)**

## Setup

1. Clone repo if you haven't. Navigate to the `real-esrgan` folder.
1. Build container. Don't forget to change the `project_id` to yours.

    ```bash
    docker build . -t gcr.io/{project_id}/real-esrgan:latest
    ```

1. Run container. You need [NVIDIA docker](https://github.com/NVIDIA/nvidia-docker) and a GPU.

    ```bash
    docker run -p 80:8080 --gpus all -e AIP_HEALTH_ROUTE=/health -e AIP_HTTP_PORT=8080 -e AIP_PREDICT_ROUTE=/predict gcr.io/{project_id}/real-esrgan:latest -d
    ```

1. Make a prediction

    ```bash
    python generate_request.py
    curl -X POST -d @request.json -H "Content-Type: application/json; charset=utf-8" localhost/predict > response.json
    ```

## Deploy in Vertex AI.

You'll need to enable Vertex AI and have authenticated with a service account that has the Vertex AI admin or editor role.

1. Push the image

    ```bash
    gcloud auth configure-docker
    docker build . -t gcr.io/{project_id}/real-esrgan:latest
    docker push gcr.io/{project_id}/real-esrgan:latest
    ```

 1. Deploy in Vertex AI Endpoints.

    ```bash
    python ../gcp_deploy.py --image-uri gcr.io/<project_id>/real-esrgan:latest --model-name real-esrgan --endpoint-name real-esrgan-endpoint --endpoint-deployed-name real-esrgan-deployed-name
    ```

1. Test the endpoint.

    ```python
    from google.cloud import aiplatform

    from google.protobuf import json_format
    from google.protobuf.struct_pb2 import Value

    from PIL import Image

    # Format is projects/<project_id>/locations/us-central1/endpoints/<endpoint_id>
    ENDPOINT_NAME = ''

    def im_2_b64(image, format='PNG'):
        buff = BytesIO()
        image.save(buff, format=format)
        img_str = base64.b64encode(buff.getvalue())
        return img_str

    image = Image.open("image.png")

    base64_image = im_2_b64(image).decode('utf-8')

    instances_list = [{'image' : base64_image}]
    instances = [json_format.ParseDict(s, Value()) for s in instances_list]

    parameters = {
        'face_enhance' : True,
        'tile' : 0, 
        'tile_pad' : 10, 
        'prepad' : 0,
        'fp32' : true,
        'outscale' : 4
    }
    parameters = json_format.ParseDict(parameters,Value())
    endpoint = aiplatform.Endpoint(ENDPOINT_NAME)
    results = endpoint.predict(instances=instances,parameters=parameters)
        ```