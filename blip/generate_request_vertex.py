from google.cloud import aiplatform

import base64
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
import argparse
import json

with open('../images/blip_test.jpeg', "rb") as image_file:
    img = base64.b64encode(image_file.read())

def main(opt):
    instances_list = {
            "instances" : [
                {
                    "image": img.decode('utf-8'),
                    "parameters" : {
                        "type" : "captioning",
                        "sample" : False,
                        "img_size" : 384,
                        "num_beams" : 3,
                        "max_length" : 20,
                        "min_length" : 5
                    }
                }
            ]
        }
    instances = json.dumps(instances_list).encode('utf-8')

    endpoint = aiplatform.Endpoint(opt.endpoint_name)
    results = endpoint.raw_predict(body=instances, headers={'Content-Type':'application/json'})
    print(results)
    with open("response.json", "w") as text_file:
        text_file.write(results.text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--endpoint-name",
        type=str,
        required=True,
        help="Endpoint name in format projects/<project_id>/locations/us-central1/endpoints/<endpoint_id>"
    )

    opt = parser.parse_args()
    main(opt)