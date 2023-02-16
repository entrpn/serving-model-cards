from google.cloud import aiplatform

import base64
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
import argparse
import json

def main(opt):
    prompt = "Generate an article about quantum physics:"
    instances_list = {
            "instances" : [
                {
                    "prompt": prompt
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