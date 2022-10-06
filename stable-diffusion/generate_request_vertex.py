from google.cloud import aiplatform

from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
import argparse
import json

def main(opt):
    instances_list = {
            "instances" : [
                {
                    "prompt": "A dog wearing a hat",
                    "parameters" : {
                        "scale" : 7.5,
                        "seed" : 42,
                        "W" : 512,
                        "H" : 512,
                        "ddim_steps" : 30,
                        "n_samples" : 2,
                        "n_iter" : 2,
                        "type" : "txt2img"
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