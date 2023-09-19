from google.cloud import aiplatform

from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
import argparse
import json

def main(opt):
    instances_list = {
            "instances": [
                {
                    "user_input" : "Tell me about dark matter and black holes.", 
                }
            ],
        }
    instances = json.dumps(instances_list)#.encode('utf-8')
    print(instances)
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