import json
import base64
import os
import argparse
import uuid

os.makedirs('./outputs/',exist_ok=True)

def main(opt):
    response = opt.response_json
    with open(response,'r') as json_file:
        data = json.load(json_file)
    predictions = data["predictions"]
    for prediction in predictions:
        print(prediction["prompt"])
        for image in prediction["images"]:
            unique_id = str(uuid.uuid4())[:8]
            img_path = f"outputs/{unique_id}.png"
            with open(img_path, "wb") as fh:
                fh.write(base64.b64decode(image))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--response-json',
        type=str,
        default="response.json",
        help='Response json location'
    )
    opt = parser.parse_args()
    main(opt)