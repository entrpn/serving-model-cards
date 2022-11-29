from huggingface_hub import hf_hub_download
import argparse

def main(opt):
    hf_hub_download(repo_id="CompVis/stable-diffusion-v-1-4-original", filename="sd-v1-4.ckpt", use_auth_token=opt.hf_token,cache_dir='.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--hf_token',
        type=str
    )
    opt = parser.parse_args()
    main(opt)