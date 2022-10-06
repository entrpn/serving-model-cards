
import os
import base64

import cv2
import numpy as np
from PIL import Image
from io import BytesIO

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from gfpgan import GFPGANer

import logging
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


DEFAULT_TILE = 0
DEFAULT_TILE_PAD = 10
DEFAULT_PREPAD = 0
DEFAULT_FACE_ENHANCE = True
DEFAULT_FP32 = True
DEFAULT_OUTSCALE = 4

def inference_realesrgan(img, config):
    tile = config.get('tile', DEFAULT_TILE)
    tile_pad = config.get('tile_pad', DEFAULT_TILE_PAD)
    prepad = config.get('prepad', DEFAULT_PREPAD)
    face_enhance = config.get('face_enhance', DEFAULT_FACE_ENHANCE)
    fp32 = config.get('fp32', DEFAULT_FP32)
    outscale = config.get('outscale', DEFAULT_OUTSCALE)

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    netscale = 4

    model_path = os.path.join('./experiments/pretrained_models','RealESRGAN_x4plus.pth')

    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        model=model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=prepad,
        half= not fp32,
        gpu_id=None
    )

    if face_enhance:
        face_enhancer = GFPGANer(
            model_path=os.path.join('./experiments/pretrained_models','GFPGANv1.3.pth'),
            upscale=outscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upsampler
        )
    os.makedirs('results',exist_ok=True)

    ## TODO read file
    img = base64.b64decode(img); 
    npimg = np.fromstring(img, dtype=np.uint8); 
    img = cv2.imdecode(npimg, 1) # 1 corresponds to RGB
    retval = None
    error = False
    try:
        if face_enhance:
            _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
        else:
            output, _ = upsampler.enhance(img, outscale=outscale)
    except RuntimeError as error:
        print('Error', error)
        print('If you encountered CUDA out of memory, try to set --tile with a smaller number')
        error = True
    else:
        # cv2 uses BRG while PIL uses RGB
        img = Image.fromarray(output[:,:,:3][:,:,::-1])
        buff = BytesIO()
        img.save(buff, format='PNG')
        retval = base64.b64encode(buff.getvalue())
    
    return retval, error
    
    