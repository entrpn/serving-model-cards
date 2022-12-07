
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
DEFAULT_OUTSCALE = 2
DEFAULT_MODEL_ID = 'RealESRGAN_x4plus'

def inference_realesrgan(paths, config):
    tile = config.get('tile', DEFAULT_TILE)
    tile_pad = config.get('tile_pad', DEFAULT_TILE_PAD)
    prepad = config.get('prepad', DEFAULT_PREPAD)
    face_enhance = config.get('face_enhance', DEFAULT_FACE_ENHANCE)
    fp32 = config.get('fp32', DEFAULT_FP32)
    outscale = config.get('outscale', DEFAULT_OUTSCALE)
    if outscale <= 1 or outscale >4:
        outscale = 2
    model_id = config.get('model_id', DEFAULT_MODEL_ID)

    if model_id == 'RealESRGAN_x4plus_anime_6B':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
    else:
        model_id == 'RealESRGAN_x4plus'
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4

    model_path = os.path.join('./experiments/pretrained_models',f'{model_id}.pth')

    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        model=model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=prepad,
        half= not fp32,
        gpu_id=0
    )

    if face_enhance:
        face_enhancer = GFPGANer(
            model_path=os.path.join('./experiments/pretrained_models','GFPGANv1.3.pth'),
            upscale=outscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upsampler
        )

    for path in paths:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
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
            cv2.imwrite(path, output)
    
    return retval, error