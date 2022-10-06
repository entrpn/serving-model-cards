import os
import cv2
import argparse
import glob
import torch
import base64
import numpy as np
from torchvision.transforms.functional import normalize
from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from facelib.utils.face_restoration_helper import FaceRestoreHelper
from facelib.utils.misc import is_gray
import torch.nn.functional as F
from io import BytesIO
from basicsr.utils.registry import ARCH_REGISTRY
from PIL import Image

BG_UPSAMPLER = 'bg_upsampler'
FACE_UPSAMPLE = 'face_upsample'
BG_TILE = 'bg_tile'
UPSCALE = 'upscale'
HAS_ALIGNED = 'has_aligned'
ONLY_CENTER_FACE = "only_center_face"
DRAW_BOX = "draw_box"
W = "w"

pretrain_model_url = {
    'restoration': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
}

def set_realesrgan(bg_tile):
    if not torch.cuda.is_available():  # CPU
        import warnings
        warnings.warn('The unoptimized RealESRGAN is slow on CPU. We do not use it. '
                        'If you really want to use it, please modify the corresponding codes.',
                        category=RuntimeWarning)
        bg_upsampler = None
    else:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from basicsr.utils.realesrgan_utils import RealESRGANer
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        bg_upsampler = RealESRGANer(
            scale=2,
            model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
            model=model,
            tile=bg_tile,
            tile_pad=40,
            pre_pad=0,
            half=True)  # need to set False in CPU mode
    return bg_upsampler

def face_restore(net, image, config):

    retval = None
    err = None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bg_upsampler = config.get(BG_UPSAMPLER,None)
    face_upsampler = config.get(FACE_UPSAMPLE,False)
    bg_tile = config.get(BG_TILE, 400)
    upscale = config.get(UPSCALE, 2)
    has_aligned = config.get(HAS_ALIGNED, False)
    only_center_face = config.get(ONLY_CENTER_FACE, False)
    face_upsample = config.get(FACE_UPSAMPLE, False)
    draw_box = config.get(DRAW_BOX, False)
    w = config.get(W,0.5)

    if bg_upsampler == 'realesrgan':
        bg_upsampler = set_realesrgan(bg_tile)
    else:
        bg_upsampler = None
    
    if face_upsampler:
        if bg_upsampler is not None:
            face_upsampler = bg_upsampler
        else:
            face_upsampler = set_realesrgan(bg_tile)
    else:
        face_upsampler = None

    face_helper = FaceRestoreHelper(
        upscale,
        face_size=512,
        crop_ratio=(1, 1),
        det_model = 'retinaface_resnet50',
        save_ext='png',
        use_parse=True,
        device=device
    )
    face_helper.clean_all()

    img = base64.b64decode(image)
    npimg = np.fromstring(img, dtype=np.uint8); 
    img = cv2.imdecode(npimg, 1) # 1 corresponds to RGB
    face_helper.is_gray = is_gray(img, threshold=5)

    if has_aligned:
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
        face_helper.is_gray = is_gray(img, threshold=5)
        if face_helper.is_gray:
            print('Grayscale input: True')
        face_helper.cropped_faces = [img]
    else:
        face_helper.read_image(img)
        num_det_faces = face_helper.get_face_landmarks_5(
            only_center_face=only_center_face, resize=640, eye_dist_threshold=5
        )
        print(f'\tdetect {num_det_faces} faces')
        face_helper.align_warp_face()
    
    for idx, cropped_face in enumerate(face_helper.cropped_faces):
        cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
        normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

        try:
            with torch.no_grad():
                output = net(cropped_face_t, w=w, adain=True)[0]
                restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
            del output
            torch.cuda.empty_cache()
        except Exception as error:
            print(f'\tFailed inference for Codeformer: {error}')
            err = error
            restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))
        
        restored_face = restored_face.astype('uint8')
        face_helper.add_restored_face(restored_face)

    if not has_aligned:
        if bg_upsampler is not None:
            bg_img = bg_upsampler.enhance(img, outscale=upscale)[0]
        else:
            bg_img = None
        face_helper.get_inverse_affine(None)

        if face_upsample and face_upsampler is not None:
            restored_img = face_helper.paste_faces_to_input_image(upsample_img=bg_img, draw_box=draw_box, face_upsampler=face_upsampler)
        else:
            restored_img = face_helper.paste_faces_to_input_image(upsample_img=bg_img, draw_box=draw_box)
    restored_img = Image.fromarray(restored_img[:,:,:3][:,:,::-1])
    buff = BytesIO()
    restored_img.save(buff, format='PNG')
    retval = base64.b64encode(buff.getvalue())
            
    
    return retval, err
