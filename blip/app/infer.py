from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

import BytesIO
import base64

from models.blip import blip_decoder
from models.blip_vqa import blip_vqa
from models.blip import blip_feature_extractor
from models.blip_itm import blip_itm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(image, image_size):
    img_bytes = base64.b64decode(base64_image_str)
    img_file = BytesIO(img_bytes)  # convert image to file-like object
    image = Image.open(img_file)   # img is now PIL Image object
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 
    image = transform(image).unsqueeze(0).to(device)   
    return image

def caption(image, image_size, num_beams, max_length, min_length):
    image = load_image(image, image_size)
    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'
    model = blip_decoder(pretrained=model_url, image_size=image_size, vit='base')
    model.eval()
    model = model.to(device)
    captions = []
    with torch.no_grad():
        # beam search
        caption = model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5) 
        # nucleus sampling
        # caption = model.generate(image, sample=True, top_p=0.9, max_length=20, min_length=5) 
        for c in caption:
            captions.append(c)
    return {"captions" : captions}


def qna(image, image_size, questions):
    image = load_image(image, image_size)
    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_vqa_capfilt_large.pth'
    
    model = blip_vqa(pretrained=model_url, image_size=image_size, vit='base')
    model.eval()
    model = model.to(device)
    retval = []
    for question in questions:
        answers = []
        with torch.no_grad():
            answer = model(image, question, train=False, inference='generate') 
            for an in answer:
                answers.append(an)
        retval.append({"question" : question, "answers" : answers})
    return retval

def img2txt_matching(image, image_size, captions):
    image = load_image(image, image_size)
    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth'
    
    model = blip_itm(pretrained=model_url, image_size=image_size, vit='base')
    model.eval()
    model = model.to(device=device)

    retval = []
    for caption in captions:
        itm_output = model(image,caption,match_head='itm')
        itm_score = torch.nn.functional.softmax(itm_output,dim=1)[:,1]

        itc_score = model(image,caption,match_head='itc')

        retval.append({"caption" : caption, "match_probability" : itm_score, "cosine_similarity" : itc_score})
    
    return retval

def feature_extraction(image, image_size, captions):
    image = load_image(image, image_size)
    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth'
    
    model = blip_feature_extractor(pretrained=model_url, image_size=image_size, vit='base')
    model.eval()
    model = model.to(device)

    retval = []
    for caption in captions:
        multimodal_feature = model(image, caption, mode='multimodal')[0,0]
        image_feature = model(image, caption, mode='image')[0,0]
        text_feature = model(image, caption, mode='text')[0,0]
        retval.append("caption" : caption, "image_feature" : image_feature, "text_feature" : text_feature)
    return retval