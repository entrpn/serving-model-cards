import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
from PIL import Image
from huggingface_hub import from_pretrained_keras
from diffusers import DiffusionPipeline
import torch

model = keras.models.load_model("./segmentation")

colormap = np.array([[0,0,0], [31,119,180], [44,160,44], [44, 127, 125], [52, 225, 143],
                    [217, 222, 163], [254, 128, 37], [130, 162, 128], [121, 7, 166], [136, 183, 248],
                    [85, 1, 76], [22, 23, 62], [159, 50, 15], [101, 93, 152], [252, 229, 92],
                    [167, 173, 17], [218, 252, 252], [238, 126, 197], [116, 157, 140], [214, 220, 252]], dtype=np.uint8)

colormap = np.full(shape=(20,3),fill_value=[255,255,255])
colormap[0] = [0,0,0]

model = from_pretrained_keras("keras-io/deeplabv3p-resnet50")
img_size = 512

def read_image(image):
    image = tf.convert_to_tensor(image)
    image.set_shape([None, None, 3])
    image = tf.image.resize(images=image, size=[img_size, img_size])
    image = image / 127.5 - 1
    return image

def infer(model, image_tensor):
    predictions = model.predict(np.expand_dims((image_tensor), axis=0))
    predictions = np.squeeze(predictions)
    predictions = np.argmax(predictions, axis=2)
    return predictions

def decode_segmentation_masks(mask, colormap, n_classes):
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    for l in range(0, n_classes):
        idx = mask == l
        r[idx] = colormap[l, 0]
        g[idx] = colormap[l, 1]
        b[idx] = colormap[l, 2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb

def get_overlay(image, colored_mask):
    image = tf.keras.preprocessing.image.array_to_img(image)
    image = np.array(image).astype(np.uint8)
    overlay = cv2.addWeighted(image, 0.35, colored_mask, 0.65, 0)
    return overlay

def segmentation(input_image):
    image_tensor = read_image(input_image)
    prediction_mask = infer(image_tensor=image_tensor, model=model)
    prediction_colormap = decode_segmentation_masks(prediction_mask, colormap, 20)
    overlay = get_overlay(image_tensor, prediction_colormap)
    return (overlay, prediction_colormap)

img = np.asarray(Image.open('merged.png'))

overlay, prediction_colormap = segmentation(img)

Image.fromarray(np.uint8(overlay)).save("overlay.png")
Image.fromarray(np.uint8(prediction_colormap)).save("mask.png")

image = cv2.imread('mask.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

for c in cnts:
    cv2.drawContours(gray,[c], 0, (255,255,255), -1)
gray = cv2.bitwise_not(gray)
backtorgb = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
cv2.imwrite("mask.png",backtorgb)
print("gray.shape:",backtorgb.shape)

pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", custom_pipeline="stable_diffusion_mega", torch_dtype=torch.float32)
pipe.to("cuda")
pipe.enable_attention_slicing()

# ### Image-to-Image

### Inpainting
init_image = Image.open("merged.png").convert("RGB")
mask_image = Image.open("mask.png").convert("RGB")

prompt = "RAW photo, a photograph of a beach, ocean, sunset, highly detailed, close up shot."
images = pipe.inpaint(prompt=prompt, image=init_image, mask_image=mask_image, strength=0.8, guidance_scale=7.5).images
images[0].save("final.png")