import os 
from typing import Tuple
from typing import List
import json
from io import BytesIO
import requests
import apache_beam as beam
from PIL import Image
import tensorflow as tf

class ReadImagesFromGcsUrl(beam.DoFn):
    def process(self, element: str) -> Tuple[str, Image.Image]:
        print("element:",element)
        with tf.io.gfile.GFile(element, "rb") as f:
            #print("type:",image)
            return [(element, Image.open(BytesIO(f.read())).convert("RGB"))]


class ReadImagesFromUrl(beam.DoFn):
  """
  Read an image from a given URL and return a tuple of the images_url
  and image data.
  """
  def process(self, element: str) -> Tuple[str, Image.Image]:
    response = requests.get(element)
    image = Image.open(BytesIO(response.content)).convert('RGB')
    return [(element, image)]


class FormatCaptions(beam.DoFn):
  """
  Print the image name and its most relevant captions after CLIP ranking.
  """
  def __init__(self, number_of_top_captions: int):
    self._number_of_top_captions = number_of_top_captions

  def process(self, element: Tuple[str, List[str]]):
    image_url, caption_list = element
    caption_list = caption_list[:self._number_of_top_captions]
    img_name = os.path.basename(image_url).rsplit('.')[0]
    print(f'\tTop {self._number_of_top_captions} captions ranked by CLIP:')
    for caption_rank, caption_prob_pair in enumerate(caption_list):
      print(f'\t\t{caption_rank+1}: {caption_prob_pair[0]}. (Caption probability: {caption_prob_pair[1]:.2f})')
    print('\n')
    retval = json.dumps({"file_name" : image_url, "text" : caption_prob_pair[0]})
    return [retval]