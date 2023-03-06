from typing import Iterable
from typing import Tuple

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import apache_beam as beam
from apache_beam.ml.inference.base import PredictionResult
from apache_beam.ml.inference.pytorch_inference import PytorchModelHandlerTensor
from model_handler import PytorchModelHandlerKeyedTensor
import torch
from models.blip import blip_decoder

model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_caption.pth'

class PreprocessBLIPInput(beam.DoFn):

  """
  Process the raw image input to a format suitable for BLIP inference. The processed
  images are duplicated to the number of desired captions per image. 

  Preprocessing transformation taken from: 
  https://github.com/salesforce/BLIP/blob/d10be550b2974e17ea72e74edc7948c9e5eab884/predict.py
  """

  def __init__(self, captions_per_image: int):
    self._captions_per_image = captions_per_image

  def setup(self):

    # Initialize the image transformer.
    self._transform = transforms.Compose([
      transforms.Resize((384, 384),interpolation=InterpolationMode.BICUBIC),
      transforms.ToTensor(),
      transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])

  def process(self, element):
    image_url, image = element 
    # The following lines provide a workaround to turn off BatchElements.
    preprocessed_img = self._transform(image).unsqueeze(0)
    preprocessed_img = preprocessed_img.repeat(self._captions_per_image, 1, 1, 1)
    # Parse the processed input to a dictionary to a format suitable for RunInference.
    preprocessed_dict = {'inputs': preprocessed_img}

    return [(image_url, preprocessed_dict)]

class PostprocessBLIPOutput(beam.DoFn):
  """
  Process the PredictionResult to get the generated image captions
  """
  def process(self, element : Tuple[str, Iterable[PredictionResult]]):
    image_url, prediction = element 

    return [(image_url, prediction.inference)]

class BLIPWrapper(torch.nn.Module):
  """
   Wrapper around the BLIP model to overwrite the default "forward" method with the "generate" method, because BLIP uses the 
  "generate" method to produce the image captions.
  """

  def __init__(self, base_model: blip_decoder, num_beams: int, max_length: int,
                min_length: int):
    super().__init__()
    self._model = base_model()
    self._num_beams = num_beams
    self._max_length = max_length
    self._min_length = min_length

  def forward(self, inputs: torch.Tensor):
    # Squeeze because RunInference adds an extra dimension, which is empty.
    # The following lines provide a workaround to turn off BatchElements.
    inputs = inputs.squeeze(0)
    captions = self._model.generate(inputs,
                                    sample=True,
                                    num_beams=self._num_beams,
                                    max_length=self._max_length,
                                    min_length=self._min_length)
    return [captions]

  def load_state_dict(self, state_dict: dict):
    self._model.load_state_dict(state_dict)