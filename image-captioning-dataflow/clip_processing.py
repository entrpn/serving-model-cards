from typing import Iterator
from typing import Iterable
from typing import Tuple
from typing import Optional
from typing import Dict
from typing import List
from typing import Any

import torch
import apache_beam as beam
import numpy as np
from apache_beam.ml.inference.base import PredictionResult
from transformers import CLIPProcessor
from transformers import CLIPTokenizer
from transformers import CLIPModel
from transformers import CLIPConfig
from transformers import CLIPFeatureExtractor

class PreprocessCLIPInput(beam.DoFn):

  """
  Process the image-caption pair to a format suitable for CLIP inference. 

  After grouping the raw images with the generated captions, we need to 
  preprocess them before passing them to the ranking stage (CLIP model).
  """

  def __init__(self,
               feature_extractor_config_path: str,
               tokenizer_vocab_config_path: str,
               merges_file_config_path: str):

    self._feature_extractor_config_path = feature_extractor_config_path
    self._tokenizer_vocab_config_path = tokenizer_vocab_config_path 
    self._merges_file_config_path = merges_file_config_path


  def setup(self):

    # Initialize the CLIP feature extractor.
    feature_extractor_config = CLIPConfig.from_pretrained(self._feature_extractor_config_path)
    feature_extractor = CLIPFeatureExtractor(feature_extractor_config)

    # Initialize the CLIP tokenizer.
    tokenizer = CLIPTokenizer(self._tokenizer_vocab_config_path,
                              self._merges_file_config_path)

    # Initialize the CLIP processor used to process the image-caption pair.
    self._processor = CLIPProcessor(feature_extractor=feature_extractor,
                                    tokenizer=tokenizer)

  def process(self, element: Tuple[Any, Dict[str, Iterable[Any]]]):

    image_url, image_captions_pair = element 
    # Unpack the image and captions after grouping them with 'CoGroupByKey()'.
    image = image_captions_pair['image'][0]
    captions = image_captions_pair['captions'][0]
    preprocessed_clip_input = self._processor(images = image,
                                              text = captions,
                                              return_tensors="pt",
                                              padding=True)

    image_url_caption_pair = (image_url, captions)
    return [(image_url_caption_pair, preprocessed_clip_input)]


class RankCLIPOutput(beam.DoFn):
  """
  Process the output of CLIP to get the captions sorted by ranking order.

  The logits are the output of the CLIP model. Here, we apply a softmax activation
  function to the logits to get the probabilistic distribution of the relevance
  of each caption to the target image. After that, we sort the captions in descending
  order with respect to the probabilities as a caption-probability pair. 
  """

  def process(self, element : Tuple[Tuple[str, List[str]], Iterable[PredictionResult]]):
    (image_url, captions), prediction = element
    prediction_results = prediction.inference
    prediction_probs = prediction_results.softmax(dim=-1).cpu().detach().numpy()
    ranking = np.argsort(-prediction_probs)
    sorted_caption_prob_pair = [(captions[idx], prediction_probs[idx]) for idx in ranking]

    return [(image_url, sorted_caption_prob_pair)]

class CLIPWrapper(CLIPModel):

  def forward(self, **kwargs: Dict[str, torch.Tensor]):
    # Squeeze because RunInference adds an extra dimension, which is empty.
    # The following lines provide a workaround to turn off BatchElements.
    kwargs = {key: tensor.squeeze(0) for key, tensor in kwargs.items()}
    output = super().forward(**kwargs)
    logits = output.logits_per_image
    return logits