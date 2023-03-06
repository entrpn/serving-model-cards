# from apache_beam.internal import pickler
# pickler.set_library(pickler.USE_CLOUDPICKLE)
import argparse
import apache_beam as beam
from apache_beam.ml.inference.base import KeyedModelHandler
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions

from image_utils import ReadImagesFromUrl, ReadImagesFromGcsUrl, FormatCaptions

from blip_processing import PreprocessBLIPInput, PostprocessBLIPOutput, BLIPWrapper
from models.blip import blip_decoder

from clip_processing import PreprocessCLIPInput, RankCLIPOutput, CLIPWrapper
from transformers import CLIPProcessor
from transformers import CLIPTokenizer
from transformers import CLIPModel
from transformers import CLIPConfig
from transformers import CLIPFeatureExtractor

from apache_beam.ml.inference.base import RunInference
from model_handler import PytorchModelHandlerKeyedTensor, PytorchNoBatchModelHandlerKeyedTensor

def main(parser, save_main_session=True):

    # Increasing Beam search might improve the quality of the captions,
    # but also results in more compute time
    NUM_BEAMS = 5
    # Number of captions generated per image.
    NUM_CAPTIONS_PER_IMAGE = 10

    # Number of top captions to display.
    NUM_TOP_CAPTIONS_TO_DISPLAY = 3

    clip_feature_extractor_config_path = '/captioning/clip-vit-base-patch32/preprocessor_config.json'
    clip_tokenizer_vocab_config_path = '/captioning/clip-vit-base-patch32/vocab.json'
    clip_merges_config_path = '/captioning/clip-vit-base-patch32/merges.txt'
    clip_model_config_path = '/captioning/clip-vit-base-patch32/config.json'
    clip_state_dict_path = '/captioning/clip-vit-base-patch32/pytorch_model.bin'

    known_args, pipeline_args = parser.parse_known_args()
    
    dataset_filename = known_args.dataset_filename
    output_filename = known_args.output_filename

    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = save_main_session

    CLIP_model_handler = PytorchNoBatchModelHandlerKeyedTensor(
    state_dict_path=clip_state_dict_path,
    model_class=CLIPWrapper,
    model_params={'config': CLIPConfig.from_pretrained(clip_model_config_path)},
    device='GPU')

    CLIP_keyed_model_handler = KeyedModelHandler(CLIP_model_handler)

    blip_state_dict_path = 'blip_state_dict.pth'

    MAX_CAPTION_LENGTH = 80
    MIN_CAPTION_LENGTH = 10
    # Increasing Beam search might improve the quality of the captions,
    # but also results in more compute time
    NUM_BEAMS = 1

    BLIP_model_handler = PytorchNoBatchModelHandlerKeyedTensor(
        state_dict_path=blip_state_dict_path,
        model_class=BLIPWrapper,
        model_params={'base_model': blip_decoder, 'num_beams': NUM_BEAMS,
                    'max_length': MAX_CAPTION_LENGTH, 'min_length': MIN_CAPTION_LENGTH},
        device='GPU')

    BLIP_keyed_model_handler = KeyedModelHandler(BLIP_model_handler)

    with beam.Pipeline(options=pipeline_options) as pipeline:


        read_images = (
                    pipeline 
                    | "ReadUrl" >> beam.io.ReadFromText(dataset_filename)
                    | "ReadImages" >> beam.ParDo(ReadImagesFromGcsUrl()))

        blip_caption_generation = (
                    read_images
                    | "PreprocessBlipInput" >> beam.ParDo(PreprocessBLIPInput(NUM_CAPTIONS_PER_IMAGE)) 
                    | "GenerateCaptions" >> RunInference(BLIP_keyed_model_handler)
                    | "PostprocessCaptions" >> beam.ParDo(PostprocessBLIPOutput()))

        clip_captions_ranking = (
                    ({'image' : read_images, 'captions': blip_caption_generation})
                    | "CreateImageCaptionPair" >> beam.CoGroupByKey()
                    | "PreprocessClipInput" >> beam.ParDo(
                        PreprocessCLIPInput(
                            clip_feature_extractor_config_path,
                            clip_tokenizer_vocab_config_path,
                            clip_merges_config_path))
                    | "GetRankingLogits" >> RunInference(CLIP_keyed_model_handler)
                    | "RankClipOutput" >> beam.ParDo(RankCLIPOutput()))

        results = (clip_captions_ranking | "FormatCaptions" >> beam.ParDo(FormatCaptions(NUM_TOP_CAPTIONS_TO_DISPLAY)))
        
        results | "Write Results" >> beam.io.WriteToText(output_filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass your arguments")
    parser.add_argument(
        "--dataset-filename",
        type=str,
        required=True,
        help="Dataset filename location. Ex: gs://<project-id>/dataset.txt"
    )
    parser.add_argument(
        "--output-filename",
        type=str,
        required=True,
        help="Write output training file. Ex: gs://<project-id>/metadata.jsonl"
    )
    main(parser)