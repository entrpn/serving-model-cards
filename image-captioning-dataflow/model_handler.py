
from apache_beam.ml.inference.pytorch_inference import PytorchModelHandlerKeyedTensor

class PytorchNoBatchModelHandlerKeyedTensor(PytorchModelHandlerKeyedTensor):
      """Wrapper to PytorchModelHandler to limit batch size to 1.
    The caption strings generated from the BLIP tokenizer might have different
    lengths. Different length strings don't work with torch.stack() in the current RunInference
    implementation, because stack() requires tensors to be the same size.
    Restricting max_batch_size to 1 means there is only 1 example per `batch`
    in the run_inference() call.
    """
      # The following lines provide a workaround to turn off BatchElements.
      def batch_elements_kwargs(self):
          return {'max_batch_size': 1}