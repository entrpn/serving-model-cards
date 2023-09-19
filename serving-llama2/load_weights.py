import os
import torch
import transformers

import logging
logger = logging.getLogger()
logging.basicConfig(level=logging.DEBUG)

DEFAULT_INSTRUCTION = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant.

If you are unsure about an answer, truthfully say "I don't know"
<</SYS>>

"""

HF_AUTH = os.getenv("HF_AUTH",None)
MODEL_ID = os.getenv("MODEL_ID","meta-llama/Llama-2-13b-chat-hf")
MAX_NEW_TOKENS = os.getenv("MAX_NEW_TOKENS",128)

INSTRUCTION = os.getenv("INSTRUCTION",DEFAULT_INSTRUCTION)

model_config = transformers.AutoConfig.from_pretrained(
    MODEL_ID,
    use_auth_token=HF_AUTH
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    config=model_config,
    device_map='auto',
    use_auth_token=HF_AUTH
)
model.eval()

tokenizer = transformers.AutoTokenizer.from_pretrained(
    MODEL_ID,
    use_auth_token=HF_AUTH
)

generate_text = transformers.pipeline(
    model=model, tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    # we pass model parameters here too
    temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=MAX_NEW_TOKENS,  # mex number of tokens to generate in the output
    repetition_penalty=1.1  # without this output begins repeating
)