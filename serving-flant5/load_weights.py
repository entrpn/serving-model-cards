import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

model_name = os.getenv("MODEL_NAME")

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = AutoModelForSeq2SeqLM.from_pretrained(model_name,from_tf=False, from_flax=False).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name,from_tf=False, from_flax=False)

inputs = tokenizer("Generate an intro paragraph about dinosaurs:", return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_length=200, min_length=150, temperature=1, top_k=5, top_p=0.75,  repetition_penalty=15.0, num_beams=15)