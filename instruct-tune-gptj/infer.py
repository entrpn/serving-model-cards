
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer


peft_model_id = "output"
model_name = "EleutherAI/gpt-j-6B"
config = PeftConfig.from_pretrained(peft_model_id)
print("loading model")
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto",revision="float16", load_in_8bit=True)
print("loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load the Lora model
print("loading lora model")
model = PeftModel.from_pretrained(model, peft_model_id).to("cuda")

batch = tokenizer("Two things are infinite: ", return_tensors="pt")
batch = {key: value.to("cuda") for key, value in batch.items()}

with torch.cuda.amp.autocast():
    output_tokens = model.generate(**batch, max_new_tokens=50)

print("\n\n", tokenizer.decode(output_tokens[0], skip_special_tokens=False))