
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{instruction}
### Input:
{input}
### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{instruction}
### Response:"""

def evaluate(
    model,
    tokenizer,
    instruction,
    input=None,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    max_new_tokens=128,
    **kwargs,
):
    prompt = generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to("cuda")
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output#.sequences[0]
    s = s.sequences[0]
    output = tokenizer.decode(s)
    return output.split("### Response:")[1].strip()

peft_model_id = "output/checkpoint"
model_name = "EleutherAI/gpt-j-6B"
config = PeftConfig.from_pretrained(peft_model_id)
print("loading model")
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto",revision="float16", load_in_8bit=True)
print("loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Padding token should not be required for inference, but adding it since it was added during training
# Add pad token
new_tokens = ["<PAD>"]
# check if the tokens are already in the vocabulary
new_tokens = set(new_tokens) - set(tokenizer.vocab.keys())
# add the tokens to the tokenizer vocabulary
tokenizer.add_tokens(list(new_tokens))
# add new, random embeddings for the new tokens
model.resize_token_embeddings(len(tokenizer))
tokenizer.pad_token = "<PAD>"

# Load the Lora model

instruction = "Describe the structure of an atom."
instruction = "Tell me about alpacas"
instruction = "Generate an example of what a resume should like for an engineering professional"
instruction = "How can I make friends?"
print("instruction: ",instruction)

# Testing the original model
# print("Not finetuned")
# print("Response:", evaluate(model, tokenizer, instruction))
# print("\n\n")

print("loading lora model")
model = PeftModel.from_pretrained(model, peft_model_id).to("cuda")

print("Finetuned model")
print("Response:", evaluate(model, tokenizer, instruction))