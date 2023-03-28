import torch
import torch.nn.functional as F
from torch import nn

import os
from tqdm.auto import tqdm

from peft import prepare_model_for_int8_training
from peft import LoraConfig, get_peft_model

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch

from datasets import load_dataset
from bitsandbytes.optim import Adam8bit
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from accelerate.logging import get_logger

logger = get_logger(__name__, log_level="INFO")

def generate_prompt_for_eval(instruction, input=None):
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
    prompt = generate_prompt_for_eval(instruction, input)
    tokenizer.padding_side = "right"
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
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    tokenizer.padding_side = "left"
    return output.split("### Response:")[1].strip()

# Untested function.
def convert_8_to_16(layer : nn.Module):
    CxB = layer.state.CxB
    SCB = layer.state.SCB
    scb_size = SCB.size()
    cxb_size = CxB.size()
    # This overs fc_in and fc_out. Not been tested with other layers
    if cxb_size[1] != scb_size[0]:   
        SCB = SCB.reshape(1,-1).t()
    fp16_weights = ((CxB * SCB) / 127).to(torch.float16)
    if isinstance(layer,nn.Linear):
        new_layer = nn.Linear(layer.in_features, layer.out_features, dtype=weight_dtype)
    
    new_layer.weight = nn.Parameter(fp16_weights)
    new_layer.bias = layer.bias
    return new_layer

def save_model(model, output_dir, is_checkpoint=False, use_lora=False):
    # only saves one checkpoint right now.
    if is_checkpoint:
        output_dir = os.path.join(output_dir, "checkpoint")
    
    os.makedirs(output_dir,exist_ok=True)
    
    print("output_dir:",output_dir)
    if use_lora:
        model.save_pretrained(output_dir)
    else:    
        for _, module in model.named_modules():
            if 'GPTJAttention' in repr(type(module)):
                module.k_proj = convert_8_to_16(module.k_proj)
                module.v_proj = convert_8_to_16(module.v_proj)
                module.q_proj = convert_8_to_16(module.q_proj)
                module.out_proj = convert_8_to_16(module.out_proj)
            if 'GPTJMLP' in repr(type(module)):
                module.fc_in = convert_8_to_16(module.fc_in)
                module.fc_out = convert_8_to_16(module.fc_out)
        
        model.save_pretrained(output_dir)

# Load config
batch_size = os.getenv("BATCH_SIZE",2)
dataset_name = os.getenv("DATASET_NAME","transformersbook/codeparrot-train")
is_dataset_streaming = os.getenv("DATASET_STREAMING",True)
model_name = os.getenv("MODEL_NAME", "EleutherAI/gpt-j-6B")
model_revision = os.getenv("MODEL_REVISION","float16")
load_in_8bit = os.getenv("LOAD_8BIT",True)
enable_gradient_checkpointing = os.getenv("GRADIENT_CHECKPOINTING",True)
gradient_accumulation_steps = os.getenv("GRADIENT_ACCUMULATION_STEPS",4)
learning_rate = os.getenv("LEARNING_RATE",1e-5)
max_train_steps = os.getenv("MAX_TRAIN_STEPS",6000)
scale_lr = os.getenv("SCALE_LR",False)
checkpointing_steps = os.getenv("CHECKPOINTING_STEPS",1000)
use_lora = os.getenv("USE_LORA",True)
model_output_dir = os.getenv("MODEL_OUTPUT_DIR","output")
evaluate_steps = os.getenv("EVALUATE_STEPS",500)

weight_dtype=torch.float16

per_device_memory = torch.cuda.get_device_properties(0).total_memory
if per_device_memory <= 16e9:
    print("Device memory too low, using LoRA for finetuning")
    use_lora = True

# Set up accelerator
# Total limit is the limit of checkpoints to save.
# Older checkpoints are removed
accelerator_project_config = ProjectConfiguration(total_limit=5)

accelerator = Accelerator(
    gradient_accumulation_steps=gradient_accumulation_steps,
    mixed_precision="fp16",
    log_with="tensorboard",
    logging_dir="logs/",
    project_config=accelerator_project_config,
)

device = accelerator.device

# The trackers initializes automatically on the main process.
config = {
    "batch_size" : batch_size,
    "dataset_name:" : dataset_name,
    "is_dataset_streaming:" : is_dataset_streaming,
    "model_name" : model_name,
    "model_revision" : model_revision,
    "load_in_8bit" : load_in_8bit,
    "enable_gradient_checkpointing" : enable_gradient_checkpointing,
    "gradient_accumulation_steps" : gradient_accumulation_steps,
    "learning_rate" :learning_rate,
    "max_train_steps" : max_train_steps,
    "scale_ir" : scale_lr,
    "checkpointing_steps" : checkpointing_steps
}
if accelerator.is_main_process:
    accelerator.init_trackers("gptj-finetune", config=config)

# Load model

if load_in_8bit:
    print("loading in 8bit")
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", revision=model_revision, load_in_8bit=load_in_8bit)
else:
    print(f"loading in {model_revision}")
    model = AutoModelForCausalLM.from_pretrained(model_name, revision=model_revision, load_in_8bit=load_in_8bit,torch_dtype=weight_dtype)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Add pad token
new_tokens = ["<PAD>"]
# check if the tokens are already in the vocabulary
new_tokens = set(new_tokens) - set(tokenizer.vocab.keys())
# add the tokens to the tokenizer vocabulary
tokenizer.add_tokens(list(new_tokens))
# add new, random embeddings for the new tokens
model.resize_token_embeddings(len(tokenizer))
tokenizer.pad_token = "<PAD>" #tokenizer.eos_token
tokenizer.padding_side = "left"

if enable_gradient_checkpointing:
    model.gradient_checkpointing_enable()

# Support Lower memory card training using LoRA
if use_lora:

    def print_trainable_parameters(model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )
    
    config = LoraConfig(
        r=16, lora_alpha=32, target_modules=["k_proj", "q_proj", "v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    model = prepare_model_for_int8_training(model)
    # LoRAs usually require higher learning rate but this one worked best for me.
    learning_rate=1e-6
    model = get_peft_model(model, config)
    print_trainable_parameters(model)

# Test saving checkpoint before running
save_model(model, model_output_dir, is_checkpoint=True, use_lora=use_lora)

# Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.
if scale_lr:
    learning_rate = learning_rate * gradient_accumulation_steps * batch_size * accelerator.num_processes
    print("scaling learninig rate to: ",learning_rate)

# Optimizer
optimizer = Adam8bit(model.parameters(), lr=learning_rate)

# Load data
dataset = load_dataset("json",data_files="alpaca_data_cleaned.json")

# from https://github.com/tloen/alpaca-lora/blob/main/finetune.py
def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{data_point["instruction"]}
### Input:
{data_point["input"]}
### Response:
{data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{data_point["instruction"]}
### Response:
{data_point["output"]}"""

def encode(data_point):
    full_prompt = generate_prompt(data_point)
    retval = tokenizer(full_prompt, padding='max_length', truncation=True, max_length=255, return_tensors='pt')
    input_ids = torch.cat((retval.input_ids,torch.tensor([[tokenizer.eos_token_id]],dtype=retval.input_ids.dtype)),1)
    attention_mask = torch.cat((retval.attention_mask,torch.tensor([[1]],dtype=retval.attention_mask.dtype)),1)
    retval = {
        "input_ids" : input_ids,
        "attention_mask" : attention_mask
    }
    return retval

dataset = dataset["train"].shuffle().map(encode,remove_columns=["instruction","input","output"], batched=True, batch_size=batch_size)

# cached datasets files don't keep torch tensors, so must add this line
dataset.set_format("pt", columns=["input_ids","attention_mask"], output_all_columns=True)

train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

model, optimizier, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

print("Evaluate before training")
if accelerator.is_main_process:
    print("Response:", evaluate(model, tokenizer, "Generate a list of five things one should keep in mind when considering a career change."))
    print("\n\n")
# Train
global_step = 0

progress_bar = tqdm(range(global_step,max_train_steps), disable=not accelerator.is_local_main_process)
progress_bar.set_description("Steps")
for epoch in range(100000):
    train_loss = 0.0
    for batch in train_dataloader:
        with accelerator.accumulate(model):
            out = model.forward(**batch)
            loss = F.cross_entropy(out.logits[:, :-1, :].flatten(0, -2), batch['input_ids'][:, 1:].flatten(), 
                reduction='mean')
            
            # Gather the losses across all processes for logging (if we use distributed training).
            # https://huggingface.co/docs/accelerate/v0.17.1/en/package_reference/accelerator#accelerate.Accelerator.gather
            avg_loss = accelerator.gather(loss.repeat(batch_size)).mean()
            train_loss += avg_loss.item() / gradient_accumulation_steps
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
        
            # Checks if the accelerator has performed an optimization step behind the scenes. Ex: If accumulation steps == 4, then this will get called every 4 itter.
            if accelerator.sync_gradients:
                global_step+=1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                progress_bar.update(1)
                progress_bar.set_description(f"loss: {train_loss}, batch_size: {batch_size * gradient_accumulation_steps}, steps: {global_step}")
                train_loss = 0.0
                if global_step % checkpointing_steps == 0 and accelerator.is_main_process:
                    print("saving checkpoint:",global_step)
                    save_path = os.path.join("checkpoints", f"checkpoint-{global_step}")
                    save_model(model, output_dir=model_output_dir,
                    use_lora=use_lora, is_checkpoint=True)
                    logger.info(f"Saved state to {save_path}")
                if global_step % evaluate_steps == 0 and accelerator.is_main_process:
                    print("Response:", evaluate(model, tokenizer, "Generate a list of five things one should keep in mind when considering a career change."))

            if global_step >= max_train_steps:
                print(f"breaking from loop")
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    save_model(model,output_dir=model_output_dir,use_lora=use_lora, is_checkpoint=False)
                    exit()