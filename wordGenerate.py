import pandas as pd
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
import torch
from torch.utils.data import Dataset       

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

def load_model(model_name):
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    # Load Model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer

# Initialize the tokenizer and model
model, tokenizer = load_model("/users/adbt150/archive/Llama-2-7b-hf")

print("Model Device:", next(model.parameters()).device)

translation_pipeline = pipeline('text-generation', model=model, tokenizer=tokenizer)

# Initialize a list to store responses
responses = []

prompt = "Generate a sentence with the word 'dog' in it."

pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=30)
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]['generated_text'])

"""
Some modules are dispatched on the CPU or the disk. Make sure you have enough GPU RAM to fit the
                    quantized model. If you want to dispatch the model on the CPU or the disk while keeping these modules
                    in 32-bit, you need to set `load_in_8bit_fp32_cpu_offload=True` and pass a custom `device_map` to
                    `from_pretrained`. Check
                    https://huggingface.co/docs/transformers/main/en/main_classes/quantization#offload-between-cpu-and-gpu
                    for more details.


"""

