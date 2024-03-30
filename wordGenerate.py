import pandas as pd
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    PreTrainedTokenizerFast,
)
import torch
from torch.utils.data import Dataset       

def load_model(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    # See if fast tokenizer helps
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_object=tokenizer)
    return tokenizer, model

def generate_text(model, tokenizer, prompt, max_length=30):
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    answer = generator(prompt, max_length=max_length)
    return answer[0]['generated_text']

model_id = "/users/adbt150/archive/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

# Use the pipeline to generate text
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompt = "Write a sentence with the word 'dog' in it."

# Generate text
answer = generator(prompt, max_length=30)
print(answer[0]['generated_text'])







