import pandas as pd
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline
)
import torch
from torch.utils.data import Dataset       

model_id = "/users/adbt150/archive/Llama-2-13b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

# Use the pipeline to generate text
generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

prompt = "Write a sentence with the word 'dog' in it."

# Generate text
answer = generator(prompt, max_length=30)
print(answer[0]['generated_text'])







