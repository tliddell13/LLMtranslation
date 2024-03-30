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
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", load_in_8bit=True)
    # Print teh model device
    print("Model Device:", next(model.parameters()).device)
    return tokenizer, model

def generate_text(model, tokenizer, prompt, max_length=30):
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    answer = generator(prompt, max_length=max_length)
    return answer[0]['generated_text']







