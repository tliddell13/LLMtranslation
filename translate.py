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

# Specify the file path
english_path = "wmt07/dev/nc-dev2007.en"
spanish_path = "wmt07/dev/nc-dev2007.es"

# Load the text file into a dataframe
english_df = pd.read_csv(english_path, delimiter="\t", header=None)
spanish_df = pd.read_csv(spanish_path, delimiter="\t", header=None)

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

prompt = "Translate this to spanish: How are you?"

pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=30)
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]['generated_text'])
"""
# Loop through the dataframe and generate translations in batches
for idx, row in english_df.iterrows():
    text = "Translate this to spanish: " + row[0]
    print(text)
    response = translation_pipeline(text, max_length=50)
    print(response)
    responses.append(response[0]['generated_text'])

# Convert the list to a DataFrame
responses_df = pd.DataFrame({'Response': responses})

# Save the responses to a CSV file
responses.to_csv("responses.csv", index=False)
"""
