import pandas as pd
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline
)
import torch
from torch.utils.data import Dataset       

# Specify the file path
english_path = "wmt07/dev/nc-dev2007.en"
spanish_path = "wmt07/dev/nc-dev2007.es"

# Load the text file into a dataframe
english_df = pd.read_csv(english_path, delimiter="\t", header=None)
spanish_df = pd.read_csv(spanish_path, delimiter="\t", header=None)

def load_model(model_name):
    # Load Model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        load_in_8bit = True
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

english_df = english_df[0]

# Loop through the dataframe and generate translations in batches
for row in english_df.iterrows():
    text = "Translate this to spanish: " + row[0]
    print(text)
    response = translation_pipeline(text)
    print(response)
    responses.append(response[0]['generated_text'])

# Convert the list to a DataFrame
responses_df = pd.DataFrame({'Response': responses})

# Save the responses to a CSV file
responses.to_csv("responses.csv", index=False)

