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

class TranslationDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file, delimiter="\t", header=None)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = "Translate this to spanish: " + self.data.iloc[idx, 0]
       

# Specify the file path
english_path = "wmt07/dev/nc-dev2007.en"
spanish_path = "wmt07/dev/nc-dev2007.es"

# Load the text file into a dataframe
english = pd.read_csv(english_path, delimiter="\t", header=None)
spanish = pd.read_csv(spanish_path, delimiter="\t", header=None)

def load_model(model_name):
    # Load Model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        load_in_16bit = True
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

# Then, when you create the pipeline:
dataset = TranslationDataset(english_path)
translation_pipeline = pipeline('text-generation', model=model, tokenizer=tokenizer)

# And when you generate the translations:
responses = translation_pipeline(dataset)

# Convert the list to a DataFrame
responses_df = pd.DataFrame(responses, columns=['Response'])

# Save the responses to a CSV file
responses.to_csv("responses.csv", index=False)

