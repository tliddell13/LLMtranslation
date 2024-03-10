import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

class TranslationDataset(torch.utils.dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file, delimiter="\t", header=None)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return "Translate this to spanish: " + self.data.iloc[idx, 0]

# Specify the file path
english_path = "wmt07/dev/nc-dev2007.en"
spanish_path = "wmt07/dev/nc-dev2007.es"

# Load the text file into a dataframe
english = pd.read_csv(english_path, delimiter="\t", header=None)
spanish = pd.read_csv(spanish_path, delimiter="\t", header=None)

# Initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("/users/adbt150/archive/Llama-2-7b-hf")
model = AutoModelForCausalLM.from_pretrained("/users/adbt150/archive/Llama-2-7b-hf")

# Check if a GPU is available and if not, use a CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move the model to the device
model.to(device)

# Then, when you create the pipeline:
dataset = TranslationDataset(english_path)
translation_pipeline = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

# And when you generate the translations:
responses = translation_pipeline(dataset)

# Convert the list to a DataFrame
responses_df = pd.DataFrame(responses, columns=['Response'])

# Save the responses to a CSV file
responses.to_csv("responses.csv", index=False)

