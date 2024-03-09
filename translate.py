import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

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

# Create a translation pipeline
translation_pipeline = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

# Create an empty list to store the response
responses = []

# Iterate over the sentences in the english DataFrame
for index, row in english.iterrows():
    # Move the sentence to the device
    sentence = "Translate this to spanish: " + row[0]
    # Translate the sentence
    translation = translation_pipeline(sentence)
    # Append the translated text to the responses DataFrame
    responses.append(translation[0]['generated_text'])

# Convert the list to a DataFrame
responses_df = pd.DataFrame(responses, columns=['Response'])

# Save the responses to a CSV file
responses.to_csv("responses.csv", index=False)



