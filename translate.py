import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Specify the file path
english_path = "wmt07/dev/nc-dev2007.en"
spanish_path = "wmt07/dev/nc-dev2007.es"

# Load the text file into a dataframe
english = pd.read_csv(english_path, delimiter="\t", header=None)
spanish = pd.read_csv(spanish_path, delimiter="\t", header=None)

# Initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("/users/adbt150/archive/Llama-2-7b-hf")
model = AutoModelForSeq2SeqLM.from_pretrained("/users/adbt150/archive/Llama-2-7b-hf")

# Create a translation pipeline
translation_pipeline = pipeline('translation_en_to_es', model=model, tokenizer=tokenizer)

# Create an empty DataFrame to store the responses
responses = pd.DataFrame(columns=['Response'])

# Iterate over the sentences in the english DataFrame
for index, row in english.iterrows():
    # Translate the sentence
    translation = translation_pipeline(row[0])
    # Append the translated text to the responses DataFrame
    responses = responses.append({'Response': translation[0]['translation_text']}, ignore_index=True)

print(responses.head())
# Save the responses to a CSV file
responses.to_csv("responses.csv", index=False)

