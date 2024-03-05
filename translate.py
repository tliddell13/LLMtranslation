import pandas as pd
from transformers import AutoCausalLM, AutoTokenizer
# Load txt files into a pandas dataframe

# Specify the file path
english_path = "wmt07/dev/nc-dev2007.en"
spanish_path = "wmt07/dev/nc-dev2007.es"

# Load the text file into a dataframe
english = pd.read_csv(english_path, delimiter="\t", header=None)
spanish = pd.read_csv(spanish_path, delimiter="\t", header=None)

# Initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("/users/adbt150/archive/Llama-2-7b-hf")
model = AutoCausalLM.from_pretrained("/users/adbt150/archive/Llama-2-7b-hf")

# Create an empty DataFrame to store the responses
responses = pd.DataFrame(columns=['Response'])

# Iterate over the sentences in the english DataFrame
for index, row in english.iterrows():
    # Modify the sentence
    sentence = "Translate this to Spanish: " + row[0]
    # Encode the sentence
    inputs = tokenizer.encode(sentence, return_tensors="pt")
    # Generate a response
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1, temperature=0.7)
    # Decode the response
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Append the decoded response to the responses DataFrame
    responses = responses.append({'Response': decoded_output}, ignore_index=True)

print(responses.head())
# Save the responses to a CSV file
responses.to_csv("responses.csv", index=False)
