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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = "Translate this to spanish: " + self.data.iloc[idx, 0]
        return text.to(self.device)
       

# Specify the file path
english_path = "wmt07/dev/nc-dev2007.en"
spanish_path = "wmt07/dev/nc-dev2007.es"

# Load the text file into a dataframe
english = pd.read_csv(english_path, delimiter="\t", header=None)
spanish = pd.read_csv(spanish_path, delimiter="\t", header=None)

# Activate 4-bit precision base model loading
use_4bit = True
# Activate nested quantization for 4-bit base models
use_nested_quant = False
# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"
# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"
# Load the entire model on the GPU 0
device_map = {"": 0}

def load_model(model_name):
    # Load tokenizer and model with QLoRA configuration
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )

    if compute_dtype == torch.float16 and use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
            print("=" * 80)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
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

model.to("cuda")

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

