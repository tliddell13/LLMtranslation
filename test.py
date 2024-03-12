import pandas as pd

english_path = "wmt07/dev/nc-dev2007.en"
spanish_path = "wmt07/dev/nc-dev2007.es"

english_df = pd.read_csv(english_path, delimiter="\t", header=None)
spanish_df = pd.read_csv(spanish_path, delimiter="\t", header=None)

print(english_df[:5])
# Loop through the dataframe and generate translations in batches
for idx, row in english_df[:5].iterrows():
    text = "Translate this to spanish: " + row[0]
    print(text)