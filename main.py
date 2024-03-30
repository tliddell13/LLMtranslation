from wordGenerate import load_model, generate_text
import time

model_id = "/users/adbt150/archive/Llama-2-7b-hf"
tokenizer, model = load_model(model_id)

prompts = ["The dog", "The cat", "The bird", "The fish"]

for prompt in prompts:
    start_time = time.time()
    answer = generate_text(model, tokenizer, prompt, max_length=30)
    end_time = time.time()
    print(f"Prompt: {prompt}")
    print(f"Generated text: {answer}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print()
