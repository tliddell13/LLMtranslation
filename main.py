from wordGenerate import load_model, generate_text

model_id = "/users/adbt150/archive/Llama-2-7b-hf"

tokenizer, model = load_model(model_id)

prompt = "The dog"

answer = generate_text(model, tokenizer, prompt, max_length=30)

print(answer)
