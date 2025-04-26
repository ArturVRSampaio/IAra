from gpt4all import GPT4All

model_name = "Llama-3.2-1B-Instruct-Q4_0.gguf"
model_path = "/home/arturvrsampaio/.local/share/nomic.ai/GPT4All/"

gpt4all = GPT4All(model_name, model_path=model_path)

text = "tell me a history about a big castle"

response = gpt4all.generate(text, max_tokens=200, n_batch=60, temp=0.8, streaming=True)


for token in response:
    print(token, end="", flush=True)
print()