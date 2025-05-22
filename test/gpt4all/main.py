import time

from gpt4all import GPT4All

model_name = "Meta-Llama-3-8B-Instruct.Q4_0.gguf"

# agatha ->
model_path = "C:/Users/agath/AppData/Local/nomic.ai/GPT4All/"

# artur ->
# model_path = "/home/arturvrsampaio/.local/share/nomic.ai/GPT4All/"

gpt4all = GPT4All(model_name,
                model_path=model_path,
                device='kompute',
                )

text = "tell me a history about a big castle"
start = time.time()

response = gpt4all.generate(text,
                            max_tokens=200,
                            n_batch=16,
                            temp=0.8,
                            streaming=False)

end = time.time()
print("time: " + str(end - start))
print(response)

# for token in response:
#     print(token, end="", flush=True)
# print()