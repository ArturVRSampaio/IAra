import time

import torch
from gpt4all import GPT4All

torch.set_num_threads(14)
torch.set_default_device("cuda")




model_name = "Meta-Llama-3-8B-Instruct.Q4_0.gguf"

# agatha ->
# model_path = "C:/Users/agath/AppData/Local/nomic.ai/GPT4All/"
# artur ->
model_path = "C:/Users/ArturVRSampaio//AppData/Local/nomic.ai/GPT4All/"

gpt4all = GPT4All(
            model_name,
            model_path=model_path,
            n_ctx=2048,
            ngl=100,
            device='cuda'
        )

text = "como voce esta?"
start = time.time()

response = gpt4all.generate(
                text,
                max_tokens=250,
                n_batch=128,
                temp=0.75,
                top_p=0.8,
                top_k=40,
                repeat_penalty=1.30,
                repeat_last_n=128,
                streaming=False
            )

end = time.time()
print("time: " + str(end - start))
print(response)

# for token in response:
#     print(token, end="", flush=True)
# print()