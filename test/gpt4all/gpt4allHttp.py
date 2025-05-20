import requests
import json
import time

# Informações do servidor GPT4All
url = "http://localhost:4891/v1/chat/completions"
headers = {
    'Content-Type': 'application/json'
}

# Sua mensagem
text = "tell me a history about a big castle"

# Dados para enviar na requisição POST
data = {
    "model": "Llama-3.2-3B-Instruct-Q4_0.gguf",
    "messages": [{"role": "user", "content": text}],
    "max_tokens": 200,
    "n": 1,
    "temperature": 0.8,
    "stream": False
}

start = time.time()

response = requests.post(url, headers=headers, data=json.dumps(data))
response.raise_for_status()
end = time.time()
print('Resposta recebida do servidor:')
print('time: ' + str(end - start))
result = response.json()
print(result['choices'][0]['message']['content'])