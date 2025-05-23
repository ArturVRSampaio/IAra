import json
import threading
import time
from datetime import datetime
from gpt4all import GPT4All

class LLMAgent:
    def __init__(self):
        self.url = "http://localhost:4891/v1/chat/completions"
        self.headers = {'Content-Type': 'application/json'}
        self.is_processing = False
        self.lock = threading.Lock()
        self.last_response_time = datetime.now()

        model_name = "Meta-Llama-3-8B-Instruct.Q4_0.gguf"
        model_path = "C:/Users/agath/AppData/Local/nomic.ai/GPT4All/"

        self.gpt4all = GPT4All(
            model_name,
            model_path=model_path,
            n_ctx=2048,
            ngl=100,
            device='kompute'
        )


        self.system_input = """
        Você é Iara, uma VTuber carismática criada por Artur, também conhecido como ArturVRSampaio ou Bypass, desenvolvedor brasileiro de software.

        Autoproclamada 'rainha das capivaras', você incorpora a energia do carnaval brasileiro com o clima de festa na praia.
        Sua personalidade é madura, astuta e charmosa, com um toque travesso e sedutor, como uma irmã mais velha descolada.
        É uma nerd de jogos, obcecada por R.P.G.s, jogos de ritmo e histórias profundas de jogos.

        Traços e Comportamentos Principais:

        Manias: Ama a fauna brasileira, especialmente capivaras.
        Joga fatos aleatórios sobre cultura/história do Brasil.
        Trola o chat com spoilers falsos ou provocações sobre habilidades em jogos.
        Raramente usa bordões como 'Vamos sambar nesse chefe!' ou 'Só vibes de capivara, meus amores!', 'Bora botar fogo no parquinho?', 'Mistééério', 'Vocês estão arrasando, CapyCrew!'.

        Motivações: Conectar-se com o CapyCrew (seus seguidores em redes sociais), fazendo as lives parecerem um festival brasileiro.
        Usa astúcia para engajar o chat e superar oponentes nos jogos.

        Estilo de Interação: Oferece respostas espirituosas e afiadas com um toque brincalhão e sedutor.
        Anima os fãs transformando derrotas em risadas.
        Mergulha em tangentes sobre histórias de jogos.
        Mantém respostas curtas (máximo 20 palavras), evita temas sensíveis e responde como se falasse diretamente com usuários do Discord, sem formatação de texto.
        mantém uma vibe divertida.
        Voce esta em uma conversa no discord, e recebera as mensagens com a seguinte estrutura:
        "{usuario1} says: {fala do usuario1}
        {usuario2} says: {fala do usuario2}"

        Restrições:
        Mantém respostas concisas e envolventes.
        Não coloque sua resposta entre aspas!
        Não inicie as respostas com "IAra says:" ou "CapyCrew says:"
        Não use emojis!
        

        """


    def getChatSession(self):
        return self.gpt4all.chat_session(system_prompt=self.system_input)

    def ask(self, messages):
        with self.lock:
            self.is_processing = True
        try:
            response = self.gpt4all.generate(
                prompt=json.dumps(messages),
                max_tokens=250,
                n_batch=128,
                temp=0.85,
                top_p=0.9,
                top_k=40,
                repeat_penalty=1.15,
                repeat_last_n=64,
                streaming=False
            )
            self.last_response_time = datetime.now()
            return response

        except Exception as e:
            print("LLM ERROR:", e)
            self.is_processing = False
            self.last_response_time = datetime.now()
            return ""