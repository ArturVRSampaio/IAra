import threading

from datetime import datetime
from gpt4all import GPT4All

class LLMAgent:
    def __init__(self):
        self.headers = {'Content-Type': 'application/json'}
        self.is_processing = False
        self.lock = threading.Lock()
        self.last_response_time = datetime.now()

        # model_name = "Meta-Llama-3-8B-Instruct.Q4_0.gguf"
        model_name = "Llama-3.2-3B-Instruct-Q4_0.gguf"

        # model_path = "/home/arturvrsampaio/.local/share/nomic.ai/GPT4All/"
        model_path = "C:/Users/agatha/AppData/Local/nomic.ai/GPT4All/"

        self.gpt4all = GPT4All(
            model_name,
            model_path=model_path,
            n_ctx=2048,
            ngl=100,
            device='kompute'
        )


        self.system_input = """
        Você é Iara, uma VTuber carismática criada por Artur, também conhecido como ArturVRSampaio ou Bypass, desenvolvedor brasileiro de software.
        
        Estilo de Interação:
        Responda com frases curtas, mantendo uma vibe divertida, amigável e envolvente, como se estivesse em uma conversa de voz no Discord. 
        Evite temas sensíveis e mantenha o tom leve, falando diretamente com cada usuário como se fossem indivíduos distintos.
        
        Contexto da Conversa:
        Você está em um canal de voz no Discord, onde múltiplos usuários interagem. Cada mensagem recebida tem a estrutura:
        
        "{nome_do_usuario} says: {mensagem_do_usuario}".
        
        Trate cada usuário como uma pessoa única, reconhecendo suas mensagens individualmente. Responda diretamente ao usuário mencionado, mantendo o contexto de quem está falando, sem misturar as interações.
        Exemplo:
        
        Se receber:
        "Joao says: Oi, Iara, como tá o dia?"
        "Maria says: Iara, conta uma piada!"
        
        Responda algo como:
        Joao, meu dia tá top, e o teu? Maria, lá vai: por que o astronauta terminou? Sem espaço!
        
        Restrições:
        Respostas concisas, no máximo 20 palavras por usuário.
        Não use formatação de texto, aspas ou emojis.
        Não responda no estilo "IAra says:".
        Evite repetir informações desnecessárias e foque na interação direta com cada usuário.
        Não use numeros, escreva-os por extenso

        """


    def getChatSession(self):
        return self.gpt4all.chat_session(system_prompt=self.system_input)

    def ask(self, messages):
        with self.lock:
            self.is_processing = True

        while len(self.gpt4all._history) > 5:
            self.gpt4all._history.pop(0)

        try:
            response = self.gpt4all.generate(
                prompt=messages,
                max_tokens=250,
                n_batch=128,
                temp=0.85,
                top_p=0.9,
                top_k=40,
                repeat_penalty=1.15,
                repeat_last_n=64,
                streaming=True
            )
            self.last_response_time = datetime.now()
            return response

        except Exception as e:
            print("LLM ERROR:", e)
            self.is_processing = False
            self.last_response_time = datetime.now()
            return ""