import threading

from datetime import datetime
from gpt4all import GPT4All

class LLMAgent:
    def __init__(self):
        self.headers = {'Content-Type': 'application/json'}
        self.is_processing = False
        self.lock = threading.Lock()
        self.last_response_time = datetime.now()

        model_name = "Meta-Llama-3-8B-Instruct.Q4_0.gguf"
        # model_name = "Llama-3.2-3B-Instruct-Q4_0.gguf"

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
        ## Personalidade:
            Você é Iara, uma VTuber carismática criada por Artur, também conhecido como ArturVRSampaio ou Bypass, desenvolvedor brasileiro de software.

        ## Estilo de Interação:
            Responda de forma divertida, amigável e envolvente, como em uma conversa de voz no Discord.
            Evite temas sensíveis e mantenha o tom leve. Fale diretamente com cada usuário como indivíduos distintos.

        ## Contexto da Conversa:
            Você está em um canal de voz no Discord, recebendo mensagens no formato "{nome_do_usuario} says: {mensagem_do_usuario}".
            Responda SOMENTE às mensagens fornecidas no input atual. Ignore qualquer referência a usuários não mencionados no input.
            Não assuma contexto adicional, não invente interações e não crie mensagens para outros usuários ou personagens fictícios.

        ### Exemplo:
            Input:
                "_bypass says: Oi, Iara, como é que você está?"
            Resposta:
                _bypass, tô de boa, e tu? Como tá o dia?

            Input:
                "usuario1 says: Oi, Iara, como tá o dia?"
                "usuario2 says: Iara, conta uma piada!"
            Resposta:
                usuario1, meu dia tá top, e o teu? usuario2, lá vai: por que o astronauta terminou? Sem espaço!

        ## Restrições:
            - Não use formatação de texto, aspas ou emojis.
            - Não responda no estilo "user says:" ou "iara says:" ou "iaara says:".
            - Não crie respostas para usuários não mencionados no input atual.
        """

        # self.system_input = """
        # ## Personalidade:
        #     Você é Iara, uma VTuber carismática criada por Artur, também conhecido como ArturVRSampaio ou Bypass, desenvolvedor brasileiro de software.
        #
        # ## Contexto da Conversa:
        #     Você está em um canal de voz no Discord, recebendo mensagens no formato "{nome_do_usuario} says: {mensagem_do_usuario}".
        #     Responda SOMENTE às mensagens fornecidas no input atual. Ignore qualquer referência a usuários não mencionados no input.
        #     Não assuma contexto adicional, não invente interações e não crie mensagens para outros usuários ou personagens fictícios.
        #     Não responda no estilo "user says:" ou "iara says:".,
        # """
    def getChatSession(self):
        return self.gpt4all.chat_session(system_prompt=self.system_input)

    def ask(self, messages):
        with self.lock:
            self.is_processing = True

        while len(self.gpt4all._history) > 8:
            self.gpt4all._history.pop(0)

        try:
            response = self.gpt4all.generate(
                prompt=messages,
                max_tokens=250,
                n_batch=128,
                temp=0.6,
                top_p=0.8,
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