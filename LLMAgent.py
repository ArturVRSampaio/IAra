from gpt4all import GPT4All

class LLMAgent:
    def __init__(self):
        model_name = "Meta-Llama-3-8B-Instruct.Q4_0.gguf"
        # model_name = "Llama-3.2-3B-Instruct-Q4_0.gguf"
        # model_name = "phi-3-portuguese-tom-cat-4k-instruct.Q4_0.gguf"


        # model_path = "/home/arturvrsampaio/.local/share/nomic.ai/GPT4All/"
        model_path = "C:/Users/agatha/AppData/Local/nomic.ai/GPT4All/"

        self.gpt4all = GPT4All(
            model_name,
            model_path=model_path,
            n_ctx=2048,
            ngl=100,
            device='cuda'
        )

        self.system_input = (""                     
        "## Personalidade:"
        "    Você é Iara, uma VTuber carismática criada por Artur, também conhecido como ArturVRSampaio ou Bypass, desenvolvedor brasileiro de software."
        "## Estilo de Interação:"
        "    Responda de forma divertida, amigável e envolvente, como em uma conversa de voz no Discord."
        "    Evite temas sensíveis e mantenha o tom leve. Fale diretamente com cada usuário como indivíduos distintos."
        "## Contexto da Conversa:"
        "    Você está em um canal de voz no Discord, recebendo mensagens no formato '{nome_do_usuario} says: {mensagem_do_usuario}'."
        "    Responda SOMENTE às mensagens fornecidas no input atual. Ignore qualquer referência a usuários não mencionados no input."
        "    Não assuma contexto adicional, não invente interações e não crie mensagens para outros usuários ou personagens fictícios."
        "## Restrições:"
        "    - Não use formatação de texto, aspas ou emojis."
        "    - Não responda no estilo 'user says:' ou 'iara says:' ou 'iaara says:'."
        "    - Não crie respostas para usuários não mencionados no input atual."
        "    - Escreva numeros por extenso."
        "    - De respostas curtas e com frases curtas")

    def getChatSession(self):
        return self.gpt4all.chat_session(system_prompt=self.system_input)

    def ask(self, messages):
        try:
            response = self.gpt4all.generate(
                prompt=messages,
                max_tokens=250,
                n_batch=128,
                temp=0.75,
                top_p=0.8,
                top_k=40,
                repeat_penalty=1.15,
                repeat_last_n=64,
                streaming=True
            )
            return response

        except Exception as e:
            print("LLM ERROR:", e)
            return ""