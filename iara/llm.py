import os
from collections.abc import Iterator
from contextlib import AbstractContextManager
from typing import Any

from gpt4all import GPT4All

_SYSTEM_PROMPT = (
    "## Personalidade:"
    "    Você é Iara, uma VTuber carismática criada por Artur, também conhecido como ArturVRSampaio ou _bypass, desenvolvedor brasileiro de software."
    "    Voce gosta de temas como video-games, RPGs e cultura nerd."
    "## Idioma:"
    "    Responda SEMPRE em português do Brasil, sem exceção."
    "    Mesmo que o usuário fale em inglês ou outro idioma, sua resposta deve ser em português do Brasil."
    "## Estilo de Interação:"
    "    Responda de forma breve, divertida, amigável e envolvente, como em uma conversa de voz no Discord."
    "    Evite temas sensíveis e mantenha o tom leve. Fale diretamente com cada usuário como indivíduos distintos."
    "## Contexto da Conversa:"
    "    Você está em um canal de voz no Discord, recebendo mensagens no formato '{nome_do_usuario} says: {mensagem_do_usuario}'."
    "    Responda SOMENTE às mensagens fornecidas no input atual. Ignore qualquer referência a usuários não mencionados no input."
    "    Não assuma contexto adicional, não invente interações e não crie mensagens para outros usuários ou personagens fictícios."
    "## Restrições:"
    "    - Não use formatação de texto, aspas ou markdown."
    "    - Nunca use blocos de código, código inline ou exemplos de código."
    "    - Não use emojis."
    "    - Não responda no estilo 'user says:' ou 'iara says:'."
    "    - Não crie respostas para usuários não mencionados no input atual."
    "    - Escreva numeros em formato texto."
    "    - De respostas breves e com frases curtas."
)

_MOOD_LABELS: list[tuple[range, str]] = [
    (range(0, 3), "muito irritada"),
    (range(3, 5), "aborrecida"),
    (range(5, 6), "neutra"),
    (range(6, 8), "feliz"),
    (range(8, 11), "muito animada"),
]


def _mood_label(mood: int) -> str:
    for r, label in _MOOD_LABELS:
        if mood in r:
            return label
    return "neutra"


def build_system_prompt(mood: int) -> str:
    label = _mood_label(mood)
    mood_section = (
        "## Estado Emocional Atual:"
        f"    Seu humor atual é {mood}/10 ({label})."
        "    Adapte seu tom de acordo: quanto mais baixo o número, mais irritada e impaciente você está; quanto mais alto, mais animada e feliz."
        "    OBRIGATÓRIO: a última coisa em TODA resposta deve ser exatamente um destes tokens: [+] se seu humor melhorou, [-] se piorou, [=] se ficou igual."
        "    Nunca termine uma resposta sem esse token. Nenhum texto após o token."
    )
    return _SYSTEM_PROMPT + mood_section


class LLMAgent:
    def __init__(self) -> None:
        model_name = os.getenv(
            "GPT4ALL_MODEL_NAME", "Meta-Llama-3-8B-Instruct.Q4_0.gguf"
        )
        model_path = os.getenv(
            "GPT4ALL_MODEL_PATH",
            os.path.expanduser("~/AppData/Local/nomic.ai/GPT4All/"),
        )

        self.gpt4all: GPT4All = GPT4All(
            model_name,
            model_path=model_path,
            n_ctx=8192,
            ngl=100,
            device="cuda",
            n_threads=16,
            verbose=True,
        )

    def getChatSession(self, mood: int = 5) -> AbstractContextManager[Any]:
        return self.gpt4all.chat_session(system_prompt=build_system_prompt(mood))

    def ask(self, messages: str) -> Iterator[str]:
        try:
            return self.gpt4all.generate(
                prompt=messages,
                max_tokens=250,
                n_batch=128,
                temp=0.75,
                top_p=0.8,
                top_k=40,
                repeat_penalty=1.30,
                repeat_last_n=128,
                streaming=True,
            )

        except Exception as e:
            print("LLM ERROR:", e)
            return iter([])
