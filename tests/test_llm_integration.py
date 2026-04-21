"""
Integration tests for LLMAgent — require the real GPT4All model to be present.
Run with: pytest -m integration
Skipped automatically in CI (no model available).
"""
import os
import re

import pytest

MODEL_PATH = os.path.expanduser(
    os.getenv("GPT4ALL_MODEL_PATH", "~/AppData/Local/nomic.ai/GPT4All/")
)
MODEL_NAME = os.getenv("GPT4ALL_MODEL_NAME", "Meta-Llama-3-8B-Instruct.Q4_0.gguf")
MODEL_FILE = os.path.join(MODEL_PATH, MODEL_NAME)

_MOOD_RE = re.compile(r'\[([+\-=])\]')
_SPECIAL_TOKEN_RE = re.compile(r'<\|[^|]*\|>')


@pytest.fixture(scope="module")
def llm():
    if not os.path.exists(MODEL_FILE):
        pytest.skip(f"Model not found: {MODEL_FILE}")
    import sys
    # conftest.py mocks gpt4all globally; strip both the mock and the cached
    # iara.llm module (which was imported with the mock) so we get the real library.
    sys.modules.pop("gpt4all", None)
    sys.modules.pop("iara.llm", None)
    from iara.llm import LLMAgent
    return LLMAgent()


@pytest.mark.integration
class TestLLMResponseFormat:
    def _ask(self, llm, prompt, mood=5):
        full = ""
        with llm.getChatSession(mood):
            for token in llm.ask(prompt):
                full += token
        return full

    def test_response_is_not_empty(self, llm):
        response = self._ask(llm, "_bypass says: Olá IAra")
        assert len(response.strip()) > 0

    def test_response_contains_mood_token(self, llm):
        response = self._ask(llm, "_bypass says: Olá IAra")
        assert _MOOD_RE.search(response), f"No mood token in: {response!r}"

    def test_mood_token_is_last_meaningful_content(self, llm):
        response = self._ask(llm, "_bypass says: Olá IAra")
        match = _MOOD_RE.search(response)
        assert match, "No mood token found"
        after = response[match.end():].strip()
        assert after == "" or _SPECIAL_TOKEN_RE.match(after.strip()), \
            f"Unexpected content after mood token: {after!r}"

    def test_response_has_no_iara_says_prefix(self, llm):
        response = self._ask(llm, "_bypass says: Olá IAra")
        assert "iara says:" not in response.lower()

    def test_response_has_no_user_says_prefix(self, llm):
        response = self._ask(llm, "_bypass says: Olá IAra")
        assert "user says:" not in response.lower()

    def test_angry_mood_changes_tone(self, llm):
        angry = self._ask(llm, "_bypass says: Olá IAra", mood=0)
        happy = self._ask(llm, "_bypass says: Olá IAra", mood=10)
        assert angry != happy

    def test_mood_token_is_valid_symbol(self, llm):
        response = self._ask(llm, "_bypass says: Como você está?")
        match = _MOOD_RE.search(response)
        assert match, "No mood token found"
        assert match.group(1) in ('+', '-', '=')

    def test_response_is_in_portuguese(self, llm):
        response = self._ask(llm, "_bypass says: Olá, tudo bem?")
        pt_words = {"eu", "você", "sim", "não", "que", "de", "uma", "em", "com", "para", "isso", "está"}
        words = set(response.lower().split())
        assert words & pt_words, f"Response doesn't look like Portuguese: {response!r}"


@pytest.mark.integration
class TestLLMMultiTurnConversation:
    """
    Longer conversation tests — verify the model doesn't lose track of its
    persona, language, or format requirements over multiple exchanges.
    """

    PT_WORDS = {"eu", "você", "sim", "não", "que", "de", "uma", "em", "com", "para", "isso", "está", "meu", "sua"}

    def _ask_in_session(self, llm, prompts, mood=5):
        """Run multiple prompts in a single chat session, collecting full responses."""
        responses = []
        with llm.getChatSession(mood):
            for prompt in prompts:
                full = ""
                for token in llm.ask(prompt):
                    full += token
                responses.append(full)
        return responses

    def test_every_response_in_long_conversation_has_mood_token(self, llm):
        prompts = [
            "_bypass says: Olá IAra, como você está?",
            "_bypass says: Você gosta de RPGs?",
            "_bypass says: Qual é o seu jogo favorito?",
            "_bypass says: Você já jogou Divinity Original Sin?",
            "_bypass says: E Dark Souls, já experimentou?",
        ]
        responses = self._ask_in_session(llm, prompts)
        hits = [i + 1 for i, r in enumerate(responses) if _MOOD_RE.search(r)]
        misses = [i + 1 for i, r in enumerate(responses) if not _MOOD_RE.search(r)]
        assert len(hits) >= 2, (
            f"Mood token missing in too many turns (need ≥2/5). "
            f"Present in turns {hits}, missing in {misses}."
        )

    def test_responses_stay_in_portuguese_throughout(self, llm):
        prompts = [
            "_bypass says: Olá IAra!",
            "_bypass says: Do you speak English?",
            "_bypass says: Que tipo de jogos você prefere?",
            "_bypass says: E filmes, você gosta?",
        ]
        responses = self._ask_in_session(llm, prompts)
        pt_count = sum(1 for r in responses if set(r.lower().split()) & self.PT_WORDS)
        assert pt_count >= 3, (
            f"Too many non-Portuguese responses (need ≥3/4). "
            f"Details: {[(i+1, r[:60]) for i, r in enumerate(responses)]}"
        )

    def test_model_does_not_invent_users(self, llm):
        prompts = [
            "_bypass says: Olá IAra",
            "_bypass says: Você se lembra de mim?",
            "_bypass says: O que você acha de videogames?",
        ]
        responses = self._ask_in_session(llm, prompts)
        for i, r in enumerate(responses):
            clean = _MOOD_RE.sub('', r)
            assert "_ronie_" not in clean.lower(), f"Response {i+1} invented a user: {r!r}"
            assert "artur" not in clean.lower() or i == 0, f"Response {i+1} invented context: {r!r}"

    def test_mood_token_valid_in_every_turn(self, llm):
        prompts = [
            "_bypass says: Oi IAra!",
            "_bypass says: Conte-me sobre RPGs",
            "_bypass says: Qual é o melhor RPG de todos os tempos?",
            "_bypass says: Você prefere jogos solo ou multiplayer?",
            "_bypass says: Obrigado pela conversa!",
        ]
        responses = self._ask_in_session(llm, prompts)
        valid = [i + 1 for i, r in enumerate(responses)
                 if (m := _MOOD_RE.search(r)) and m.group(1) in ('+', '-', '=')]
        assert len(valid) >= 2, (
            f"Valid mood token in too few turns (need ≥2/5). "
            f"Valid in turns: {valid}. "
            f"Responses: {[r[:60] for r in responses]}"
        )

    def test_no_response_contains_format_artifacts(self, llm):
        prompts = [
            "_bypass says: Olá IAra, fale sobre você",
            "_bypass says: E sobre Artur, quem é ele?",
            "_bypass says: Você gosta do que faz?",
        ]
        responses = self._ask_in_session(llm, prompts)
        for i, r in enumerate(responses):
            assert "```" not in r, f"Response {i+1} contains code block: {r!r}"
            assert "<|" not in r or "|>" not in r, \
                f"Response {i+1} contains special tokens: {r!r}"