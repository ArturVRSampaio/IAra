import importlib
import os
import sys
from unittest.mock import MagicMock, patch


def _make_llm():
    gpt4all_cls = MagicMock()
    gpt4all_mod = MagicMock()
    gpt4all_mod.GPT4All = gpt4all_cls
    sys.modules["gpt4all"] = gpt4all_mod

    import iara.llm as _mod
    importlib.reload(_mod)

    return _mod.LLMAgent(), gpt4all_cls, _mod


class TestLLMAgentInit:
    def test_gpt4all_instantiated(self):
        _, gpt4all_cls, _ = _make_llm()
        gpt4all_cls.assert_called_once()

    def test_uses_llama_model(self):
        _, gpt4all_cls, _ = _make_llm()
        args, _ = gpt4all_cls.call_args
        assert "Meta-Llama-3-8B-Instruct" in args[0]

    def test_uses_cuda(self):
        _, gpt4all_cls, _ = _make_llm()
        _, kwargs = gpt4all_cls.call_args
        assert kwargs.get("device") == "cuda"

    def test_model_path_from_env_var(self):
        with patch.dict(os.environ, {"GPT4ALL_MODEL_PATH": "/custom/path/"}):
            _, gpt4all_cls, _ = _make_llm()
            _, kwargs = gpt4all_cls.call_args
            assert kwargs.get("model_path") == "/custom/path/"

    def test_model_name_from_env_var(self):
        with patch.dict(os.environ, {"GPT4ALL_MODEL_NAME": "custom-model.gguf"}):
            _, gpt4all_cls, _ = _make_llm()
            args, _ = gpt4all_cls.call_args
            assert args[0] == "custom-model.gguf"

    def test_model_path_has_default(self):
        os.environ.pop("GPT4ALL_MODEL_PATH", None)
        _, gpt4all_cls, _ = _make_llm()
        _, kwargs = gpt4all_cls.call_args
        assert kwargs.get("model_path") is not None
        assert len(kwargs.get("model_path")) > 0

    def test_system_prompt_is_portuguese(self):
        _, _, _mod = _make_llm()
        assert "Iara" in _mod._SYSTEM_PROMPT
        assert "Você" in _mod._SYSTEM_PROMPT or "Voce" in _mod._SYSTEM_PROMPT

    def test_system_prompt_defines_discord_context(self):
        _, _, _mod = _make_llm()
        assert "Discord" in _mod._SYSTEM_PROMPT


class TestGetChatSession:
    def test_calls_chat_session_with_system_prompt(self):
        llm, _, _mod = _make_llm()
        llm.getChatSession()
        llm.gpt4all.chat_session.assert_called_once_with(system_prompt=_mod._SYSTEM_PROMPT)

    def test_returns_context_manager(self):
        llm, _, _ = _make_llm()
        session = llm.getChatSession()
        assert session is llm.gpt4all.chat_session.return_value


class TestAsk:
    def test_returns_generator_on_success(self):
        llm, _, _ = _make_llm()
        llm.gpt4all.generate.return_value = iter(["Olá", " mundo"])
        assert list(llm.ask("Oi")) == ["Olá", " mundo"]

    def test_generate_called_with_prompt(self):
        llm, _, _ = _make_llm()
        llm.gpt4all.generate.return_value = iter([])
        llm.ask("pergunta teste")
        _, kwargs = llm.gpt4all.generate.call_args
        assert kwargs.get("prompt") == "pergunta teste"

    def test_streaming_enabled(self):
        llm, _, _ = _make_llm()
        llm.gpt4all.generate.return_value = iter([])
        llm.ask("x")
        _, kwargs = llm.gpt4all.generate.call_args
        assert kwargs.get("streaming") is True

    def test_returns_empty_string_on_exception(self):
        llm, _, _ = _make_llm()
        llm.gpt4all.generate.side_effect = RuntimeError("model error")
        assert llm.ask("pergunta") == ""

    def test_max_tokens_set(self):
        llm, _, _ = _make_llm()
        llm.gpt4all.generate.return_value = iter([])
        llm.ask("x")
        _, kwargs = llm.gpt4all.generate.call_args
        assert kwargs.get("max_tokens", 0) > 0
