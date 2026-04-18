# update-code

Workflow for making code changes in this project. Always follow this order:

## Rules

1. **Update source code** — make the requested change in the relevant module (`main.py`, `LLMAgent.py`, `STT.py`, `SpeechSynthesizer.py`, `VTubeStudioTalk.py`, `Bcolors.py`).

2. **Update tests** — update or add tests in `tests/` to cover the change. Every source change must have a corresponding test change. No exceptions.
   - New function → new test
   - Changed behavior → updated test
   - Bug fix → regression test

3. **Tests run automatically** — after every file edit, the PostToolUse hook runs `pytest tests -q`. Watch for failures.

4. **Fix failures before proceeding** — if tests fail after a change, fix them before moving on to the next task.

## Quick reference

- Run tests manually: `.venv/Scripts/python -m pytest tests/ -v`
- Test files live in: `tests/`
- All modules use mocks for heavy deps (Whisper, GPT4All, TTS, pyvts, discord) — see `tests/conftest.py`
