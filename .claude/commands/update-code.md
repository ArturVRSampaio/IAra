# update-code

Workflow for making code changes in this project. Always follow this order:

## Rules

1. **Update source code** — make the requested change in the relevant module (`iara/bot.py`, `iara/llm.py`, `iara/stt.py`, `iara/tts.py`, `iara/vtube.py`, `iara/utils.py`).

2. **Update tests** — update or add tests in `tests/` to cover the change. Every source change must have a corresponding test change. No exceptions.
   - New function → new test
   - Changed behavior → updated test
   - Bug fix → regression test

3. **Tests run automatically** — after every file edit, the PostToolUse hook runs `pytest tests -q`. Watch for failures.

4. **Fix failures before proceeding** — if tests fail after a change, fix them before moving on to the next task.

5. **Check the README** — after every change, review `README.md` and update it if any of the following changed:
   - Pipeline or data flow
   - Module responsibilities
   - Commands or bot usage
   - Environment variables (`.env.example` is the reference)
   - Dependencies or setup steps

## Quick reference

- Run tests manually: `.venv/Scripts/python -m pytest tests/ -v`
- Test files live in: `tests/`
- All modules use mocks for heavy deps (Whisper, GPT4All, TTS, pyvts, discord) — see `tests/conftest.py`
