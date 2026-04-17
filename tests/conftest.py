import sys
from unittest.mock import MagicMock, AsyncMock

# discord.ext.commands.Cog must be a real type so DiscordBot can inherit from it
_discord_commands_mock = MagicMock()
_discord_commands_mock.Cog = object
_discord_commands_mock.command = lambda **kw: (lambda f: f)  # passthrough decorator

_discord_mock = MagicMock()
_discord_ext_mock = MagicMock()
_discord_ext_mock.commands = _discord_commands_mock

sys.modules.setdefault("discord", _discord_mock)
sys.modules.setdefault("discord.ext", _discord_ext_mock)
sys.modules.setdefault("discord.ext.commands", _discord_commands_mock)
sys.modules.setdefault("discord.ext.voice_recv", MagicMock())
sys.modules.setdefault("faster_whisper", MagicMock())
sys.modules.setdefault("gpt4all", MagicMock())
sys.modules.setdefault("TTS", MagicMock())
sys.modules.setdefault("TTS.api", MagicMock())
sys.modules.setdefault("pyvts", MagicMock())
sys.modules.setdefault("torchaudio", MagicMock())
sys.modules.setdefault("dotenv", MagicMock())
sys.modules.setdefault("pydub", MagicMock())
sys.modules.setdefault("pydub.audio_segment", MagicMock())