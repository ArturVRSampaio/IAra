# IAra

![tests](https://github.com/ArturVRSampaio/IAra/actions/workflows/tests.yml/badge.svg)

**I**nteligência **A**rtificial **R**aramente **A**curada

IAra is an interactive VTuber for Discord created by [ArturVRSampaio](https://github.com/ArturVRSampaio). She listens to users in a voice channel, transcribes their speech, generates responses with a local LLM, and talks back — with synchronized mouth animation in VTube Studio.

## Pipeline

```
Voice Channel (Discord)
    ↓ PCM chunks per user
STT — Whisper Turbo (faster-whisper, PT-BR)
    ↓ transcribed text
LLM — Llama-3.2-3B-Instruct via GPT4All (local, CUDA)
    ↓ streaming response, sentence by sentence
TTS — Kokoro (pf_dora, PT-BR, local, CUDA)
    ↓ .wav files synthesized in parallel with playback
Playback on Discord + mouth sync in VTube Studio (pyvts)
```

## Modules

| File | Responsibility |
|---|---|
| `main.py` | Entry point — calls `iara.bot.main()` |
| `iara/bot.py` | `DiscordBot` (commands and playback) + `AudioPipeline` (STT→LLM→TTS, queues, loops) |
| `iara/stt.py` | Transcription with faster-whisper, VAD, PT-BR |
| `iara/llm.py` | GPT4All wrapper, Iara's personality, streaming |
| `iara/tts.py` | Speech synthesis with Kokoro (pf_dora, PT-BR), single-thread executor |
| `iara/vtube.py` | WebSocket with VTube Studio, mouth synchronization |
| `iara/utils.py` | ANSI color utility for the terminal |

## Requirements

- Python 3.10+
- CUDA (NVIDIA GPU recommended)
- [FFmpeg](https://ffmpeg.org/) on PATH
- [VTube Studio](https://denchisoft.com/) with plugin API enabled
- LLM model downloaded automatically via GPT4All (default: `Llama-3.2-3B-Instruct-Q4_0.gguf`, configurable via `GPT4ALL_MODEL_NAME`)
- `kokoro` and `soundfile` (installed via `requirements.txt`; the Kokoro model (~300MB) is downloaded automatically on first run)

### Discord API

The bot uses **`discord.py==2.7.1`** and the experimental **`discord-ext-voice-recv`** extension, which enables receiving audio from voice channels. The extension **is not on PyPI** and must be installed directly from GitHub:

```bash
pip install discord.py==2.7.1
pip install git+https://github.com/imayhaveborkedit/discord-ext-voice-recv
```

> `discord-ext-voice-recv` is required to capture PCM from each user separately. Standard `discord.py` does not support receiving voice. Version `2.7.1+` is required for **DAVE** (Discord Audio Video Encryption) protocol support, introduced by Discord in 2024 — older versions silently fail on voice connections.

## Installation

```bash
pip install git+https://github.com/imayhaveborkedit/discord-ext-voice-recv
pip install -r requirements.txt
```

> `kokoro` and `soundfile` are included in `requirements.txt`. The Kokoro voice model (~300MB) will be downloaded automatically from HuggingFace on first run.

Copy `.env.example` to `.env` and fill in the values:

```env
DISCORD_TOKEN=your_token_here

# Optional — overrides the GPT4All model (downloaded automatically)
GPT4ALL_MODEL_NAME=Llama-3.2-3B-Instruct-Q4_0.gguf
GPT4ALL_MODEL_PATH=~/AppData/Local/nomic.ai/GPT4All/
```

On first run, VTube Studio will ask for authorization for the **"Iara VTuber"** plugin. The token will be saved to `token.txt`.

## Usage

```bash
python main.py
```

Available Discord commands:

| Command | Description |
|---|---|
| `!test` | Connects the bot to the user's voice channel |
| `!kickstart` | Reconnects the bot to the voice channel |
| `!ping` | Checks if the bot is online |

## Troubleshooting

### Bot joins and leaves the voice channel in a loop

**Symptom:** The bot connects to the voice channel and immediately disconnects, repeating in a loop. No error message appears in chat. The issue occurs on any network or machine.

**Cause:** Discord introduced the **DAVE** (Discord Audio Video Encryption) protocol in 2024 for end-to-end audio encryption. Older versions of `discord-ext-voice-recv` (before `0.5.3a180`) and `discord.py` (before `2.7.1`) do not implement the DAVE handshake, so the UDP voice connection never establishes, causing the loop.

**Fix:** Update the dependencies:

```bash
pip install "discord.py==2.7.1"
pip install --force-reinstall git+https://github.com/imayhaveborkedit/discord-ext-voice-recv
```

Verify that the `davey` package was installed as a dependency of `discord-ext-voice-recv`. If it doesn't appear, the installed version does not yet support DAVE.

### Bot doesn't receive audio — `OpusError: corrupted stream`

**Symptom:** The bot connects to the channel but never transcribes anything. The log shows `OpusError: corrupted stream` in the `PacketRouter`.

**Cause:** `discord-ext-voice-recv 0.5.3a180` implements the DAVE handshake to connect, but **does not apply DAVE decryption to received audio**. After transport decryption (XChaCha20), the payload is still encrypted by DAVE's E2EE layer. The Opus decoder receives encrypted data and fails.

**Fix (manual library patch):** Edit `.venv/Lib/site-packages/discord/ext/voice_recv/reader.py`, function `callback`, after the line `packet.decrypted_data = self.decryptor.decrypt_rtp(packet)`:

```python
# Apply DAVE (end-to-end) decryption layer if active
dave_session = getattr(getattr(self.voice_client, '_connection', None), 'dave_session', None)
if dave_session and dave_session.ready:
    import davey
    user_id = self.voice_client._ssrc_to_id.get(rtp_packet.ssrc)
    if user_id is not None:
        try:
            packet.decrypted_data = dave_session.decrypt(user_id, davey.MediaType.audio, packet.decrypted_data)
        except Exception as e:
            log.debug("DAVE decryption failed for ssrc %s: %s", rtp_packet.ssrc, e)
            return
```

> This patch must be reapplied every time the virtual environment is recreated, until `discord-ext-voice-recv` fixes the issue officially.

---

## Project Structure

```
IAra/
├── main.py            # entry point
├── iara/              # main package
│   ├── bot.py
│   ├── stt.py
│   ├── llm.py
│   ├── tts.py
│   ├── vtube.py
│   └── utils.py
├── tests/             # pytest test suite
├── experiments/       # per-subsystem prototypes and experiments
├── .env.example
└── .env
```
