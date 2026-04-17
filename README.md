# IAra

**I**nteligência **A**rtificial **R**aramente **A**curada

IAra é uma VTuber interativa para Discord criada por [ArturVRSampaio](https://github.com/ArturVRSampaio). Ela ouve a voz dos usuários no canal, transcreve, gera respostas com um LLM local e fala de volta — com animação de boca sincronizada no VTube Studio.

## Pipeline

```
Canal de Voz (Discord)
    ↓ PCM chunks por usuário
STT — Whisper Turbo (faster-whisper, PT-BR)
    ↓ transcrição de texto
LLM — Meta-Llama-3-8B-Instruct via GPT4All (local, CUDA)
    ↓ resposta em streaming, frase a frase
TTS — XTTS v2 via Coqui TTS (local, CUDA)
    ↓ arquivos .wav
Playback no Discord + sincronização de boca no VTube Studio (pyvts)
```

## Módulos

| Arquivo | Responsabilidade |
|---|---|
| `main.py` | Orquestrador async, bot Discord, filas de voz e áudio |
| `STT.py` | Transcrição com faster-whisper, VAD, PT-BR |
| `LLMAgent.py` | Wrapper GPT4All, personalidade da Iara, streaming |
| `SpeechSynthesizer.py` | Síntese de voz com XTTS v2, executa em thread executor |
| `VTubeStudioTalk.py` | WebSocket com VTube Studio, sincronização da boca |
| `Bcolors.py` | Utilitário de cores ANSI para o terminal |

## Requisitos

- Python 3.10+
- CUDA (GPU NVIDIA recomendada)
- [FFmpeg](https://ffmpeg.org/) no PATH
- [VTube Studio](https://denchisoft.com/) com plugin API habilitado
- Modelo `Meta-Llama-3-8B-Instruct.Q4_0.gguf` baixado via GPT4All

## Instalação

```bash
pip install -r requirements.txt
```

Configure o `.env`:

```env
DISCORD_TOKEN=seu_token_aqui
```

Na primeira execução, o VTube Studio pedirá autorização para o plugin **"Iara VTuber"**. O token será salvo em `token.txt`.

## Uso

```bash
python main.py
```

Comandos disponíveis no Discord:

| Comando | Descrição |
|---|---|
| `!test` | Conecta o bot ao canal de voz do usuário |
| `!kickstart` | Reconecta o bot ao canal de voz |
| `!ping` | Verifica se o bot está online |

## Estrutura do Projeto

```
IAra/
├── main.py
├── LLMAgent.py
├── STT.py
├── SpeechSynthesizer.py
├── VTubeStudioTalk.py
├── Bcolors.py
├── experiments/       # Protótipos e testes por subsistema
└── .env
```
