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
| `main.py` | Ponto de entrada — chama `iara.bot.main()` |
| `iara/bot.py` | Orquestrador async, bot Discord, filas de voz e áudio |
| `iara/stt.py` | Transcrição com faster-whisper, VAD, PT-BR |
| `iara/llm.py` | Wrapper GPT4All, personalidade da Iara, streaming |
| `iara/tts.py` | Síntese de voz com XTTS v2, executa em thread executor |
| `iara/vtube.py` | WebSocket com VTube Studio, sincronização da boca |
| `iara/utils.py` | Utilitário de cores ANSI para o terminal |

## Requisitos

- Python 3.10+
- CUDA (GPU NVIDIA recomendada)
- [FFmpeg](https://ffmpeg.org/) no PATH
- [VTube Studio](https://denchisoft.com/) com plugin API habilitado
- Modelo `Meta-Llama-3-8B-Instruct.Q4_0.gguf` baixado via GPT4All

### Discord API

O bot utiliza **`discord.py==2.7.1`** e a extensão experimental **`discord-ext-voice-recv`**, que habilita o recebimento de áudio dos canais de voz. A extensão **não está no PyPI** e deve ser instalada diretamente do GitHub:

```bash
pip install discord.py==2.7.1
pip install git+https://github.com/imayhaveborkedit/discord-ext-voice-recv
```

> `discord-ext-voice-recv` é necessária para capturar o PCM de cada usuário separadamente. O `discord.py` padrão não suporta recebimento de voz. A versão `2.7.1+` é necessária para suporte ao protocolo **DAVE** (Discord Audio Video Encryption), introduzido pelo Discord em 2024 — versões anteriores falham silenciosamente na conexão de voz.

## Instalação

```bash
pip install git+https://github.com/imayhaveborkedit/discord-ext-voice-recv
pip install -r requirements.txt
```

Copie `.env.example` para `.env` e preencha os valores:

```env
DISCORD_TOKEN=seu_token_aqui

# Opcional — sobrescreve o modelo GPT4All
GPT4ALL_MODEL_NAME=Meta-Llama-3-8B-Instruct.Q4_0.gguf
GPT4ALL_MODEL_PATH=~/AppData/Local/nomic.ai/GPT4All/
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

## Troubleshooting

### Bot entra e sai do canal de voz em loop

**Sintoma:** O bot conecta ao canal de voz e desconecta imediatamente, repetindo em loop. Nenhuma mensagem de erro aparece no chat. O problema ocorre em qualquer rede ou máquina.

**Causa:** O Discord introduziu o protocolo **DAVE** (Discord Audio Video Encryption) em 2024 para criptografia ponta-a-ponta no áudio. Versões antigas do `discord-ext-voice-recv` (anteriores a `0.5.3a180`) e do `discord.py` (anteriores a `2.7.1`) não implementam o handshake DAVE, então a conexão UDP de voz nunca se estabelece, causando o loop.

**Solução:** Atualize as dependências:

```bash
pip install "discord.py==2.7.1"
pip install --force-reinstall git+https://github.com/imayhaveborkedit/discord-ext-voice-recv
```

Verifique se o pacote `davey` foi instalado como dependência do `discord-ext-voice-recv`. Se não aparecer, a versão instalada ainda não tem suporte a DAVE.

---

## Estrutura do Projeto

```
IAra/
├── main.py            # ponto de entrada
├── iara/              # pacote principal
│   ├── bot.py
│   ├── stt.py
│   ├── llm.py
│   ├── tts.py
│   ├── vtube.py
│   └── utils.py
├── tests/             # suite de testes (pytest)
├── experiments/       # protótipos e testes por subsistema
├── .env.example
└── .env
```
