# IAra

![tests](https://github.com/ArturVRSampaio/IAra/actions/workflows/tests.yml/badge.svg)

**I**nteligência **A**rtificial **R**aramente **A**curada

IAra é uma VTuber interativa para Discord criada por [ArturVRSampaio](https://github.com/ArturVRSampaio). Ela ouve a voz dos usuários no canal, transcreve, gera respostas com um LLM local e fala de volta — com animação de boca sincronizada no VTube Studio.

## Pipeline

```
Canal de Voz (Discord)
    ↓ PCM chunks por usuário
STT — Whisper Turbo (faster-whisper, PT-BR)
    ↓ transcrição de texto
LLM — Llama-3.2-3B-Instruct via GPT4All (local, CUDA)
    ↓ resposta em streaming, frase a frase
TTS — Kokoro (pf_dora, PT-BR, local, CUDA)
    ↓ arquivos .wav em paralelo com o playback
Playback no Discord + sincronização de boca no VTube Studio (pyvts)
```

## Módulos

| Arquivo | Responsabilidade |
|---|---|
| `main.py` | Ponto de entrada — chama `iara.bot.main()` |
| `iara/bot.py` | `DiscordBot` (comandos e playback) + `AudioPipeline` (STT→LLM→TTS, filas, loops) |
| `iara/stt.py` | Transcrição com faster-whisper, VAD, PT-BR |
| `iara/llm.py` | Wrapper GPT4All, personalidade da Iara, streaming |
| `iara/tts.py` | Síntese de voz com Kokoro (pf_dora, PT-BR), executor single-thread |
| `iara/vtube.py` | WebSocket com VTube Studio, sincronização da boca |
| `iara/utils.py` | Utilitário de cores ANSI para o terminal |

## Requisitos

- Python 3.10+
- CUDA (GPU NVIDIA recomendada)
- [FFmpeg](https://ffmpeg.org/) no PATH
- [VTube Studio](https://denchisoft.com/) com plugin API habilitado
- Modelo LLM baixado automaticamente via GPT4All (padrão: `Llama-3.2-3B-Instruct-Q4_0.gguf`, configurável via `GPT4ALL_MODEL_NAME`)
- `kokoro` e `soundfile` (instalados via `requirements.txt`; o modelo Kokoro (~300MB) é baixado automaticamente na primeira execução)

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

> `kokoro` e `soundfile` estão incluídos no `requirements.txt`. O modelo de voz Kokoro (~300MB) será baixado automaticamente do HuggingFace na primeira execução.

Copie `.env.example` para `.env` e preencha os valores:

```env
DISCORD_TOKEN=seu_token_aqui

# Opcional — sobrescreve o modelo GPT4All (baixado automaticamente)
GPT4ALL_MODEL_NAME=Llama-3.2-3B-Instruct-Q4_0.gguf
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

### Bot não recebe áudio — `OpusError: corrupted stream`

**Sintoma:** O bot conecta ao canal mas nunca transcreve nada. O log mostra `OpusError: corrupted stream` no `PacketRouter`.

**Causa:** `discord-ext-voice-recv 0.5.3a180` implementa o handshake DAVE para conectar, mas **não aplica a decriptação DAVE no áudio recebido**. Após a decriptação de transporte (XChaCha20), o payload ainda está criptografado pela camada E2EE do DAVE. O decoder Opus recebe dados criptografados e falha.

**Solução (patch manual na biblioteca):** Edite `.venv/Lib/site-packages/discord/ext/voice_recv/reader.py`, função `callback`, após a linha `packet.decrypted_data = self.decryptor.decrypt_rtp(packet)`:

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

> Este patch precisa ser reaplicado sempre que o ambiente virtual for recriado, até que o `discord-ext-voice-recv` corrija o problema oficialmente.

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
