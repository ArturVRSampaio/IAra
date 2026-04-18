import asyncio
import re

import soundfile as sf
from kokoro import KPipeline


class SpeechSynthesizer:
    def __init__(self):
        self.pipeline = KPipeline(lang_code='p')
        self.voice = 'pf_dora'

    async def generate_tts_file(self, text: str, output_path: str) -> None:
        if not text.strip():
            return

        text = re.sub(r'[!.()]+', '', text)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._synthesize, text, output_path)

    def _synthesize(self, text: str, output_path: str) -> None:
        audio_chunks = []
        for _, _, audio in self.pipeline(text, voice=self.voice):
            if audio is not None:
                audio_chunks.append(audio)
        if audio_chunks:
            import numpy as np
            audio_data = np.concatenate(audio_chunks)
            sf.write(output_path, audio_data, 24000)
