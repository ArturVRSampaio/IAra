import asyncio
import re
from concurrent.futures import ThreadPoolExecutor

import soundfile as sf
from kokoro import KPipeline


class SpeechSynthesizer:
    def __init__(self):
        self.pipeline = KPipeline(lang_code='p')
        self.voice = 'pf_dora'
        self._executor = ThreadPoolExecutor(max_workers=1)

    async def generate_tts_file(self, text: str, output_path: str) -> None:
        if not text.strip():
            return

        text = re.sub(r'[\n\r]+', ' ', text)
        text = re.sub(r'[\U00010000-\U0010FFFF]', '', text)  # strip emojis
        text = re.sub(r'[!.()?,;:]+', '', text)              # strip punctuation espeak chokes on
        text = text.strip()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._synthesize, text, output_path)

    def _synthesize(self, text: str, output_path: str) -> None:
        audio_chunks = []
        for _, _, audio in self.pipeline(text, voice=self.voice):
            if audio is not None:
                audio_chunks.append(audio)
        if audio_chunks:
            import numpy as np
            audio_data = np.concatenate(audio_chunks)
            sf.write(output_path, audio_data, 24000)
