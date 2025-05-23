import time
import torch
import sounddevice as sd
import torchaudio
from TTS.api import TTS
import numpy as np
import nltk

# Download NLTK data for sentence tokenization
nltk.download('punkt')
nltk.download('punkt_tab')

# Check for CUDA availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Initialize TTS model
tts = TTS("tts_models/multilingual/multi-dataset/your_tts", progress_bar=True).to(device)
print("Available speakers:", tts.speakers)
print("Model loaded")

# Input text
text = "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent."

# Split text into sentences using NLTK
sentences = nltk.sent_tokenize(text)
print(f"Sentences: {sentences}")

# Further split long sentences into phrases based on commas or conjunctions
chunks = []
for sentence in sentences:
    # Split by commas or conjunctions for long sentences
    phrases = sentence.replace(',', ', ').split(', ')
    phrases = [phrase.strip() for phrase in phrases if phrase.strip()]
    chunks.extend(phrases)

print(f"Chunks to process: {chunks}")

# Initialize variables for audio concatenation
sample_rate = 22050  # Default sample rate for YourTTS model, adjust if needed
concatenated_waveform = np.array([], dtype=np.float32)

# Start timing
start = time.time()

# Process each chunk
for i, chunk in enumerate(chunks):
    print(f"Processing chunk: {chunk}")

    # Generate audio for the current chunk
    waveform = tts.tts(
        text=chunk,
        speaker=tts.speakers[2],
        language="en"
    )

    # Convert waveform to numpy array if it’s a list
    if isinstance(waveform, list):
        waveform = np.array(waveform, dtype=np.float32)

    # Append to concatenated waveform
    concatenated_waveform = np.concatenate(
        (concatenated_waveform, waveform)) if concatenated_waveform.size else waveform

# End timing
end = time.time()
print(f"Time to generate audio: {end - start:.2f} seconds")

# Play the concatenated audio
sd.play(concatenated_waveform, samplerate=sample_rate)
sd.wait()

# Save the concatenated audio to a file
torchaudio.save("output_natural_chunked.wav", torch.tensor(concatenated_waveform).unsqueeze(0), sample_rate)
print("Audio saved to output_natural_chunked.wav")