import torch
import sounddevice as sd
import torchaudio
import tempfile
from scipy.io.wavfile import write
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Load processor and model
processor = Wav2Vec2Processor.from_pretrained("manandey/wav2vec2-large-xlsr-assamese")
model = Wav2Vec2ForCTC.from_pretrained("manandey/wav2vec2-large-xlsr-assamese")

SAMPLE_RATE = 16000  # Model expects 16 kHz audio

# Transcribe recorded or uploaded audio
def transcribe(audio_input):
    input_values = processor(audio_input, return_tensors="pt", sampling_rate=SAMPLE_RATE).input_values
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription

# Main callable function for Streamlit use
def speech_to_text(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        tmpfile.write(uploaded_file.read())
        tmpfile_path = tmpfile.name

    waveform, sample_rate = torchaudio.load(tmpfile_path)
    if sample_rate != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=SAMPLE_RATE)
    audio_np = waveform.squeeze().numpy()
    return transcribe(audio_np)

# (Optional) standalone recording logic
def record_audio(duration=5):
    print("Recording...")
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    return audio.squeeze()

if __name__ == "__main__":
    audio = record_audio()
    print("Transcription:", transcribe(audio))