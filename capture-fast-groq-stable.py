import numpy as np
import soundcard as sc
import torch
import torchaudio
import threading
import warnings

from groq import Groq
from faster_whisper import WhisperModel
from soundcard.mediafoundation import SoundcardRuntimeWarning

# ===================== WARNINGS =====================

warnings.filterwarnings("ignore", category=SoundcardRuntimeWarning)

# ===================== CONFIG =====================

INPUT_RATE = 44100        # ↓ меньше нагрузка
TARGET_RATE = 16000
CHANNEL = 0

FRAME_MS = 40             # ↓ меньше вызовов record
FRAME_SIZE = INPUT_RATE * FRAME_MS // 1000

# Silero VAD
VAD_FRAME = 512
VAD_THRESHOLD = 0.6

# End-of-speech
SILENCE_FRAMES_LIMIT = 15   # ~0.5 сек
MIN_PHRASE_SEC = 0.7

# Whisper
WHISPER_MODEL = "small"     # 🚀 максимум скорости
LANGUAGE = "ru"

# Groq
GROQ_MODEL = "llama-3.1-8b-instant"

# =================================================


# ===================== GROQ =====================

GROQ_API_KEY = input("Enter Groq API key: ").strip()
groq_client = Groq(api_key=GROQ_API_KEY)

def send_to_groq(text: str) -> str | None:
    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "Ты полезный ассистент."},
                {"role": "user", "content": text},
            ],
            temperature=0.2,
            max_tokens=512,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print("❌ Groq error:", e)
        return None


def send_to_groq_async(text: str):
    """Асинхронный вызов Groq — НЕ блокирует аудио"""
    def worker():
        response = send_to_groq(text)
        if response:
            print("🤖 Groq:", response)

    threading.Thread(target=worker, daemon=True).start()


# ===================== LOAD MODELS =====================

print("Loading Silero VAD...")
vad_model, _ = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    trust_repo=True
)
vad_model.eval()

print("Loading Whisper...")
whisper = WhisperModel(
    WHISPER_MODEL,
    device="cpu",
    compute_type="int8"
)

# ===================== AUDIO =====================

speaker = sc.default_speaker()
mic = sc.get_microphone(id=str(speaker.name), include_loopback=True)

vad_buffer = torch.zeros(0)
speech_buffer = []
silence_frames = 0
in_speech = False

print("Realtime listening...")

# ===================== MAIN LOOP =====================

with mic.recorder(samplerate=INPUT_RATE) as recorder:
    while True:
        chunk = recorder.record(numframes=FRAME_SIZE)
        mono = chunk[:, CHANNEL]

        audio = torch.from_numpy(mono).float()
        audio = torchaudio.functional.resample(
            audio, INPUT_RATE, TARGET_RATE
        )

        vad_buffer = torch.cat([vad_buffer, audio])

        while len(vad_buffer) >= VAD_FRAME:
            frame = vad_buffer[:VAD_FRAME]
            vad_buffer = vad_buffer[VAD_FRAME:]

            with torch.no_grad():
                speech_prob = vad_model(
                    frame.unsqueeze(0), TARGET_RATE
                ).item()

            # ---- ждём начало речи ----
            if not in_speech:
                if speech_prob >= VAD_THRESHOLD:
                    in_speech = True
                    speech_buffer.append(frame)
                    silence_frames = 0
                continue

            # ---- речь идёт ----
            if speech_prob >= VAD_THRESHOLD:
                speech_buffer.append(frame)
                silence_frames = 0
            else:
                silence_frames += 1

            duration = sum(len(x) for x in speech_buffer) / TARGET_RATE

            # ---- конец фразы ----
            if in_speech and silence_frames >= SILENCE_FRAMES_LIMIT:
                if duration >= MIN_PHRASE_SEC:
                    audio_chunk = torch.cat(speech_buffer)

                    segments, _ = whisper.transcribe(
                        audio_chunk.numpy(),
                        language=LANGUAGE,
                        beam_size=1,
                        temperature=0,
                        suppress_tokens=[],                # FIX faster-whisper
                        condition_on_previous_text=False,
                        without_timestamps=True,    # стабильность
                    )

                    text = " ".join(s.text for s in segments).strip()
                    if text:
                        print("📝 STT:", text)

                        # 🚀 НЕ блокируем аудио
                        send_to_groq_async(text)

                # ---- полный сброс ----
                speech_buffer.clear()
                silence_frames = 0
                in_speech = False
