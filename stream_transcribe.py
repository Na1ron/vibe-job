import argparse
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import torch
import whisper

try:
    import webrtcvad
    VAD_AVAILABLE = True
except Exception:
    VAD_AVAILABLE = False

from groq import Groq

SYSTEM_PROMPT_DEFAULT = (
    "Ты senior DevOps инженер и проходишь собеседование. "
    "Я буду задавать тебе вопросы, на которые ты должен давать "
    "краткие, но ёмкие ответы.\n\n"
    "Требования к ответам:\n"
    "- Используй маркированные списки\n"
    "- Без воды и лишних объяснений\n"
    "- Ответ должен умещаться в один экран\n"
    "- Пиши по-русски"
)

SAMPLE_RATE = 16000


def log(msg):
    print(msg, flush=True)


def pick_input_device(user_value: str) -> int:
    if user_value != "auto":
        return int(user_value)

    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and "cable output" in d["name"].lower():
            return i

    return sd.default.device[0]


def energy_is_speech(frame: np.ndarray, threshold: float = 0.01) -> bool:
    rms = float(np.sqrt(np.mean(frame * frame))) if len(frame) else 0.0
    return rms >= threshold


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--whisper-model", default="small",
                   choices=["tiny", "base", "small", "medium", "large", "large-v3"])
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"])
    p.add_argument("--language", default="ru")
    p.add_argument("--input", default="auto")
    p.add_argument("--vad", default="webrtc", choices=["webrtc", "energy", "off"])
    p.add_argument("--frame-ms", type=int, default=30)
    p.add_argument("--silence-timeout", type=float, default=1.0)

    p.add_argument("--chat", action="store_true")
    p.add_argument("--groq-model", default="llama-3.1-8b-instant")
    p.add_argument("--system-prompt", default=SYSTEM_PROMPT_DEFAULT)

    args = p.parse_args()

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    log(f"[INFO] Compute device: {device}")

    log(f"[MODEL] Loading Whisper: {args.whisper_model}")
    wmodel = whisper.load_model(args.whisper_model, device=device)
    log("[MODEL] Whisper loaded")

    groq = None
    messages = [{"role": "system", "content": args.system_prompt}]
    if args.chat:
        groq = Groq()

    use_webrtc = args.vad == "webrtc" and VAD_AVAILABLE
    vad = webrtcvad.Vad(1) if use_webrtc else None

    frame_samples = int(SAMPLE_RATE * args.frame_ms / 1000)
    silence_limit = int(args.silence_timeout * 1000 / args.frame_ms)

    input_dev = pick_input_device(args.input)
    log(f"[AUDIO] Input device: {sd.query_devices(input_dev)['name']}")

    buffer = []
    silence = 0
    speaking = False

    def process():
        nonlocal buffer
        if not buffer:
            return

        audio = np.concatenate(buffer).astype(np.float32)
        buffer = []

        res = wmodel.transcribe(audio, language=args.language, fp16=(device == "cuda"))
        text = res.get("text", "").strip()
        if len(text.split()) < 2:
            return

        t = time.strftime("%H:%M:%S")
        log(f"\n[{t}] You: {text}")

        if args.chat and groq:
            messages.append({"role": "user", "content": text})
            r = groq.chat.completions.create(
                model=args.groq_model,
                messages=messages,
                temperature=0.3,
                max_tokens=400,
            )
            reply = r.choices[0].message.content.strip()
            messages.append({"role": "assistant", "content": reply})
            log(f"[{t}] Assistant:\n{reply}")

    def callback(indata, frames, _, status):
        nonlocal silence, speaking, buffer
        if status:
            log(f"[AUDIO WARNING] {status}")

        audio = indata[:, 0]

        if args.vad == "off":
            speech = True
        elif vad:
            pcm = (audio * 32767).astype(np.int16).tobytes()
            speech = vad.is_speech(pcm, SAMPLE_RATE)
        else:
            speech = energy_is_speech(audio)

        if speech:
            speaking = True
            silence = 0
            buffer.append(audio.copy())
        elif speaking:
            silence += 1
            buffer.append(audio.copy())
            if silence >= silence_limit:
                process()
                speaking = False
                silence = 0

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        blocksize=frame_samples,
        device=input_dev,
        callback=callback,
    ):
        log("[AUDIO] Listening... Ctrl+C to stop")
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            log("\n[STOP]")


if __name__ == "__main__":
    main()
