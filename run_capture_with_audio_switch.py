import argparse
import subprocess
import sys
import time

import comtypes.client
from pycaw.constants import (
    CLSID_MMDeviceEnumerator,
    CLSID_PolicyConfigClient,
    eCommunications,
    eConsole,
    eMultimedia,
    eRender,
)
from pycaw.pycaw import AudioUtilities, IMMDeviceEnumerator, IPolicyConfigVista


ROLES = (eConsole, eMultimedia, eCommunications)


def log(msg: str):
    print(f"[MASTER] {msg}", flush=True)


def get_default_render_device_id():
    enumerator = comtypes.client.CreateObject(
        CLSID_MMDeviceEnumerator, interface=IMMDeviceEnumerator
    )
    return enumerator.GetDefaultAudioEndpoint(eRender, eConsole).GetId()


def find_render_device_id(target: str) -> tuple[str | None, str | None]:
    target_lower = target.lower()
    for device in AudioUtilities.GetAllDevices():
        if device.DataFlow != eRender or device.State != 1:
            continue
        if target_lower in device.FriendlyName.lower():
            return device.id, device.FriendlyName
    return None, None


def set_default_render(device_id: str):
    policy = comtypes.client.CreateObject(
        CLSID_PolicyConfigClient, interface=IPolicyConfigVista
    )
    for role in ROLES:
        policy.SetDefaultEndpoint(device_id, role)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--target-render",
        default=None,
        help=(
            "Имя (или часть имени) выходного устройства, которое нужно сделать "
            "дефолтным перед запуском. Если не указано — сохраняется текущий дефолт."
        ),
    )

    # passthrough
    p.add_argument("--whisper-model", default="small")
    p.add_argument("--device", default="auto")
    p.add_argument("--language", default="ru")
    p.add_argument("--input", default="auto")
    p.add_argument("--vad", default="webrtc")
    p.add_argument("--silence-timeout", default="1.0")
    p.add_argument("--frame-ms", default="30")
    p.add_argument("--chat", action="store_true")
    p.add_argument("--groq-model", default="llama-3.1-8b-instant")

    args = p.parse_args()

    log(f"Python executable: {sys.executable}")

    original_default = get_default_render_device_id()
    target_friendly = None

    if args.target_render:
        device_id, friendly = find_render_device_id(args.target_render)
        if not device_id:
            raise RuntimeError(
                f'Не удалось найти рендер-устройство с именем, содержащим "{args.target_render}"'
            )

        if device_id != original_default:
            log(f'Переключаю дефолтный вывод на "{friendly}"...')
            set_default_render(device_id)
            target_friendly = friendly
            time.sleep(1.0)
        else:
            log(f'Устройство "{friendly}" уже выбрано по умолчанию')

    cmd = [
        sys.executable,
        "stream_transcribe.py",
        "--whisper-model",
        args.whisper_model,
        "--device",
        args.device,
        "--language",
        args.language,
        "--input",
        args.input,
        "--vad",
        args.vad,
        "--silence-timeout",
        args.silence_timeout,
        "--frame-ms",
        args.frame_ms,
        "--loopback",
    ]

    if args.chat:
        cmd += ["--chat", "--groq-model", args.groq_model]

    if target_friendly:
        log(f"Запускаю стриминговую транскрибацию с loopback {target_friendly}...")
    else:
        log("Запускаю стриминговую транскрибацию с loopback дефолтного вывода...")

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        pass
    finally:
        if args.target_render and original_default:
            log("Возвращаю исходное дефолтное устройство вывода...")
            set_default_render(original_default)

    log("Done")


if __name__ == "__main__":
    main()
