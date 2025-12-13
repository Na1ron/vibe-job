import argparse
import csv
import subprocess
import sys
import time
import zipfile
import urllib.request
import tempfile
from pathlib import Path

# ================== CONFIG ==================

SVV_URL = "https://www.nirsoft.net/utils/soundvolumeview-x64.zip"
SVV_DIR = Path(r"C:\tools\SoundVolumeView")
SVV_EXE = SVV_DIR / "SoundVolumeView.exe"

# ============================================

def log(msg: str):
    print(f"[MASTER] {msg}", flush=True)

# ---------- SoundVolumeView bootstrap ----------

def ensure_soundvolumeview() -> Path:
    if SVV_EXE.exists():
        log("SoundVolumeView already installed")
        return SVV_EXE

    log("SoundVolumeView not found")
    log("Downloading from nirsoft.net")

    SVV_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        zip_path = tmpdir / "soundvolumeview.zip"

        urllib.request.urlretrieve(SVV_URL, zip_path)

        log("Extracting SoundVolumeView...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(SVV_DIR)

    if not SVV_EXE.exists():
        raise RuntimeError("SoundVolumeView download failed")

    log("SoundVolumeView installed successfully")
    return SVV_EXE

# ---------- SVV helpers ----------

def run_svv(svv: Path, args: list[str], check=True):
    subprocess.run(
        [str(svv), *args],
        check=check,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

def dump_sessions_csv(svv: Path, csv_path: Path):
    run_svv(svv, ["/scomma", str(csv_path)])

def detect_encoding(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return "utf-16"
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    return "utf-8-sig"

def read_render_app_rows(csv_path: Path):
    enc = detect_encoding(csv_path)
    log(f"CSV decoded as {enc}")

    with csv_path.open("r", encoding=enc, newline="") as f:
        reader = csv.DictReader(f)
        log(f"CSV fields: {reader.fieldnames}")

        return [
            r for r in reader
            if r.get("Type") == "Application"
            and r.get("Direction") == "Render"
        ]

def get_active_render_sessions(svv: Path, csv_path: Path):
    dump_sessions_csv(svv, csv_path)
    rows = read_render_app_rows(csv_path)

    sessions = {}
    for r in rows:
        exe = (r.get("Process Path") or "").strip()
        if not exe:
            continue

        sessions[exe.lower()] = {
            "name": r.get("Name", "").strip(),
            "exe": exe,
            "device": r.get("Device Name", "").strip(),
        }

    return list(sessions.values())

def set_app_default(svv: Path, exe_path: str, device_name: str):
    run_svv(svv, ["/SetAppDefault", device_name, "all", exe_path])

def kill_and_restart(exe_path: str, wait_s: float = 1.0):
    exe_name = Path(exe_path).name
    subprocess.run(
        ["taskkill", "/IM", exe_name, "/F"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    time.sleep(wait_s)
    subprocess.Popen(exe_path, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# ---------- main ----------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target-render", default="CABLE Input")
    p.add_argument("--no-restart", action="store_true")

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

    svv = ensure_soundvolumeview()

    csv_path = Path(tempfile.gettempdir()) / "svv_sessions.csv"

    log("Scanning render audio sessions...")
    sessions = get_active_render_sessions(svv, csv_path)

    if not sessions:
        raise RuntimeError("No active render audio sessions found")

    log(f"Found {len(sessions)} render sessions")
    for s in sessions:
        log(f" - {s['name']} | {Path(s['exe']).name} | device={s['device']}")

    originals = sessions.copy()

    log(f'Routing apps → "{args.target_render}"')
    for s in sessions:
        set_app_default(svv, s["exe"], args.target_render)
        if not args.no_restart:
            kill_and_restart(s["exe"])

    log("Waiting 3s for audio to stabilize...")
    time.sleep(3)

    cmd = [
        sys.executable,
        "stream_transcribe.py",
        "--whisper-model", args.whisper_model,
        "--device", args.device,
        "--language", args.language,
        "--input", args.input,
        "--vad", args.vad,
        "--silence-timeout", args.silence_timeout,
        "--frame-ms", args.frame_ms,
    ]
    if args.chat:
        cmd += ["--chat", "--groq-model", args.groq_model]

    log("Starting streaming transcription...")
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        pass

    log("Restoring original routing...")
    for s in originals:
        set_app_default(svv, s["exe"], s["device"])
        if not args.no_restart:
            kill_and_restart(s["exe"])

    log("Done")

if __name__ == "__main__":
    main()
