from __future__ import annotations

import io
import logging
import os
import re
import time
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

log = logging.getLogger("speech-server")
MODELS_DIR = Path(os.getenv("SPEECH_MODELS_DIR", Path.home() / ".cache" / "voice-agent-models"))
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small.en")
KOKORO_RELEASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
KOKORO_FILES = ("kokoro-v1.0.onnx", "voices-v1.0.bin")

_whisper = None
_kokoro = None


def _ensure_kokoro_files() -> tuple[Path, Path]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in KOKORO_FILES:
        path = MODELS_DIR / name
        if not path.exists():
            log.info("downloading %s ...", name)
            partial = path.with_suffix(path.suffix + ".part")
            urllib.request.urlretrieve(f"{KOKORO_RELEASE}/{name}", partial)
            partial.rename(path)
        paths.append(path)
    return paths[0], paths[1]


def whisper():
    global _whisper
    if _whisper is None:
        from faster_whisper import WhisperModel

        _whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8", download_root=str(MODELS_DIR))
    return _whisper


def kokoro():
    global _kokoro
    if _kokoro is None:
        from kokoro_onnx import Kokoro

        _kokoro = Kokoro(*map(str, _ensure_kokoro_files()))
    return _kokoro


@asynccontextmanager
async def lifespan(_app: FastAPI):
    t = time.perf_counter()
    whisper()
    kokoro().create("Ready.", voice="af_heart", speed=1.0, lang="en-us")
    log.info("models loaded in %.1fs", time.perf_counter() - t)
    yield


app = FastAPI(title="Local speech server", lifespan=lifespan)


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    prompt: str | None = Form(None),
) -> dict:
    audio, _sr = sf.read(io.BytesIO(await file.read()), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if _sr != 16_000:
        n = int(len(audio) * 16_000 / _sr)
        audio = np.interp(np.linspace(0, len(audio), n, endpoint=False), np.arange(len(audio)), audio)
        audio = audio.astype(np.float32)
    t = time.perf_counter()
    segments, _ = whisper().transcribe(
        audio, language="en", beam_size=1, vad_filter=True, initial_prompt=prompt or None
    )
    text = " ".join(s.text.strip() for s in segments).strip()
    log.info("stt %.0fms: %r", (time.perf_counter() - t) * 1000, text)
    return {"text": text}


ABBREVIATIONS = {"Dr.": "Doctor", "Mr.": "Mister", "Mrs.": "Missus", "Ms.": "Miz"}


def speakable(text: str) -> str:
    for short, full in ABBREVIATIONS.items():
        text = re.sub(rf"\b{re.escape(short)}(?=\s)", full, text)
    return re.sub(r"\b(\d{1,2}):00\b", r"\1", text)


class SpeechRequest(BaseModel):
    input: str
    voice: str = "af_heart"
    response_format: str = "pcm"
    speed: float = 1.0


@app.post("/v1/audio/speech")
async def speech(req: SpeechRequest) -> StreamingResponse:
    if req.response_format not in ("pcm", "wav"):
        raise HTTPException(400, "only 'pcm' and 'wav' response formats are supported")
    text = speakable(req.input)

    async def stream():
        t, first = time.perf_counter(), True
        if req.response_format == "wav":
            samples, sr = kokoro().create(text, voice=req.voice, speed=req.speed, lang="en-us")
            buf = io.BytesIO()
            sf.write(buf, samples, sr, format="WAV", subtype="PCM_16")
            yield buf.getvalue()
            return
        async for samples, _sr in kokoro().create_stream(text, voice=req.voice, speed=req.speed, lang="en-us"):
            if first:
                log.info("tts first audio %.0fms", (time.perf_counter() - t) * 1000)
                first = False
            yield (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()

    media = "audio/pcm" if req.response_format == "pcm" else "audio/wav"
    return StreamingResponse(stream(), media_type=media)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("SPEECH_PORT", "8001")))


if __name__ == "__main__":
    main()
