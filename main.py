"""
Gujarati Speech-to-Text API (FastAPI + SpeechRecognition + PyAudio)

Install:
    pip install fastapi uvicorn python-multipart SpeechRecognition PyAudio

Run:
    uvicorn main:app --reload

Endpoints:
    POST /speech-to-text/upload   -> upload any audio file (mp3, wav, m4a, webm...)
    GET  /speech-to-text/mic      -> record from the server's microphone (PyAudio)
"""

import logging
import os
import subprocess
import tempfile
import traceback

import imageio_ffmpeg  # bundles an ffmpeg binary, installed with pip
import speech_recognition as sr
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("stt")

app = FastAPI(title="Gujarati Speech to Text")

LANGUAGE = "gu-IN"  # Gujarati (India)
CHUNK_SECONDS = 30  # free Google API fails on long clips, so split
recognizer = sr.Recognizer()


def convert_to_wav(src_path: str, dst_path: str) -> None:
    """Convert any audio (mp3, m4a, webm, ogg, wav...) to 16 kHz mono WAV."""
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    result = subprocess.run(
        [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
         "-i", src_path, "-ar", "16000", "-ac", "1", dst_path],
        capture_output=True,
    )
    if result.returncode != 0:
        raise ValueError(result.stderr.decode(errors="ignore"))


def api_response(success: bool, code: int, data=None):
    """Standard JSON response: {success, code, data: []}"""
    return JSONResponse(
        status_code=code,
        content={"success": success, "code": code, "data": data if data is not None else []},
    )


def transcribe(audio: sr.AudioData) -> str:
    """Raises sr.UnknownValueError / sr.RequestError on failure."""
    return recognizer.recognize_google(audio, language=LANGUAGE)


def error_from_exception(e: Exception):
    log.error("STT error: %r", e)
    traceback.print_exc()
    if isinstance(e, sr.UnknownValueError):
        return api_response(False, 422, [{"message": "Could not understand the audio"}])
    if isinstance(e, sr.RequestError):
        return api_response(False, 503, [{"message": f"Speech service error: {e}"}])
    return api_response(False, 500, [{"message": str(e)}])


@app.get("/")
def health():
    return api_response(True, 200, [{"message": "Gujarati Speech to Text API is running"}])


@app.get("/check-internet")
def check_internet():
    """Tests whether the server can reach Google's speech endpoint."""
    import urllib.request

    try:
        status = urllib.request.urlopen("https://www.google.com", timeout=10).status
        return api_response(True, 200, [{"google_status": status}])
    except Exception as e:
        return api_response(False, 503, [{"message": f"Cannot reach Google: {e}"}])


@app.post("/speech-to-text/upload")
def speech_to_text_upload(file: UploadFile = File(...)):
    """Upload a WAV / AIFF / FLAC audio file and get Gujarati text."""
    try:
        contents = file.file.read()
        texts = []
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "input" + os.path.splitext(file.filename or "")[1])
            dst = os.path.join(tmp, "output.wav")
            with open(src, "wb") as f:
                f.write(contents)
            convert_to_wav(src, dst)

            with sr.AudioFile(dst) as source:
                total = source.DURATION
                log.info("Received %s, duration %.1fs", file.filename, total)
                pos = 0.0
                while pos < total:
                    audio = recognizer.record(source, duration=CHUNK_SECONDS)
                    pos += CHUNK_SECONDS
                    try:
                        texts.append(transcribe(audio))
                    except sr.UnknownValueError:
                        continue  # silent / unclear chunk, skip it

        if not texts:
            return api_response(False, 422, [{"message": "Could not understand the audio"}])

        return api_response(
            True, 200, [{"text": " ".join(texts), "language": LANGUAGE}]
        )
    except ValueError:
        return api_response(
            False, 400, [{"message": "Invalid or unsupported audio file"}]
        )
    except Exception as e:
        return error_from_exception(e)


@app.get("/speech-to-text/mic")
def speech_to_text_mic(timeout: int = 5, phrase_time_limit: int = 10):
    """Listen from the server's microphone (needs PyAudio) and return Gujarati text."""
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(
                source, timeout=timeout, phrase_time_limit=phrase_time_limit
            )
        text = transcribe(audio)
        return api_response(True, 200, [{"text": text, "language": LANGUAGE}])
    except sr.WaitTimeoutError:
        return api_response(False, 408, [{"message": "No speech detected (timeout)"}])
    except OSError:
        return api_response(False, 500, [{"message": "Microphone not available"}])
    except Exception as e:
        return error_from_exception(e)