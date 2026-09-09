"""Shared test stubs for the external dependencies of bookchanger.py.

Installs lightweight stand-ins for fastapi / pydub / speech_recognition /
httpx ONLY when the real package isn't importable -- so the suite runs with
zero installs on a dev machine, and transparently uses the real packages
when they are installed (e.g. on CI with requirements.txt applied).

Imported by tests/conftest.py (pytest) and by tests/test_bookchanger.py
(direct / fallback-runner execution).
"""
from __future__ import annotations

import sys
import types


def _try_import(name: str):
    """Import a real module if present; record the result and return None."""
    try:
        module = __import__(name, fromlist=["*"])
        return module
    except Exception:
        return None


def install() -> dict:
    """
    Ensure bookchanger.py's imports resolve. Returns a dict describing what
    was stubbed vs. real, e.g. {"fastapi": "stub", "httpx": "real"}.
    Safe to call multiple times; never clobbers an already-importable module.
    """
    status: dict[str, str] = {}

    # ── fastapi (app object, route decorators, PlainTextResponse) ──────────
    fastapi = _try_import("fastapi")
    if fastapi is None:
        fastapi = types.ModuleType("fastapi")

        class FastAPI:
            def __init__(self, *args, **kwargs):
                pass

            def api_route(self, *args, **kwargs):
                def deco(fn):
                    return fn
                return deco

            def get(self, *args, **kwargs):
                def deco(fn):
                    return fn
                return deco

        fastapi.FastAPI = FastAPI
        fastapi.Request = object
        sys.modules["fastapi"] = fastapi
        status["fastapi"] = "stub"
    else:
        status["fastapi"] = "real"

    responses_mod = _try_import("fastapi.responses")
    if responses_mod is None or not hasattr(responses_mod, "PlainTextResponse"):
        responses_mod = types.ModuleType("fastapi.responses")
        responses_mod.parent = fastapi

        class PlainTextResponse:
            def __init__(self, content="", *args, **kwargs):
                self.body = content
                self.content = content

        responses_mod.PlainTextResponse = PlainTextResponse
        fastapi.responses = responses_mod
        sys.modules["fastapi.responses"] = responses_mod
        # For the real package, a PlainTextResponse may already exist.
        if status["fastapi"] == "real" and "fastapi.responses" not in sys.modules:
            status["fastapi.responses"] = "real"
        else:
            status["fastapi.responses"] = "stub" if status["fastapi"] == "stub" else "partial"
    else:
        status["fastapi.responses"] = "real"

    # ── pydub (AudioSegment is only touched inside process_audio_stt) ──────
    pydub = _try_import("pydub")
    if pydub is None:
        pydub = types.ModuleType("pydub")
        pydub.AudioSegment = type("AudioSegment", (), {})
        sys.modules["pydub"] = pydub
        status["pydub"] = "stub"
    else:
        status["pydub"] = "real"

    # ── speech_recognition (Recognizer + exception types) ──────────────────
    sr = _try_import("speech_recognition")
    if sr is None:
        sr = types.ModuleType("speech_recognition")
        sr.Recognizer = type("Recognizer", (), {})
        sr.AudioFile = type("AudioFile", (), {})
        sr.UnknownValueError = type("UnknownValueError", (Exception,), {})
        sr.RequestError = type("RequestError", (Exception,), {})
        sys.modules["speech_recognition"] = sr
        status["speech_recognition"] = "stub"
    else:
        status["speech_recognition"] = "real"

    # ── httpx (AsyncClient is constructed inside every REST helper) ────────
    httpx = _try_import("httpx")
    if httpx is None:
        httpx = types.ModuleType("httpx")
        httpx.AsyncClient = type("AsyncClient", (), {})
        sys.modules["httpx"] = httpx
        status["httpx"] = "stub"
    else:
        status["httpx"] = "real"

    return status
