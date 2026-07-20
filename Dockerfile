FROM python:3.11-slim

# ffmpeg: safety net for pydub's audio decoding. process_audio_stt() tries an
# explicit WAV decode first specifically to avoid depending on ffmpeg, but
# falls back to pydub's auto-detect path for anything that isn't plain PCM
# WAV -- and that fallback does need ffmpeg. Cheap to install, and it
# removes an entire class of "STT silently returns nothing" failures.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY bookchanger.py .

# Render sets $PORT at runtime; 8000 is just the local-run default.
ENV PORT=8000
EXPOSE 8000

# Optional: override the YHM API base URL if it's ever not
# https://www.call2all.co.il/ym/api (bookchanger.py already defaults to
# that if YHM_API_BASE isn't set) --
# ENV YHM_API_BASE=https://www.call2all.co.il/ym/api

CMD ["sh", "-c", "uvicorn bookchanger:app --host 0.0.0.0 --port ${PORT}"]
