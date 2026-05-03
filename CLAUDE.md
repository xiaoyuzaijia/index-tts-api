# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

IndexTTS2 is a zero-shot TTS model supporting voice cloning, emotion control, and duration-controlled auto-regressive speech synthesis in Chinese and English. The repo includes a Gradio WebUI, a FastAPI REST API server, and the core inference engine.

## Common Commands

```bash
# Install all dependencies (use uv, not pip/conda)
uv sync --all-extras

# Run the Gradio WebUI
uv run webui.py

# Run the FastAPI server
python api.py
python api.py --port 9000 --host 127.0.0.1

# Run API tests (server must be running separately)
python tests/test_api.py
python tests/test_api.py --skip-inference

# Run a single Python script within the uv environment
PYTHONPATH="$PYTHONPATH:." uv run <script.py>
```

## Architecture

### Core Model (`indextts/`)

- **`infer_v2.py`** — `IndexTTS2` class. Main entry point for TTS inference. Constructor loads all sub-models: GPT token generator, s2mel spectrogram decoder, BigVGAN vocoder, CAMPPlus speaker encoder, QwenEmotion text analyzer, BPE tokenizer, TextNormalizer.
- **`infer()`** — Runs full inference pipeline synchronously. Returns `(sampling_rate, numpy_array)` when `output_path=None`. Uses internal caches (`cache_spk_cond`, `cache_emo_cond`) that are NOT thread-safe.
- **`infer_generator()`** — Generator version of `infer()`. With `stream_return=True`, yields individual audio segment tensors as they're generated. Always yields in pairs: `(audio_segment, silence)` for each text segment.

### API Server (`api_server/`)

```
api_server/
    config.py          # pydantic-settings, INDEXTTS_ env prefix
    models.py          # Pydantic schemas: TTSRequest, HealthResponse, InfoResponse
    file_manager.py    # Temp file staging + server audio path resolution with security checks
    service.py         # IndexTTS2 singleton with threading.Lock for thread safety
    routes.py          # 4 endpoints: health, info, tts, tts/stream
    main.py            # FastAPI app factory + lifespan (model loaded at startup)
api.py                 # One-command launcher
```

### Key Design Decisions

1. **Model is a singleton** — loaded once at app startup (lifespan), never per-request.
2. **`threading.Lock` serializes all inference** — the model's internal caches are not thread-safe.
3. **Inference runs in thread pool** via `loop.run_in_executor()` — keeps asyncio event loop responsive so health checks work during generation.
4. **`infer()` already returns `(sr, wav_np)`** — do NOT wrap with extra `list(...)[0]`.
5. **Streaming must use `infer_generator()` directly** — `infer()` with `stream_return=True` consumes the generator into a list and returns only the first element.
6. **BigVGAN outputs float32 in int16 range** (not [-1, 1]). Use `.to(torch.int16)` before `torchaudio.save()`.
7. **Two ways to specify reference audio**: `spk_audio_path` / `emo_audio_path` (server-side path, preferred) or `spk_audio` / `emo_audio` (file upload). Server path takes priority when both given. Paths are validated to stay within `allowed_audio_dirs` (default: `examples/`).

### Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/v1/health` | Liveness: model loaded, GPU info |
| GET | `/api/v1/info` | Model metadata: version, sampling rate, token limits, emotion order |
| POST | `/api/v1/tts` | Non-streaming TTS. multipart/form-data → `audio/wav` |
| POST | `/api/v1/tts/stream` | Streaming TTS. multipart/form-data → SSE (base64 WAV per segment) |

### Dependencies

- **Package manager**: `uv` only (pip/conda unsupported)
- **API extras**: `uv sync --extra api` adds fastapi, uvicorn, python-multipart, pydantic-settings
- **PyTorch**: CUDA 12.8 from pytorch-cuda index
- **Model files**: Expected in `checkpoints/` (download via huggingface-hub or modelscope)
