import io
import json
import base64
import asyncio
import traceback
from pathlib import Path

import torch
import torchaudio
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse, Response

from api_server.models import TTSRequest, HealthResponse, InfoResponse, ErrorResponse
from api_server.service import TTSService

router = APIRouter(prefix="/api/v1")


# ── 依赖注入 ──────────────────────────────────────────────

def get_service(request: Request) -> TTSService:
    return request.app.state.service


def get_file_manager(request: Request):
    return request.app.state.file_manager


# ── 健康检查 ──────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
async def health(service: TTSService = Depends(get_service)):
    gpu_name = None
    gpu_total = None
    gpu_free = None
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        gpu_free = free / 1024**2
        gpu_total = total / 1024**2
        gpu_name = torch.cuda.get_device_name(0)

    return HealthResponse(
        status="ok" if service.is_loaded else "degraded",
        model_loaded=service.is_loaded,
        device=str(service.model.device) if service.is_loaded else "unknown",
        cuda_available=torch.cuda.is_available(),
        gpu_name=gpu_name,
        gpu_memory_total_mb=round(gpu_total, 1) if gpu_total else None,
        gpu_memory_free_mb=round(gpu_free, 1) if gpu_free else None,
    )


# ── 模型信息 ──────────────────────────────────────────────

@router.get("/info", response_model=InfoResponse)
async def info(service: TTSService = Depends(get_service)):
    if not service.is_loaded:
        raise HTTPException(status_code=503, detail="模型尚未加载完成")
    cfg = service.model.cfg
    return InfoResponse(
        version=str(service.model.model_version or "2.0"),
        sampling_rate=cfg.s2mel["preprocess_params"]["sr"],
        max_text_tokens=cfg.gpt.max_text_tokens,
        max_mel_tokens=cfg.gpt.max_mel_tokens,
        emotion_order=["happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"],
    )


# ── 表单解析依赖 ──────────────────────────────────────────

async def parse_tts_form(
    text: str = Form(..., description="要合成的文本"),
    emo_alpha: float = Form(1.0, ge=0.0, le=1.0),
    use_emo_text: bool = Form(False),
    emo_text: str | None = Form(None),
    use_random: bool = Form(False),
    emo_vector: str | None = Form(None),
    interval_silence: int = Form(200, ge=0, le=2000),
    max_text_tokens_per_segment: int = Form(120, ge=20, le=600),
    do_sample: bool = Form(True),
    top_p: float = Form(0.8, ge=0.0, le=1.0),
    top_k: int = Form(30, ge=0, le=200),
    temperature: float = Form(0.8, ge=0.0, le=2.0),
    length_penalty: float = Form(0.0, ge=-2.0, le=2.0),
    num_beams: int = Form(3, ge=1, le=10),
    repetition_penalty: float = Form(10.0, ge=0.1, le=20.0),
    max_mel_tokens: int = Form(1500, ge=50, le=1815),
) -> TTSRequest:
    """将 Form 字段解析并校验为 TTSRequest 模型。"""
    try:
        return TTSRequest(
            text=text,
            emo_alpha=emo_alpha,
            use_emo_text=use_emo_text,
            emo_text=emo_text,
            use_random=use_random,
            emo_vector=emo_vector,
            interval_silence=interval_silence,
            max_text_tokens_per_segment=max_text_tokens_per_segment,
            do_sample=do_sample,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            length_penalty=length_penalty,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            max_mel_tokens=max_mel_tokens,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


# ── 非流式 TTS ────────────────────────────────────────────

@router.post("/tts")
async def tts(
    spk_audio: UploadFile = File(..., description="说话人参考音频 (wav)"),
    emo_audio: UploadFile | None = File(None, description="情感参考音频 (wav)"),
    request: TTSRequest = Depends(parse_tts_form),
    service: TTSService = Depends(get_service),
    fm=Depends(get_file_manager),
):
    """非流式 TTS：上传参考音频和文本，返回完整 WAV 音频。"""
    if not service.is_loaded:
        raise HTTPException(status_code=503, detail="模型尚未加载完成")

    # 保存上传的音频文件
    spk_path = fm.save_upload(spk_audio)
    emo_path = fm.save_upload(emo_audio) if emo_audio else None

    try:
        loop = asyncio.get_running_loop()

        def _sync_infer():
            with service.inference_lock:
                kwargs = request.to_generation_kwargs()
                # infer() 已返回 (sampling_rate, numpy_array)
                return service.model.infer(
                    spk_audio_prompt=str(spk_path),
                    text=request.text,
                    output_path=None,
                    emo_audio_prompt=str(emo_path) if emo_path else None,
                    emo_alpha=request.emo_alpha,
                    emo_vector=request.emo_vector,
                    use_emo_text=request.use_emo_text,
                    emo_text=request.emo_text,
                    use_random=request.use_random,
                    interval_silence=request.interval_silence,
                    verbose=False,
                    max_text_tokens_per_segment=request.max_text_tokens_per_segment,
                    stream_return=False,
                    **kwargs,
                )

        sr, wav_np = await loop.run_in_executor(None, _sync_infer)

        # 将 numpy 数组转为 WAV 字节流
        wav_tensor = torch.from_numpy(wav_np).T  # (samples, channels) → (channels, samples)
        buf = io.BytesIO()
        torchaudio.save(buf, wav_tensor, sr, format="wav")
        buf.seek(0)

        return Response(content=buf.getvalue(), media_type="audio/wav")

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"推理失败: {e}")
    finally:
        fm.cleanup(spk_path, *([emo_path] if emo_path else []))


# ── 流式 TTS (SSE) ────────────────────────────────────────

@router.post("/tts/stream")
async def tts_stream(
    spk_audio: UploadFile = File(..., description="说话人参考音频 (wav)"),
    emo_audio: UploadFile | None = File(None, description="情感参考音频 (wav)"),
    request: TTSRequest = Depends(parse_tts_form),
    service: TTSService = Depends(get_service),
    fm=Depends(get_file_manager),
):
    """流式 TTS：上传参考音频和文本，通过 SSE 逐段返回音频。"""
    if not service.is_loaded:
        raise HTTPException(status_code=503, detail="模型尚未加载完成")

    spk_path = fm.save_upload(spk_audio)
    emo_path = fm.save_upload(emo_audio) if emo_audio else None

    async def generate_sse():
        try:
            loop = asyncio.get_running_loop()
            sr = service.model.cfg.s2mel["preprocess_params"]["sr"]
            kwargs = request.to_generation_kwargs()

            def _sync_stream():
                # 直接调用 infer_generator 以获取真正的生成器（而非 infer() 的 list()[0]）
                gen = service.model.infer_generator(
                    spk_audio_prompt=str(spk_path),
                    text=request.text,
                    output_path=None,
                    emo_audio_prompt=str(emo_path) if emo_path else None,
                    emo_alpha=request.emo_alpha,
                    emo_vector=request.emo_vector,
                    use_emo_text=request.use_emo_text,
                    emo_text=request.emo_text,
                    use_random=request.use_random,
                    interval_silence=request.interval_silence,
                    verbose=False,
                    max_text_tokens_per_segment=request.max_text_tokens_per_segment,
                    stream_return=True,
                    **kwargs,
                )
                chunks = []
                segment_idx = 0
                with service.inference_lock:
                    for item in gen:
                        if item is None:
                            continue
                        if isinstance(item, torch.Tensor):
                            # BigVGAN 输出 float32，值在 int16 范围，需转为 int16
                            audio_tensor = item.to(torch.int16)
                            buf = io.BytesIO()
                            torchaudio.save(buf, audio_tensor, sr, format="wav")
                            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                            chunks.append({
                                "segment": segment_idx,
                                "sample_rate": sr,
                                "audio_base64": b64,
                            })
                            segment_idx += 1
                return chunks

            chunks = await loop.run_in_executor(None, _sync_stream)

            # 发送所有片段
            for chunk in chunks:
                yield f"data: {json.dumps(chunk)}\n\n"

            # 发送结束信号
            yield f"data: {json.dumps({'done': True, 'total_segments': len(chunks)})}\n\n"

        except Exception as e:
            traceback.print_exc()
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            fm.cleanup(spk_path, *([emo_path] if emo_path else []))

    return StreamingResponse(
        generate_sse(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用 nginx 缓冲
        },
    )
