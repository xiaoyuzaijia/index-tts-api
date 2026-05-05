"""
IndexTTS2 API 功能测试脚本。

用法:
    # 先启动服务：
    python api.py

    # 然后运行测试：
    python tests/test_api.py
    python tests/test_api.py --base-url http://127.0.0.1:9000
    python tests/test_api.py --skip-inference  # 只测 health/info，不跑推理
"""

import argparse
import base64
import io
import json
import sys
import time
import wave
from pathlib import Path

import httpx
import torch

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_WAV = ROOT / "tests" / "sample_prompt.wav"       # 文件上传方式测试用
SAMPLE_PROMPT_NAME = "voice_09.wav"                  # 服务端路径方式测试用
OUTPUT_DIR = ROOT / "outputs"

PASS = "✓"
FAIL = "✗"


def green(s):
    return f"\033[32m{s}\033[0m"


def red(s):
    return f"\033[31m{s}\033[0m"


def bold(s):
    return f"\033[1m{s}\033[0m"


class APITester:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.results = []

    def _check(self, name: str, condition: bool, detail: str = ""):
        status = green(f"{PASS} {name}") if condition else red(f"{FAIL} {name}")
        if detail:
            status += f"  — {detail}"
        print(status)
        self.results.append((name, condition, detail))
        return condition

    async def test_health(self):
        """测试健康检查端点"""
        print(bold("\n── GET /api/v1/health ──"))
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(f"{self.base_url}/api/v1/health")
            self._check("HTTP 200", r.status_code == 200, f"status={r.status_code}")
            data = r.json()
            self._check("status 字段存在", "status" in data)
            self._check("model_loaded 字段存在", "model_loaded" in data)
            self._check("模型已加载", data.get("model_loaded"), str(data.get("device", "")))
            self._check("device 字段存在", "device" in data)
            self._check("cuda_available 字段存在", "cuda_available" in data)
            if data.get("cuda_available"):
                self._check("GPU 显存信息", data.get("gpu_memory_total_mb") is not None)
        except httpx.ConnectError:
            self._check("连接服务器", False, f"无法连接到 {self.base_url}，请确认服务已启动: python api.py")
        except Exception as e:
            self._check("请求成功", False, str(e))

    async def test_info(self):
        """测试模型信息端点"""
        print(bold("\n── GET /api/v1/info ──"))
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(f"{self.base_url}/api/v1/info")
            self._check("HTTP 200", r.status_code == 200, f"status={r.status_code}")
            data = r.json()
            self._check("version 字段", "version" in data, str(data.get("version")))
            self._check("sampling_rate 字段", "sampling_rate" in data, str(data.get("sampling_rate")))
            self._check("max_text_tokens 字段", "max_text_tokens" in data, str(data.get("max_text_tokens")))
            self._check("max_mel_tokens 字段", "max_mel_tokens" in data, str(data.get("max_mel_tokens")))
            self._check("emotion_order 字段", "emotion_order" in data, str(data.get("emotion_order")))
            self._check("情感列表长度=8", len(data.get("emotion_order", [])) == 8)
        except Exception as e:
            self._check("请求成功", False, str(e))

    async def test_tts_non_streaming(self, text: str, suffix: str = "", emo_vector: str | None = None):
        """测试非流式 TTS（使用服务端路径）"""
        emo_label = f", emo_vector={emo_vector}" if emo_vector else ""
        print(bold(f"\n── POST /api/v1/tts (text=\"{text[:20]}...\"{emo_label}) ──"))

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                data = {
                    "spk_audio_path": SAMPLE_PROMPT_NAME,
                    "text": text,
                    "max_text_tokens_per_segment": 80,
                    "top_p": 0.8,
                    "temperature": 0.8,
                }
                if emo_vector:
                    data["emo_vector"] = emo_vector
                    data["emo_alpha"] = 0.6

                r = await client.post(
                    f"{self.base_url}/api/v1/tts",
                    data=data,
                )

            self._check("HTTP 200", r.status_code == 200, f"status={r.status_code}")
            if r.status_code != 200:
                self._check("错误详情", False, r.text[:200])
                return

            content = r.content
            self._check("响应体非空", len(content) > 0, f"{len(content)} bytes")

            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            tag = suffix or text[:10].replace(" ", "_")
            out_path = OUTPUT_DIR / f"api_test_{tag}.wav"
            out_path.write_bytes(content)

            try:
                with wave.open(io.BytesIO(content), "rb") as wf:
                    channels = wf.getnchannels()
                    sample_width = wf.getsampwidth()
                    framerate = wf.getframerate()
                    n_frames = wf.getnframes()
                    duration = n_frames / framerate
                    self._check(
                        "WAV 格式有效",
                        True,
                        f"{channels}ch, {sample_width*8}bit, {framerate}Hz, {duration:.1f}s → {out_path.name}",
                    )
                    self._check("音频时长 > 0.1s", duration > 0.1, f"{duration:.2f}s")
            except Exception as e:
                self._check("WAV 解析", False, str(e))

        except Exception as e:
            self._check("请求成功", False, str(e))

    async def test_tts_file_upload(self, text: str, suffix: str = ""):
        """测试非流式 TTS（使用文件上传方式）"""
        print(bold(f"\n── POST /api/v1/tts (文件上传, text=\"{text[:20]}...\") ──"))

        if not SAMPLE_WAV.exists():
            self._check("参考音频", False, f"文件不存在: {SAMPLE_WAV}")
            return

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                files = {"spk_audio": (SAMPLE_WAV.name, open(SAMPLE_WAV, "rb"), "audio/wav")}
                data = {
                    "text": text,
                    "max_text_tokens_per_segment": 80,
                    "top_p": 0.8,
                    "temperature": 0.8,
                }

                r = await client.post(
                    f"{self.base_url}/api/v1/tts",
                    files=files,
                    data=data,
                )

            self._check("HTTP 200", r.status_code == 200, f"status={r.status_code}")
            if r.status_code != 200:
                self._check("错误详情", False, r.text[:200])
                return

            content = r.content
            self._check("响应体非空", len(content) > 0, f"{len(content)} bytes")

            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            tag = suffix or text[:10].replace(" ", "_")
            out_path = OUTPUT_DIR / f"api_test_{tag}.wav"
            out_path.write_bytes(content)

            try:
                with wave.open(io.BytesIO(content), "rb") as wf:
                    channels = wf.getnchannels()
                    sample_width = wf.getsampwidth()
                    framerate = wf.getframerate()
                    n_frames = wf.getnframes()
                    duration = n_frames / framerate
                    self._check(
                        "WAV 格式有效",
                        True,
                        f"{channels}ch, {sample_width*8}bit, {framerate}Hz, {duration:.1f}s → {out_path.name}",
                    )
                    self._check("音频时长 > 0.1s", duration > 0.1, f"{duration:.2f}s")
            except Exception as e:
                self._check("WAV 解析", False, str(e))

        except Exception as e:
            self._check("请求成功", False, str(e))

    async def test_tts_streaming(self, text: str, suffix: str = ""):
        """测试流式 TTS (SSE)（使用服务端路径）"""
        print(bold(f"\n── POST /api/v1/tts/stream (text=\"{text[:20]}...\") ──"))

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                data = {
                    "spk_audio_path": SAMPLE_PROMPT_NAME,
                    "text": text,
                    "max_text_tokens_per_segment": 80,
                    "top_p": 0.8,
                    "temperature": 0.8,
                }

                segments = []
                errors = []
                start_time = time.time()

                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/v1/tts/stream",
                    data=data,
                ) as response:
                    self._check("HTTP 200", response.status_code == 200, f"status={response.status_code}")
                    if response.status_code != 200:
                        content = await response.aread()
                        self._check("错误详情", False, content.decode()[:200])
                        return

                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            payload = line[6:]
                            try:
                                msg = json.loads(payload)
                                if "error" in msg:
                                    errors.append(msg["error"])
                                elif "done" in msg:
                                    break
                                elif "audio_base64" in msg:
                                    segments.append(msg)
                            except json.JSONDecodeError:
                                pass

                elapsed = time.time() - start_time

                self._check("收到音频片段", len(segments) > 0, f"{len(segments)} 个片段")
                self._check("无错误", len(errors) == 0, "; ".join(errors) if errors else "OK")

                # 保存每个片段到 outputs/ 并校验
                if segments:
                    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                    tag = suffix or text[:10].replace(" ", "_")
                    for seg in segments:
                        seg_idx = seg.get("segment", 0)
                        seg_bytes = base64.b64decode(seg["audio_base64"])

                        # 用 wave 解析各片段并统计
                        with wave.open(io.BytesIO(seg_bytes), "rb") as wf:
                            dur = wf.getnframes() / wf.getframerate()
                            # 计算 RMS 判断是语音还是静音
                            audio_np = (
                                torch.frombuffer(bytearray(seg_bytes[44:]), dtype=torch.int16)
                                .float()
                            )
                            rms = audio_np.pow(2).mean().sqrt().item() if audio_np.numel() > 0 else 0

                        seg_path = OUTPUT_DIR / f"api_stream_{tag}_seg{seg_idx}.wav"
                        seg_path.write_bytes(seg_bytes)

                        kind = "语音" if rms > 50 else "静音"
                        self._check(
                            f"  片段 #{seg_idx}: {dur:.2f}s, RMS={rms:.0f} ({kind})",
                            dur > 0.05,
                        )

                    self._check(
                        f"音频已保存到 outputs/api_stream_{tag}_seg*.wav",
                        True,
                    )

                if segments:
                    self._check(
                        "总耗时",
                        elapsed > 0,
                        f"{elapsed:.1f}s, {len(segments)} segments",
                    )

        except Exception as e:
            self._check("请求成功", False, str(e))

    async def test_error_cases(self):
        """测试错误处理"""
        print(bold("\n── 错误场景测试 ──"))
        async with httpx.AsyncClient(timeout=10) as client:
            # 空文本
            r = await client.post(
                f"{self.base_url}/api/v1/tts",
                data={"spk_audio_path": SAMPLE_PROMPT_NAME, "text": ""},
            )
            self._check("空文本 → 422", r.status_code == 422, f"status={r.status_code}")

            # 无效 emo_vector
            r = await client.post(
                f"{self.base_url}/api/v1/tts",
                data={
                    "spk_audio_path": SAMPLE_PROMPT_NAME,
                    "text": "测试",
                    "emo_vector": "1,2,3",
                },
            )
            self._check("无效 emo_vector → 422", r.status_code == 422, f"status={r.status_code}")

            # 无参考音频
            r = await client.post(
                f"{self.base_url}/api/v1/tts",
                data={"text": "测试"},
            )
            self._check("无参考音频 → 422", r.status_code == 422, f"status={r.status_code}")

            # 无效的服务端路径
            r = await client.post(
                f"{self.base_url}/api/v1/tts",
                data={"spk_audio_path": "nonexistent_file.wav", "text": "测试"},
            )
            self._check("无效 spk_audio_path → 500", r.status_code == 500, f"status={r.status_code}")

        # 服务端路径 + 上传文件同时提供（路径优先）— 会触发推理，需要更长超时
        if SAMPLE_WAV.exists():
            async with httpx.AsyncClient(timeout=120) as client:
                files = {"spk_audio": (SAMPLE_WAV.name, open(SAMPLE_WAV, "rb"), "audio/wav")}
                r = await client.post(
                    f"{self.base_url}/api/v1/tts",
                    files=files,
                    data={
                        "spk_audio_path": SAMPLE_PROMPT_NAME,
                        "text": "两个音频都提供时路径优先",
                    },
                )
                self._check("同时提供路径和文件 → 200", r.status_code == 200, f"status={r.status_code}")

    def summary(self):
        """打印测试摘要"""
        total = len(self.results)
        passed = sum(1 for _, ok, _ in self.results if ok)
        failed = total - passed

        print(bold(f"\n{'='*50}"))
        print(bold(f"测试结果: {passed}/{total} 通过"))
        if failed:
            print(red(f"{failed} 个失败:"))
            for name, ok, detail in self.results:
                if not ok:
                    print(red(f"  {FAIL} {name}") + (f" — {detail}" if detail else ""))
        else:
            print(green("全部通过!"))
        print()


async def main():
    parser = argparse.ArgumentParser(description="IndexTTS2 API 功能测试")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="API 服务地址 (默认: http://localhost:8000)",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="跳过推理测试，仅测试 health/info",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="打印响应详情",
    )
    args = parser.parse_args()

    print(bold(f"IndexTTS2 API 测试"))
    print(f"服务地址: {args.base_url}")
    print(f"参考音频 (服务端路径): {SAMPLE_PROMPT_NAME}")
    print(f"参考音频 (文件上传): {SAMPLE_WAV}")

    tester = APITester(args.base_url)

    # 基础端点测试（不需要推理）
    await tester.test_health()
    await tester.test_info()

    if args.skip_inference:
        print(bold(f"\n── 跳过推理测试 (--skip-inference) ──"))
    else:
        # 非流式 TTS
        await tester.test_tts_non_streaming("你好，欢迎使用 IndexTTS2 语音合成服务。", suffix="hello")

        # 带情感向量控制
        await tester.test_tts_non_streaming(
            "今天真是太开心了，天气真好！",
            suffix="happy",
            emo_vector="0.5,0,0,0,0,0,0,0",
        )

        # 文件上传方式
        await tester.test_tts_file_upload("你好，欢迎使用 IndexTTS2 语音合成服务。", suffix="upload")

        # 流式 TTS（使用较长文本以产生多个片段）
        long_text = (
            "清晨拉开窗帘，阳光洒在窗台的花艺礼盒上。"
            "薰衣草香薰蜡烛唤醒嗅觉，永生花束折射出晨露般的光泽。"
            "设计师将自然绽放美学融入每个细节。"
        )
        await tester.test_tts_streaming(long_text, suffix="stream")

        # 错误场景
        await tester.test_error_cases()

    tester.summary()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
