# IndexTTS2 API Server

本项目基于 [IndexTTS2](https://github.com/index-tts/index-tts) 构建，提供了一个开箱即用的 FastAPI REST API 服务端，支持非流式和流式语音合成接口。

IndexTTS2 是一个支持情感表达和时长控制的自回归零样本文本转语音（TTS）模型，支持中文和英文。基于单一参考音频即可克隆音色，同时支持通过情感参考音频、情感向量、或文本描述来控制合成语音的情感。

详细的项目文档、模型下载和环境配置请参阅 [docs/README.md](docs/README.md)。

## 快速开始

```bash
# 安装全部依赖
uv sync --all-extras

# 或安装api相关依赖
uv sync --extra api

# 启动 FastAPI 服务
python api.py
```

```

## FastAPI 接口

启动服务后，接口默认运行在 `http://localhost:8000`。

```bash
python api.py                  # 默认 0.0.0.0:8000
python api.py --port 9000      # 自定义端口
```

### 端点一览

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/v1/health` | 健康检查 |
| GET | `/api/v1/info` | 模型信息 |
| POST | `/api/v1/tts` | 非流式 TTS（返回完整 WAV） |
| POST | `/api/v1/tts/stream` | 流式 TTS（SSE 逐段返回） |

### GET `/api/v1/health`

返回服务状态和 GPU 信息。

```bash
curl http://localhost:8000/api/v1/health
```

```python
import requests

r = requests.get("http://localhost:8000/api/v1/health")
print(r.json())
```

响应示例：

```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cuda:0",
  "cuda_available": true,
  "gpu_name": "NVIDIA GeForce RTX 4090",
  "gpu_memory_total_mb": 24564.0,
  "gpu_memory_free_mb": 18123.5
}
```

### GET `/api/v1/info`

返回模型元信息。

```bash
curl http://localhost:8000/api/v1/info
```

```python
import requests

r = requests.get("http://localhost:8000/api/v1/info")
print(r.json())
```

响应示例：

```json
{
  "version": "2.0",
  "sampling_rate": 22050,
  "max_text_tokens": 600,
  "max_mel_tokens": 1815,
  "emotion_order": ["happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"]
}
```

### POST `/api/v1/tts`

非流式语音合成。可通过服务端路径或上传文件指定参考音频，返回完整 WAV。

#### 基本示例

```bash
curl -X POST http://localhost:8000/api/v1/tts \
  -F "spk_audio_path=voice_01.wav" \
  -F "text=你好，欢迎使用 IndexTTS2 语音合成服务。" \
  -o output.wav
```

```python
import requests

r = requests.post(
    "http://localhost:8000/api/v1/tts",
    data={
        "spk_audio_path": "voice_01.wav",
        "text": "你好，欢迎使用 IndexTTS2 语音合成服务。",
    },
    timeout=120,
)
with open("output.wav", "wb") as out:
    out.write(r.content)
```

指定上传音频文件（替代 `spk_audio_path`）：

```bash
curl -X POST http://localhost:8000/api/v1/tts \
  -F "spk_audio=@examples/voice_01.wav" \
  -F "text=你好，欢迎使用 IndexTTS2 语音合成服务。" \
  -o output.wav
```

#### 从文本自动决定情感的示例

```bash
curl -X POST http://localhost:8000/api/v1/tts \
  -F "spk_audio_path=voice_01.wav" \
  -F "text=今天真是太开心了！" \
  -F "use_emo_text=true" \
  -o happy.wav
```

```python
import requests

r = requests.post(
    "http://localhost:8000/api/v1/tts",
    data={
        "spk_audio_path": "voice_01.wav",
        "text": "今天真是太开心了！",
        "use_emo_text": "true",
    },
    timeout=120,
)
with open("happy.wav", "wb") as out:
    out.write(r.content)
print(f"生成完成，{len(r.content)} bytes")
```

#### 带情感向量的示例

```bash
curl -X POST http://localhost:8000/api/v1/tts \
  -F "spk_audio_path=voice_01.wav" \
  -F "text=今天真是太开心了！" \
  -F "emo_vector=0.6,0,0,0,0,0,0,0" \
  -F "emo_alpha=0.7" \
  -o happy.wav
```

```python
import requests

r = requests.post(
    "http://localhost:8000/api/v1/tts",
    data={
        "spk_audio_path": "voice_01.wav",
        "text": "今天真是太开心了！",
        "emo_vector": "0.6,0,0,0,0,0,0,0",
        "emo_alpha": 0.7,
    },
    timeout=120,
)
with open("happy.wav", "wb") as out:
    out.write(r.content)
```

#### 带情感参考音频的示例

```bash
curl -X POST http://localhost:8000/api/v1/tts \
  -F "spk_audio_path=voice_01.wav" \
  -F "emo_audio_path=emo_sad.wav" \
  -F "text=酒楼丧尽天良，开始借机竞拍房间。" \
  -F "emo_alpha=0.9" \
  -o sad.wav
```

```python
import requests

r = requests.post(
    "http://localhost:8000/api/v1/tts",
    data={
        "spk_audio_path": "voice_01.wav",
        "emo_audio_path": "emo_sad.wav",
        "text": "酒楼丧尽天良，开始借机竞拍房间。",
        "emo_alpha": 0.9,
    },
    timeout=120,
)
with open("sad.wav", "wb") as out:
    out.write(r.content)
```

### POST `/api/v1/tts/stream`

流式语音合成。参数与 `/api/v1/tts` 完全相同，返回 SSE（Server-Sent Events）流，每段音频以 base64 编码的 WAV 格式逐段发送。

```bash
curl -X POST http://localhost:8000/api/v1/tts/stream \
  -F "spk_audio_path=voice_01.wav" \
  -F "text=清晨拉开窗帘，阳光洒在窗台。薰衣草香薰蜡烛唤醒嗅觉，永生花束折射出晨露般的光泽。"
```

```python
import json

import requests

r = requests.post(
    "http://localhost:8000/api/v1/tts/stream",
    data={
        "spk_audio_path": "voice_01.wav",
        "text": "清晨拉开窗帘，阳光洒在窗台。薰衣草香薰蜡烛唤醒嗅觉，永生花束折射出晨露般的光泽。",
        "max_text_tokens_per_segment": 80,
    },
    stream=True,
    timeout=120,
)
for line in r.iter_lines(decode_unicode=True):
    if line and line.startswith("data: "):
        msg = json.loads(line[6:])
        if "done" in msg:
            print(f"流式结束，共 {msg['total_segments']} 段")
            break
        if "audio_base64" in msg:
            import base64
            audio_bytes = base64.b64decode(msg["audio_base64"])
            with open(f"segment_{msg['segment']}.wav", "wb") as seg:
                seg.write(audio_bytes)
            print(f"片段 #{msg['segment']}: {len(audio_bytes)} bytes")
```

SSE 事件格式：

```
data: {"segment": 0, "sample_rate": 22050, "audio_base64": "UklGRiA..."}
data: {"segment": 1, "sample_rate": 22050, "audio_base64": "UklGRiB..."}
data: {"done": true, "total_segments": 2}
```

每个 `audio_base64` 是完整可独立播放的 WAV 文件（base64 编码），客户端可直接解码和播放。

#### 请求参数

请求使用 `multipart/form-data` 格式。参数分为三类：

**音频来源**（选择其中一种即可，服务端路径为默认方式）：

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `spk_audio_path` | string | 是（或提供 spk_audio） | 服务端 `examples/` 下的音频文件名或相对路径 |
| `emo_audio_path` | string | 否 | 服务端情感参考音频路径 |
| `spk_audio` | file | 是（或提供 spk_audio_path） | 说话人参考音频 — 上传文件 |
| `emo_audio` | file | 否 | 情感参考音频 — 上传文件 |

> `spk_audio_path` / `emo_audio_path` 优先于 `spk_audio` / `emo_audio`。路径只需传文件名（如 `voice_01.wav`），会自动在 `examples/` 目录下查找。

**表单字段类**——curl 用 `-F "name=value"` 传入，Python 放入 `requests.post(data={...})`：

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `text` | string | 是 | - | 要合成的文本（最大 5000 字符） |
| `emo_alpha` | float | 否 | 1.0 | 情感混合权重（0.0 ~ 1.0） |
| `emo_vector` | string | 否 | - | 8 个情感值，逗号分隔，范围 0.0~1.0。顺序：高兴、愤怒、悲伤、恐惧、厌恶、低落、惊讶、平静 |
| `use_emo_text` | bool | 否 | false | 从文本/emo_text 自动检测情感 |
| `emo_text` | string | 否 | - | 情感描述文本（`use_emo_text=true` 时使用） |
| `use_random` | bool | 否 | false | 随机情感采样 |
| `interval_silence` | int | 否 | 200 | 段间静音时长（毫秒） |
| `max_text_tokens_per_segment` | int | 否 | 120 | 分句最大 token 数（20 ~ 600） |
| `do_sample` | bool | 否 | true | 是否采样生成 |
| `top_p` | float | 否 | 0.8 | nucleus 采样阈值（0.0 ~ 1.0） |
| `top_k` | int | 否 | 30 | top-k 采样（0 ~ 200，0 表示不使用） |
| `temperature` | float | 否 | 0.8 | 采样温度（0.0 ~ 2.0） |
| `num_beams` | int | 否 | 3 | beam search 数量（1 ~ 10） |
| `repetition_penalty` | float | 否 | 10.0 | 重复惩罚系数（0.1 ~ 20.0） |
| `max_mel_tokens` | int | 否 | 1500 | 最大 mel token 数（50 ~ 1815） |

### 配置

所有配置可通过环境变量覆盖（前缀 `INDEXTTS_`）：

```bash
# 模型路径
INDEXTTS_CFG_PATH=checkpoints/config.yaml
INDEXTTS_MODEL_DIR=checkpoints

# 设备与优化
INDEXTTS_USE_FP16=true
INDEXTTS_USE_CUDA_KERNEL=true

# 服务
INDEXTTS_HOST=0.0.0.0
INDEXTTS_PORT=8000
INDEXTTS_TEMP_DIR=api_server/api_temp
INDEXTTS_MAX_UPLOAD_SIZE_MB=20

# 服务端参考音频路径（分号分隔多个目录）
INDEXTTS_ALLOWED_AUDIO_DIRS=examples
```

## 运行测试

```bash
# 终端 1：启动服务
python api.py

# 终端 2：运行测试
python tests/test_api.py
python tests/test_api.py --skip-inference   # 跳过推理，仅测试 health/info
python tests/test_api.py --base-url http://localhost:9000
```

## 衍生作品声明

本仓库是基于 [bilibili/index-tts](https://github.com/index-tts/index-tts) 的衍生作品，使用 bilibili Model Use License Agreement 授权。对原始模型的任何修改未经原始权利人认可、担保或保证，原始权利人不承担与此衍生作品相关的任何责任。

## 引用

```bibtex
@article{zhou2025indextts2,
  title={IndexTTS2: A Breakthrough in Emotionally Expressive and Duration-Controlled Auto-Regressive Zero-Shot Text-to-Speech},
  author={Siyi Zhou, Yiquan Zhou, Yi He, Xun Zhou, Jinchao Wang, Wei Deng, Jingchen Shu},
  journal={arXiv preprint arXiv:2506.21619},
  year={2025}
}
```
