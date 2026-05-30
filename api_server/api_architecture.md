# IndexTTS2 API Server — 开发者文档

## 目录结构

```
api_server/
    __init__.py       # 空文件，标识为 Python 包
    config.py         # 配置管理（pydantic-settings）
    models.py         # Pydantic 数据模型（请求/响应）
    file_manager.py   # 临时文件管理与音频路径安全校验
    service.py        # IndexTTS2 模型单例封装 + 推理线程池调度
    routes.py         # API 路由定义（4 个端点）
    main.py           # FastAPI 应用工厂 + lifespan 生命周期
api.py                # 一键启动脚本（uvicorn 入口）
```

## 架构总览

```
                    ┌─────────────────────────────────────┐
                    │              api.py                  │
                    │  uvicorn.run("...main:create_app")  │
                    └──────────────┬──────────────────────┘
                                   │ factory=True
                    ┌──────────────▼──────────────────────┐
                    │             main.py                  │
                    │  create_app() → FastAPI             │
                    │  lifespan: 加载模型, 后台清理任务      │
                    └──────┬──────────────────┬───────────┘
                           │                  │
              ┌────────────▼────┐   ┌─────────▼──────────┐
              │   routes.py     │   │    service.py       │
              │  4 个端点       │──▶│   IndexTTS2 单例     │
              │  依赖注入        │   │   threading.Lock    │
              └──────┬─────────┘   └─────────┬───────────┘
                     │                       │
    ┌────────────────┼───────────────────────┼──────────────┐
    │                ▼                       ▼              │
    │  models.py   file_manager.py    indextts/infer_v2.py  │
    │  请求/响应    临时文件+路径安全    核心推理引擎          │
    └───────────────────────────────────────────────────────┘
```

**分层职责：**

| 层 | 模块 | 职责 |
|---|---|---|
| 启动 | `api.py` | CLI 参数解析，启动 uvicorn |
| 应用工厂 | `main.py` | 创建 FastAPI app，管理生命周期 |
| 路由 | `routes.py` | HTTP 端点，请求解析，响应序列化 |
| 服务 | `service.py` | 模型单例，线程安全推理调度 |
| 基础设施 | `file_manager.py` | 文件暂存，路径安全校验，过期清理 |
| 基础设施 | `config.py` | 环境变量 → 配置对象 |
| 数据模型 | `models.py` | 请求校验，响应结构 |

## 启动流程 (Lifespan)

应用使用 FastAPI 的 `lifespan` 模式管理启动/关闭：

```
create_app()
  │
  ├─ 1. Settings() 实例化 — 从环境变量/ .env 读取所有配置
  │
  ├─ 2. TTSService.get_instance(settings) — 创建模型服务单例
  │
  ├─ 3. FileManager(...) — 初始化临时文件管理器
  │
  ├─ 4. await loop.run_in_executor(None, service.load_model)
  │     └─ 在线程池中加载所有子模型（阻塞操作，不阻塞事件循环）
  │        GPT → s2mel → BigVGAN → CAMPPlus → QwenEmotion → BPE → TextNormalizer
  │
  ├─ 5. 挂载到 app.state
  │     app.state.service = service
  │     app.state.file_manager = file_manager
  │
  ├─ 6. 启动后台清理任务 (每 60 秒清理过期临时文件)
  │
  └─ 7. 关闭时: 取消清理任务 → 卸载模型 → torch.cuda.empty_cache()
```

**关键细节：**

- 模型加载在 `run_in_executor` 中执行，因为它是 CPU/磁盘密集型操作。这确保在漫长的模型加载过程中，健康检查端点仍然可以响应（返回 `degraded` 状态）。
- 模型只加载**一次**（单例），所有请求共享同一个模型实例。
- 如果进程被 SIGTERM/SIGKILL 终止，`lifespan` 的 `yield` 后清理代码**不会执行**（这是进程信号处理的硬限制，非 FastAPI 问题）。

## 请求生命周期

### 非流式 TTS (`POST /api/v1/tts`)

```
Client                     routes.py                  service.py           indextts
  │                           │                           │                    │
  │  POST /api/v1/tts         │                           │                    │
  │  multipart/form-data      │                           │                    │
  ├──────────────────────────▶│                           │                    │
  │                           │                           │                    │
  │                    ┌──────┴──────┐                    │                    │
  │                    │ parse_tts_  │  TTSRequest        │                    │
  │                    │ form()      │  (Pydantic校验)     │                    │
  │                    └──────┬──────┘                    │                    │
  │                           │                           │                    │
  │                    ┌──────┴──────┐                    │                    │
  │                    │ _resolve_   │                    │                    │
  │                    │ audio()     │  路径解析/文件保存   │                    │
  │                    └──────┬──────┘                    │                    │
  │                           │                           │                    │
  │                           │  run_in_executor()        │                    │
  │                           ├──────────────────────────▶│                    │
  │                           │                           │ acquire Lock        │
  │                           │                           ├───────────────────▶│
  │                           │                           │  model.infer(...)  │
  │                           │                           │◄───────────────────┤
  │                           │                           │ release Lock        │
  │                           │  (sr, wav_np)             │                    │
  │                           │◄──────────────────────────┤                    │
  │                           │                           │                    │
  │                    ┌──────┴──────┐                    │                    │
  │                    │ numpy→tensor│                    │                    │
  │                    │ →WAV bytes  │                    │                    │
  │                    └──────┬──────┘                    │                    │
  │                           │                           │                    │
  │   200 OK  audio/wav       │                           │                    │
  │◄──────────────────────────┤                           │                    │
  │                           │                           │                    │
  │                    ┌──────┴──────┐                    │                    │
  │                    │ fm.cleanup()│  删除上传的临时文件   │                    │
  │                    └─────────────┘                    │                    │
```

### 流式 TTS (`POST /api/v1/tts/stream`)

流式端点与非流式的核心差异：

1. 调用 `model.infer_generator()` 而非 `model.infer()`
2. 在持有 `inference_lock` 期间**一次性消费整个生成器**，收集所有片段到 `chunks` 列表
3. 退出锁后，通过 SSE (`text/event-stream`) 逐段发送

```
_sync_stream() 内部:
  │
  ├─ 1. 调用 infer_generator(stream_return=True, ...) → 返回生成器
  │
  ├─ 2. with service.inference_lock:        ← 锁住整段生成过程
  │       for item in gen:
  │           if isinstance(item, Tensor):
  │               转为 int16 → WAV bytes → base64
  │               追加到 chunks[]
  │
  ├─ 3. 返回 chunks[] （锁已释放）
  │
  └─ 4. generate_sse() async generator:
          for chunk in chunks:
              yield f"data: {json}\n\n"
          yield f"data: {{done: true}}\n\n"
```

**SSE 消息格式：**

```json
// 音频片段
{"segment": 0, "sample_rate": 24000, "audio_base64": "UklGRiIw..."}

// 结束信号
{"done": true, "total_segments": 3}

// 错误
{"error": "错误描述"}
```

## 并发模型与线程安全

### 问题

IndexTTS2 内部使用实例变量作为缓存（`self.cache_spk_cond`、`self.cache_emo_cond`），这意味着**同一个模型实例不能同时处理多个推理请求**。

### 解决方案

`TTSService` 使用 `threading.Lock` 实现请求序列化：

```python
# service.py
class TTSService:
    _inference_lock = threading.Lock()  # 所有请求共享同一把锁

# routes.py — 非流式
def _sync_infer():
    with service.inference_lock:      # 阻塞等待
        return service.model.infer(...)

# routes.py — 流式
def _sync_stream():
    with service.inference_lock:      # 整段生成过程持有锁
        for item in gen:
            ...
```

### 为什么推理在线程池中执行？

- FastAPI 的 `async def` 端点运行在 asyncio 事件循环中
- `model.infer()` 是同步阻塞调用（涉及 GPU 计算）
- 如果在事件循环中直接调用，会阻塞所有其他请求（包括健康检查）
- 使用 `loop.run_in_executor(None, sync_fn)` 将它移入线程池，让事件循环保持响应

### 并发行为总结

```
请求 A 到达 ──▶ 获取 Lock ──▶ 推理中... ──▶ 释放 Lock
请求 B 到达 ──▶ 等待 Lock... ──▶ 获取 Lock ──▶ 推理中... ──▶ 释放 Lock
请求 C 到达 ──▶ 等待 Lock... ──▶ 获取 Lock ──▶ 推理中... ──▶ 释放 Lock
```

请求被**序列化**处理。健康检查和 /info 端点不受影响（它们不需要锁）。

## API 端点参考

| 方法 | 路径 | Content-Type | 说明 |
|------|------|-------------|------|
| GET | `/api/v1/health` | application/json | 健康检查（始终可用） |
| GET | `/api/v1/info` | application/json | 模型元数据（需模型已加载） |
| POST | `/api/v1/tts` | multipart/form-data → audio/wav | 非流式 TTS |
| POST | `/api/v1/tts/stream` | multipart/form-data → text/event-stream | 流式 TTS (SSE) |

### 参考音频的两种指定方式

| 方式 | 参数 | 用途 |
|------|------|------|
| 服务端路径 | `spk_audio_path` / `emo_audio_path` | 指定服务端上已有的音频文件 |
| 文件上传 | `spk_audio` / `emo_audio` | 客户端上传音频文件 |

**优先级：** 服务端路径 > 文件上传。当同时提供时，使用服务端路径并忽略上传文件。

**服务端路径安全校验（`FileManager.resolve_audio_path`）：**

1. 纯文件名（无路径分隔符）→ 在 `allowed_audio_dirs` 中依次查找
2. 相对路径 → 相对于项目根目录解析
3. 绝对路径 → 直接使用
4. 结果必须在 `allowed_audio_dirs` 之一或其子目录下

这防止了路径遍历攻击（如 `../../etc/passwd`）。

## 配置参考

所有配置通过环境变量设置，前缀 `INDEXTTS_`。也支持 `.env` 文件。

| 环境变量 | 类型 | 默认值 | 说明 |
|----------|------|--------|------|
| `INDEXTTS_CFG_PATH` | str | `checkpoints/config.yaml` | 模型配置文件路径 |
| `INDEXTTS_MODEL_DIR` | str | `checkpoints` | 模型权重目录 |
| `INDEXTTS_USE_FP16` | bool | `true` | 是否使用 FP16 推理 |
| `INDEXTTS_DEVICE` | str | `""` | 设备（空=自动检测，或 `cuda:0`） |
| `INDEXTTS_USE_CUDA_KERNEL` | bool | `false` | BigVGAN CUDA kernel |
| `INDEXTTS_USE_DEEPSPEED` | bool | `false` | DeepSpeed 加速 |
| `INDEXTTS_USE_ACCEL` | bool | `false` | 通用加速 |
| `INDEXTTS_USE_TORCH_COMPILE` | bool | `false` | torch.compile 优化 |
| `INDEXTTS_HOST` | str | `0.0.0.0` | 监听地址 |
| `INDEXTTS_PORT` | int | `8000` | 监听端口 |
| `INDEXTTS_MAX_UPLOAD_SIZE_MB` | int | `20` | 上传文件大小上限 |
| `INDEXTTS_TEMP_DIR` | str | `api_server/api_temp` | 临时文件目录 |
| `INDEXTTS_MAX_QUEUE_SIZE` | int | `10` | 请求队列上限 |
| `INDEXTTS_ALLOWED_AUDIO_DIRS` | str | `examples` | 允许的音频目录（分号分隔） |
| `INDEXTTS_TEMP_FILE_TTL_SECONDS` | int | `300` | 临时文件过期时间（秒） |

## 数据模型

### TTSRequest — 推理参数

通用推理请求模型，非流式和流式端点共用。通过 `parse_tts_form()` 依赖将 multipart/form-data 字段解析为 Pydantic 模型。

`to_generation_kwargs()` 方法提取 GPT 采样参数子集，直接传递给 `model.infer()`。

**情感控制支持三种模式：**

| 模式 | 参数 | 行为 |
|------|------|------|
| 手动向量 | `emo_vector` | 用户指定 8 维情感分布 |
| 文本推断 | `use_emo_text=true` | 从 `text` 自动检测情感 |
| 随机采样 | `use_random=true` | 从标准分布随机采样情感矩阵 |

`emo_alpha` 控制情感混合权重（0 = 只用参考音频情感，1 = 完全使用指定情感）。

### 响应模型

- **HealthResponse**: 服务状态 + GPU 显存信息
- **InfoResponse**: 模型版本、采样率、token 限制、情感列表
- **ErrorResponse**: 统一错误格式 `{detail, error_code}`

## 临时文件管理

`FileManager` 负责管理客户端上传的音频文件的完整生命周期：

```
save_upload(upload_file)
  │
  ├─ 生成 UUID 文件名 → {temp_dir}/{uuid}.wav
  ├─ 写入磁盘
  └─ 返回 Path

cleanup(*paths)
  │
  └─ 删除指定文件（推理完成后在 finally 中调用）

cleanup_expired()
  │
  └─ 删除 mtime 超过 TTL 的文件（后台每 60 秒执行）
```

**设计要点：**
- 上传文件以 UUID 命名，避免文件名冲突
- `finally` 块确保无论推理成功/失败，临时文件都会被删除
- 后台 `cleanup_expired()` 防止异常退出（如 kill -9）导致残留文件堆积

## 错误处理策略

```
routes.py 错误处理层次:

1. Pydantic 校验失败 → 422 Unprocessable Entity
   └─ Field 约束: min_length, ge, le
   └─ emo_vector 自定义校验: 必须是 8 个 [0,1] 浮点数

2. 业务逻辑错误 → 422 / 503
   ├─ 缺少参考音频 → 422
   ├─ 模型未加载 → 503
   └─ 音频文件不存在 → 500 (FileNotFoundError)

3. 推理异常 → 500 Internal Server Error
   └─ traceback.print_exc() 输出到 stderr
   └─ 客户端收到 {detail: "推理失败: ..."}

4. HTTPException 被原样透传（不包装）
```

## 测试

测试文件：`tests/test_api.py`

**运行方式：**

```bash
# 终端 1: 启动服务
python api.py

# 终端 2: 运行测试
python tests/test_api.py
python tests/test_api.py --skip-inference   # 仅 health/info
python tests/test_api.py --base-url http://127.0.0.1:9000
```

**测试覆盖：**

| 测试 | 内容 |
|------|------|
| test_health | 状态码、字段完整性、GPU 信息 |
| test_info | 模型元数据、情感列表 |
| test_tts_non_streaming | 非流式 TTS，WAV 有效性校验 |
| test_tts_non_streaming + emo_vector | 情感向量控制 |
| test_tts_auto_emotion | 从文本自动推断情感 |
| test_tts_file_upload | 文件上传方式 |
| test_tts_streaming | SSE 流式 TTS，逐段校验 |
| test_error_cases | 空文本 → 422，无效 emo_vector → 422，无参考音频 → 422，无效路径 → 500，路径+文件同时提供路径优先 |

## 设计原则

1. **模型即单例** — 启动时加载一次，全生命周期复用。绝不按请求加载。
2. **推理串行化** — `threading.Lock` 确保模型内部缓存不会被并发破坏。
3. **异步不阻塞** — 所有推理进入线程池 (`run_in_executor`)，保持事件循环响应。
4. **路径安全** — 服务端音频路径必须在白名单目录内，防止目录遍历。
5. **临时文件必清理** — `finally` 块 + 后台定期清理双重保障。
6. **配置外部化** — 所有配置通过环境变量注入，无硬编码。
7. **工厂模式启动** — `create_app()` 函数式创建 FastAPI 实例，方便测试。
