from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_prefix": "INDEXTTS_", "env_file": ".env", "extra": "ignore"}

    # 模型路径
    cfg_path: str = "checkpoints/config.yaml"
    model_dir: str = "checkpoints"

    # 设备 / 优化
    use_fp16: bool = True
    device: str = ""  # 空字符串 = 自动检测
    use_cuda_kernel: bool = False
    use_deepspeed: bool = False
    use_accel: bool = False
    use_torch_compile: bool = False

    # 服务
    host: str = "0.0.0.0"
    port: int = 8000
    max_upload_size_mb: int = 20
    temp_dir: str = "api_server/api_temp"
    max_queue_size: int = 10

    # 服务端参考音频允许的目录（分号分隔，相对于项目根目录的路径）
    allowed_audio_dirs: str = "examples"

    # 文件清理
    temp_file_ttl_seconds: int = 300  # 上传文件 5 分钟后自动删除
