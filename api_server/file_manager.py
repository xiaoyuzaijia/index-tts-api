import os
import uuid
import time
import shutil
from pathlib import Path


class FileManager:
    """管理 API 请求的临时音频文件。"""

    def __init__(self, temp_dir: str, ttl_seconds: int = 300, allowed_audio_dirs: str = "examples", project_root: str | None = None):
        self.temp_dir = Path(temp_dir)
        self.ttl = ttl_seconds
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        if project_root:
            self.project_root = Path(project_root)
        else:
            # 默认项目根目录为 api_server 的上级目录
            self.project_root = Path(__file__).resolve().parent.parent

        self.allowed_dirs = []
        for d in allowed_audio_dirs.split(";"):
            d = d.strip()
            if d:
                p = Path(d)
                if not p.is_absolute():
                    p = self.project_root / p
                self.allowed_dirs.append(p.resolve())

    def resolve_audio_path(self, raw_path: str) -> Path:
        """将请求中的音频路径解析为服务端上的绝对路径，并校验安全性。

        - 如果是纯文件名（无路径分隔符），依次在 allowed_dirs 中查找
        - 如果是相对路径，相对于 project_root 解析
        - 如果是绝对路径，直接使用
        - 结果必须在 allowed_dirs 之一或其子目录下，且文件必须存在
        """
        # 清理路径中的不安全字符
        raw_path = raw_path.strip()

        is_abs = os.path.isabs(raw_path)
        has_sep = any(sep in raw_path for sep in ("/", "\\"))

        candidates = []

        if is_abs:
            candidates.append(Path(raw_path).resolve())
        elif has_sep:
            # 相对路径，相对于 project_root 解析
            candidates.append((self.project_root / raw_path).resolve())
        else:
            # 纯文件名，在 allowed_dirs 中依次查找
            for d in self.allowed_dirs:
                candidates.append((d / raw_path).resolve())

        for p in candidates:
            try:
                p = p.resolve()
            except (OSError, RuntimeError):
                continue
            if not p.is_file():
                continue
            # 安全检查：必须在 allowed_dirs 之一或其子目录下
            for d in self.allowed_dirs:
                try:
                    p.relative_to(d)
                    return p
                except ValueError:
                    continue

        # 未找到文件，给出清晰的错误信息
        searched = "\n  - ".join(str(c) for c in candidates)
        raise FileNotFoundError(
            f"服务端上未找到参考音频: {raw_path}\n"
            f"已搜索路径:\n  - {searched}\n"
            f"允许的目录: {[str(d) for d in self.allowed_dirs]}"
        )

    def save_upload(self, upload_file) -> Path:
        """将 UploadFile 以 UUID 文件名保存到临时目录。"""
        suffix = Path(upload_file.filename).suffix or ".wav"
        dest = self.temp_dir / f"{uuid.uuid4().hex}{suffix}"
        with open(dest, "wb") as f:
            shutil.copyfileobj(upload_file.file, f)
        return dest

    def cleanup(self, *paths: Path):
        """删除指定的临时文件。"""
        for p in paths:
            try:
                if p.exists():
                    p.unlink()
            except OSError:
                pass

    def cleanup_expired(self):
        """定期清理：删除超过 TTL 的文件。"""
        now = time.time()
        for f in self.temp_dir.iterdir():
            if f.is_file() and (now - f.stat().st_mtime) > self.ttl:
                try:
                    f.unlink()
                except OSError:
                    pass
