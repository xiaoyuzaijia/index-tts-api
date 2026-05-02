import uuid
import time
import shutil
from pathlib import Path


class FileManager:
    """管理 API 请求的临时音频文件。"""

    def __init__(self, temp_dir: str, ttl_seconds: int = 300):
        self.temp_dir = Path(temp_dir)
        self.ttl = ttl_seconds
        self.temp_dir.mkdir(parents=True, exist_ok=True)

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
