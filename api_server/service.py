import threading

import torch

from indextts.infer_v2 import IndexTTS2


class TTSService:
    """
    IndexTTS2 的单例封装。

    保证：
    - IndexTTS2 只加载一次（单例）
    - 同时只有一个推理在运行（Lock）
    - 线程安全访问模型
    - 关闭时清理 GPU 显存
    """

    _instance: "TTSService | None" = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self, settings):
        self.settings = settings
        self._model: IndexTTS2 | None = None
        self._inference_lock = threading.Lock()
        self._loaded = False

    @classmethod
    def get_instance(cls, settings=None) -> "TTSService":
        """线程安全单例访问。"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    if settings is None:
                        raise RuntimeError("首次初始化需要提供 Settings")
                    cls._instance = cls(settings)
        return cls._instance

    def load_model(self):
        """启动时调用一次，阻塞直到模型完全加载。"""
        if self._loaded:
            return
        self._model = IndexTTS2(
            cfg_path=self.settings.cfg_path,
            model_dir=self.settings.model_dir,
            use_fp16=self.settings.use_fp16,
            device=self.settings.device or None,
            use_cuda_kernel=self.settings.use_cuda_kernel,
            use_deepspeed=self.settings.use_deepspeed,
            use_accel=self.settings.use_accel,
            use_torch_compile=self.settings.use_torch_compile,
        )
        self._loaded = True

    def unload_model(self):
        """关闭时调用。"""
        if self._model is not None:
            del self._model
            self._model = None
        self._loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def model(self) -> IndexTTS2:
        if not self._loaded or self._model is None:
            raise RuntimeError("模型未加载")
        return self._model

    @property
    def inference_lock(self) -> threading.Lock:
        return self._inference_lock

    @property
    def is_loaded(self) -> bool:
        return self._loaded and self._model is not None
