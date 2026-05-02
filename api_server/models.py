from pydantic import BaseModel, Field, field_validator


class TTSRequest(BaseModel):
    """所有非文件表单字段（/api/v1/tts 和 /api/v1/tts/stream 共用）"""

    text: str = Field(..., min_length=1, max_length=5000, description="要合成的文本")

    # 情感控制
    emo_alpha: float = Field(default=1.0, ge=0.0, le=1.0, description="情感混合权重")
    use_emo_text: bool = Field(default=False, description="从文本检测情感")
    emo_text: str | None = Field(default=None, max_length=1000, description="情感描述文本")
    use_random: bool = Field(default=False, description="随机情感矩阵采样")

    # 逗号分隔的 8 个情感值 [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
    emo_vector: str | None = Field(
        default=None,
        description="逗号分隔的 8 个情感值 [高兴,愤怒,悲伤,恐惧,反感,低落,惊讶,平静]",
    )

    # 音频生成
    interval_silence: int = Field(default=200, ge=0, le=2000, description="段间静音 (ms)")
    max_text_tokens_per_segment: int = Field(default=120, ge=20, le=600)

    # 生成参数 (GPT 采样)
    do_sample: bool = Field(default=True)
    top_p: float = Field(default=0.8, ge=0.0, le=1.0)
    top_k: int = Field(default=30, ge=0, le=200)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    length_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    num_beams: int = Field(default=3, ge=1, le=10)
    repetition_penalty: float = Field(default=10.0, ge=0.1, le=20.0)
    max_mel_tokens: int = Field(default=1500, ge=50, le=1815)

    @field_validator("emo_vector")
    @classmethod
    def validate_emo_vector(cls, v: str | None) -> list[float] | None:
        if v is None or v.strip() == "":
            return None
        parts = [x.strip() for x in v.split(",")]
        if len(parts) != 8:
            raise ValueError("emo_vector 必须恰好包含 8 个逗号分隔的值")
        vals = []
        for p in parts:
            try:
                f = float(p)
            except ValueError:
                raise ValueError(f"emo_vector 中包含无效数字: {p}")
            if f < 0.0 or f > 1.0:
                raise ValueError(f"emo_vector 值必须在 [0.0, 1.0] 范围内，当前值: {f}")
            vals.append(f)
        return vals

    def to_generation_kwargs(self) -> dict:
        return {
            "do_sample": self.do_sample,
            "top_p": self.top_p,
            "top_k": self.top_k if self.top_k > 0 else None,
            "temperature": self.temperature,
            "length_penalty": self.length_penalty,
            "num_beams": self.num_beams,
            "repetition_penalty": self.repetition_penalty,
            "max_mel_tokens": self.max_mel_tokens,
        }


class HealthResponse(BaseModel):
    status: str  # "ok" 或 "degraded"
    model_loaded: bool
    device: str
    cuda_available: bool
    gpu_name: str | None = None
    gpu_memory_total_mb: float | None = None
    gpu_memory_free_mb: float | None = None


class InfoResponse(BaseModel):
    version: str
    sampling_rate: int
    max_text_tokens: int
    max_mel_tokens: int
    emotion_order: list[str]


class ErrorResponse(BaseModel):
    detail: str
    error_code: str
