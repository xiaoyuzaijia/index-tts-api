import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from indextts.infer_v2 import IndexTTS2

tts = IndexTTS2(
    cfg_path="checkpoints/config.yaml",
    model_dir="checkpoints",
    use_fp16=True,
    use_cuda_kernel=False,
    use_deepspeed=False,
)
text = "我都操着一口，这么流利的中文了，还有人在那问：啊请问主播是哪国人。"
tts.infer(
    spk_audio_prompt="examples/大家可能都是脾气太好了吧，我觉得像我肯定忍不了，虽然我。.wav",
    text=text,
    output_path="outputs/gen.wav",
    emo_alpha=0.6,
    use_emo_text=False,
    use_random=False,
    verbose=True,
)

tts.infer(
    spk_audio_prompt="examples/大家可能都是脾气太好了吧，我觉得像我肯定忍不了，虽然我。.wav",
    text=text,
    output_path="outputs/gen_emo.wav",
    emo_alpha=0.6,
    use_emo_text=True,
    use_random=False,
    verbose=True,
)
