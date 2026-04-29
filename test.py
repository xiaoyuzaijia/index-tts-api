import os
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"

from indextts.infer_v2 import IndexTTS2


tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_fp16=True, use_cuda_kernel=False, use_deepspeed=False)
text = "欲买桂花同载酒，终不似，少年游。"
tts.infer(spk_audio_prompt='examples/voice_12.wav', text=text, output_path="outputs/gen.wav", emo_alpha=0.6, use_emo_text=True, use_random=False, verbose=True)