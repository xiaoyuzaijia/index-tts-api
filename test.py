import requests

with open("examples/voice_01.wav", "rb") as f:
    r = requests.post(
        "http://localhost:8000/api/v1/tts",
        files={"spk_audio": f},
        data={"text": "今天真是太开心了！",
              "use_emo_text": "true"},
        timeout=120,
    )
with open("outputs/gen.wav", "wb") as out:
    out.write(r.content)
print(f"生成完成，{len(r.content)} bytes")
