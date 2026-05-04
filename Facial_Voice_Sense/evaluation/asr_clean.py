"""SenseVoice / FunASR 原始文本中的语言与情绪标签解析（与 webtest.clean_asr_text 一致）。"""


def clean_asr_text(raw_text: str) -> tuple[str, str, str]:
    language = ""
    voice_emotion = "中性"
    emotion_map = {
        "<|HAPPY|>": "开心",
        "<|SAD|>": "伤心",
        "<|ANGRY|>": "发怒",
        "<|FEARFUL|>": "恐惧",
        "<|NEUTRAL|>": "中性",
        "<|DISGUSTED|>": "厌恶",
        "<|SURPRISED|>": "惊讶",
        "<|EMO_UNKNOWN|>": "中性",
    }

    if "<|zh|>" in raw_text:
        language = "中文"
    elif "<|en|>" in raw_text:
        language = "英文"
    elif "<|ja|>" in raw_text:
        language = "日文"
    elif "<|ko|>" in raw_text:
        language = "韩文"
    elif "<|yue|>" in raw_text:
        language = "粤语"

    found_emotion = False
    for tag, emo in emotion_map.items():
        if tag in raw_text:
            voice_emotion = emo
            found_emotion = True
            break
    if not found_emotion:
        voice_emotion = "中性"

    tags_to_remove = [
        "<|zh|>",
        "<|en|>",
        "<|ja|>",
        "<|ko|>",
        "<|yue|>",
        "<|HAPPY|>",
        "<|SAD|>",
        "<|ANGRY|>",
        "<|FEARFUL|>",
        "<|NEUTRAL|>",
        "<|DISGUSTED|>",
        "<|SURPRISED|>",
        "<|EMO_UNKNOWN|>",
        "<|Speech|>",
        "<|Laughter|>",
        "<|Applause|>",
        "<|Cough|>",
        "<|Sneeze|>",
        "<|Cry|>",
        "<|Music|>",
        "<|/zh|>",
        "<|/en|>",
        "<|/ja|>",
        "<|/ko|>",
        "<|/yue|>",
        "<|woitn|>",
        "<|withitn|>",
    ]
    text = raw_text
    for tag in tags_to_remove:
        text = text.replace(tag, "")
    return text.strip(), language, voice_emotion
