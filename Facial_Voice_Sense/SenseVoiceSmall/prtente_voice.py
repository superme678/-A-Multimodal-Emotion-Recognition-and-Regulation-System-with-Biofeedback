from funasr import AutoModel
import numpy as np
import sounddevice as sd
import wave  # 内置库
import time

# ========== 配置项（按需调整） ==========
import os
model_path = os.path.abspath(os.path.dirname(__file__))  # 自动获取当前目录作为模型路径
sample_rate = 16000  # 模型固定要求16kHz
chunk_duration = 5   # 每次采集3秒音频（可改，比如2/5秒）
channels = 1  # 单声道（模型要求）
dtype = 'int16'  # 16bit格式（和之前的标准音频一致）
# =======================================

# 1. 加载模型（复用已验证的极简配置）
model = AutoModel(
    model=model_path,
    vad_model=None,
    punc_model=None,
    device="cpu",
    disable_update=True
)


# 2. 音频处理工具函数（和之前一致，确保格式匹配）
def process_audio_chunk(audio_data):
    """将采集的音频转成模型要求的float32格式"""
    # 归一化到[-1, 1]
    audio_float = audio_data.astype(np.float32) / 32767.0
    return audio_float


def extract_info(raw_text):
    """提取识别结果、情感、语言等信息"""
    # 提取文本内容
    parts = raw_text.split('<|')
    clean_parts = []
    for part in parts:
        if '|>' in part:
            clean_part = part.split('|>', 1)[1]
            if clean_part:
                clean_parts.append(clean_part)
        else:
            if part:
                clean_parts.append(part)
    clean_text = ''.join(clean_parts).strip()
    
    # 提取语言
    language = ""
    if '<|zh|>' in raw_text:
        language = '中文'
    elif '<|en|>' in raw_text:
        language = '英文'
    elif '<|ja|>' in raw_text:
        language = '日文'
    elif '<|ko|>' in raw_text:
        language = '韩文'
    elif '<|yue|>' in raw_text:
        language = '粤语'
    
    # 提取情感标签
    emotion = ""
    emotion_map = {
        'HAPPY': '开心', 'SAD': '悲伤', 'ANGRY': '愤怒',
        'FEARFUL': '害怕', 'NEUTRAL': '中性', 'DISGUSTED': '厌恶',
        'SURPRISED': '惊讶'
    }
    for emo_en, emo_cn in emotion_map.items():
        if f'<|{emo_en}|>' in raw_text:
            emotion = emo_cn
            break
    
    return clean_text, language, emotion


# 3. 实时采集+识别主逻辑
print("=" * 50)
print("实时语音识别已启动（按 Ctrl+C 停止）")
print(f"采集时长：{chunk_duration}秒/段 | 采样率：{sample_rate}Hz")
print("=" * 50)

try:
    while True:
        # 步骤1：采集麦克风音频（3秒）
        print(f"\n[采集音频] 请说话（{chunk_duration}秒）...")
        audio_chunk = sd.rec(
            int(chunk_duration * sample_rate),
            samplerate=sample_rate,
            channels=channels,
            dtype=dtype
        )
        sd.wait()  # 等待采集完成

        # 步骤2：处理音频格式
        audio_float = process_audio_chunk(audio_chunk.flatten())

        # 步骤3：调用模型识别
        print("[识别中] 请稍候...")
        start_time = time.time()
        result = model.generate(
            input=audio_float,
            cache={},
            language="auto",  # 自动识别中/英文
            use_itn=False
        )
        end_time = time.time()

        # 步骤4：输出结果
        clean_text, language, emotion = extract_info(result[0]['text'])
        if clean_text:
            print(f"[识别结果] {clean_text}")
            if language:
                print(f"[语言] {language}")
            if emotion:
                print(f"[情感] {emotion}")
            print(f"[耗时] {end_time - start_time:.2f}秒")
        else:
            print("[识别结果] 未检测到有效语音")

except KeyboardInterrupt:
    print("\n" + "=" * 50)
    print("实时识别已停止")
    print("=" * 50)
except Exception as e:
    print(f"\n[错误] 识别失败：{e}")
