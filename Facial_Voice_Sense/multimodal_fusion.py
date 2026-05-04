"""三模态情绪加权投票融合（供 webtest 与评测共用）。"""
from collections import Counter


def fuse_multi_emotion(face_emo: str, voice_emo: str, physio: dict) -> str:
    hr = physio.get("heart_rate", 0)
    gsr = physio.get("gsr_volt", 0.0)
    spo2 = physio.get("spo2", 0)

    physio_emo = "中性"
    if hr > 100 and gsr > 1.0:
        physio_emo = "发怒"
    elif hr > 90 and gsr > 0.8:
        physio_emo = "恐惧"
    elif 80 <= hr <= 90 and 0.5 < gsr <= 0.8:
        physio_emo = "开心"
    elif hr < 70 and gsr <= 0.5:
        physio_emo = "中性"
    elif 60 <= hr < 75 and gsr <= 0.3:
        physio_emo = "伤心"

    physio_weight = 6
    face_weight = 7
    voice_weight = 7

    if hr <= 0 or gsr <= 0 or spo2 <= 0:
        physio_weight = 4
        face_weight = 12
        voice_weight = 4

    if face_emo in ["未检测到人脸", "初始化中", "摄像头异常"]:
        face_weight = 0
        total = physio_weight + voice_weight
        if total > 0:
            physio_weight = int(physio_weight / total * 18)
            voice_weight = 2

    if voice_emo == "NEUTRAL" or voice_emo == "中性":
        voice_weight = 1

    physio_weight = max(physio_weight, 1)
    face_weight = max(face_weight, 0)
    voice_weight = max(voice_weight, 1)

    vote = Counter()
    vote.update([physio_emo] * physio_weight)
    if face_weight > 0:
        vote.update([face_emo] * face_weight)
    vote.update([voice_emo] * voice_weight)

    return vote.most_common(1)[0][0] if vote else "中性"
