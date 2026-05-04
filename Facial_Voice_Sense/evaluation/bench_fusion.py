"""三模态融合：规则用例 + 模态缺失消融（量化权重重分配行为）。"""
from __future__ import annotations

from typing import Any

from multimodal_fusion import fuse_multi_emotion


def run_fusion_benchmark() -> dict[str, Any]:
    cases: list[dict[str, Any]] = [
        {
            "id": "physio_arousal_face_voice_happy",
            "face": "开心",
            "voice": "开心",
            "physio": {"heart_rate": 105, "gsr_volt": 1.2, "spo2": 98},
            "expect": "开心",
            "note": "生理推断发怒(6票)但面+声同为开心(各7票)，开心总票多",
        },
        {
            "id": "face_missing_voice_dominant",
            "face": "未检测到人脸",
            "voice": "恐惧",
            "physio": {"heart_rate": 95, "gsr_volt": 0.85, "spo2": 97},
            "expect": "恐惧",
            "note": "无人脸时降低语音权重，但生理恐惧仍参与",
        },
        {
            "id": "physio_invalid_weights_face",
            "face": "伤心",
            "voice": "开心",
            "physio": {"heart_rate": 0, "gsr_volt": 0.0, "spo2": 0},
            "expect": "伤心",
            "note": "生理无效时提高面部权重，伤心应胜出",
        },
    ]

    results = []
    pass_n = 0
    for c in cases:
        got = fuse_multi_emotion(c["face"], c["voice"], c["physio"])
        ok = got == c["expect"]
        pass_n += int(ok)
        results.append({"case_id": c["id"], "expected": c["expect"], "got": got, "pass": ok, "note": c["note"]})

    # 消融：同一输入，生理从「有效恐惧」变为「无效」
    physio_on = {"heart_rate": 95, "gsr_volt": 0.85, "spo2": 97}
    physio_off = {"heart_rate": 0, "gsr_volt": 0.0, "spo2": 0}
    ablation = {
        "face": "中性",
        "voice": "中性",
        "fuse_physio_valid": fuse_multi_emotion("中性", "中性", physio_on),
        "fuse_physio_invalid": fuse_multi_emotion("中性", "中性", physio_off),
    }

    return {
        "modality": "fusion_vote",
        "rule_case_count": len(cases),
        "rule_case_pass": pass_n,
        "rule_case_pass_rate": pass_n / len(cases) if cases else 0.0,
        "cases": results,
        "ablation_neutral_neutral": ablation,
    }
