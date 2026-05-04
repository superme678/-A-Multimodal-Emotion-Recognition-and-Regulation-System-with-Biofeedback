"""
能量门限 VAD 与 webtest.test_multimodal_realtime 中默认参数一致性的数值对照。
不访问麦克风，仅对帧能量序列做判定统计。
"""
from __future__ import annotations

from typing import Any


def vad_decide(
    energies: list[float],
    *,
    vad_threshold: float = 2.2,
    min_energy: float = 0.003,
    noise_floor: float = 0.01,
) -> list[bool]:
    """与实时循环中 `threshold = max(min_energy, noise_floor * vad_threshold)` 一致。"""
    thr = max(min_energy, noise_floor * vad_threshold)
    return [e > thr for e in energies]


def run_vad_micro_benchmark() -> dict[str, Any]:
    noise_floor = 0.01
    # 明显低于门限 / 明显高于门限
    low = [0.001 * i for i in range(1, 20)]
    high = [0.05 + 0.001 * i for i in range(20)]
    thr = max(0.003, noise_floor * 2.2)
    low_flags = vad_decide(low, noise_floor=noise_floor)
    high_flags = vad_decide(high, noise_floor=noise_floor)
    return {
        "modality": "vad_energy_gate",
        "reference_params": {"vad_threshold": 2.2, "min_energy": 0.003, "noise_floor": noise_floor},
        "effective_threshold": thr,
        "low_energy_frames": len(low),
        "low_energy_speech_flags": sum(low_flags),
        "high_energy_frames": len(high),
        "high_energy_speech_flags": sum(high_flags),
        "expected_low_speech_flags": 0,
        "expected_high_speech_flags": len(high),
        "pass_low_silent": sum(low_flags) == 0,
        "pass_high_active": sum(high_flags) == len(high),
    }
