"""硬件串口协议解析：金样回环、粘包、随机噪声误检率。"""
from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any

# 保证以仓库内 Facial_Voice_Sense 为根可导入 sscom
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluation.physio_golden import build_physio_frame
from sscom.sscom import parse_valid_frames


def _parse_stream(buf: bytes) -> tuple[list, bytes]:
    return parse_valid_frames(buf)


def run_physio_benchmark(
    golden_count: int = 200,
    random_trials: int = 5,
    random_bytes_each: int = 20000,
    seed: int = 42,
) -> dict[str, Any]:
    rng = random.Random(seed)
    golden_specs: list[tuple[int, int, int]] = []
    for i in range(golden_count):
        gsr = (i * 17 + 11) % 4000 + 100
        hr = 60 + (i % 50)
        spo2 = 90 + (i % 10)
        golden_specs.append((gsr, hr, spo2))

    stream = bytearray()
    for gsr, hr, spo2 in golden_specs:
        stream.extend(build_physio_frame(gsr, hr, spo2))

    valid, rest = _parse_stream(bytes(stream))
    decoded = []
    for row in valid:
        gsr_raw, gsr_volt, *_mid, hr, spo2 = row
        decoded.append((int(gsr_raw), int(hr), int(spo2)))

    expected = [(g, h, s) for g, h, s in golden_specs]

    golden_ok = decoded == expected
    golden_match_rate = sum(
            1 for a, b in zip(decoded, expected) if a == b
        ) / max(len(expected), 1)

    # 粘包：帧间插入随机噪声字节后再解析，应仍能提取全部金样帧
    noisy = bytearray()
    for gsr, hr, spo2 in golden_specs[:50]:
        noisy.extend(rng.randbytes(7))
        noisy.extend(build_physio_frame(gsr, hr, spo2))
    noisy.extend(rng.randbytes(13))
    v2, _ = _parse_stream(bytes(noisy))
    sticky_ok = len(v2) == 50

    false_valid = 0
    for _ in range(random_trials):
        blob = rng.randbytes(random_bytes_each)
        vnoise, _ = _parse_stream(blob)
        false_valid += len(vnoise)

    out: dict[str, Any] = {
        "modality": "physio_serial",
        "protocol": "19-byte FA..AF checksum sum(frame[1:17])%256",
        "golden_frame_count": golden_count,
        "golden_stream_decode_exact_match": golden_ok,
        "golden_per_frame_match_rate": golden_match_rate,
        "sticky_packet_frames_expected": 50,
        "sticky_packet_frames_parsed": len(v2) if sticky_ok else len(v2),
        "sticky_packet_full_recovery": sticky_ok,
        "random_trials": random_trials,
        "random_bytes_per_trial": random_bytes_each,
        "random_noise_total_spurious_frames": false_valid,
        "random_noise_spurious_per_kb": false_valid / (random_trials * random_bytes_each / 1024),
        "prng_seed": seed,
    }
    return out
