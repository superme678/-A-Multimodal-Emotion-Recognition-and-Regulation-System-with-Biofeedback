"""人脸分支：推理延迟与确定性（不依赖外部标注集时仍可量化稳定性）。"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent


def run_face_benchmark(
    stability_runs: int = 80,
    timing_runs: int = 40,
) -> dict[str, Any]:
    """
    返回 metrics；若缺少权重文件则 skipped=True，报告仍可用于说明「可复现检查项」。
    """
    out: dict[str, Any] = {
        "modality": "face_cnn",
        "weights_path": str(BASE_DIR / "SenseFaceSmall" / "models" / "cnn3_best_weights.h5"),
        "skipped": False,
        "skipped_reason": "",
    }
    weights = BASE_DIR / "SenseFaceSmall" / "models" / "cnn3_best_weights.h5"
    if not weights.is_file():
        out["skipped"] = True
        out["skipped_reason"] = "weights_file_missing"
        return out

    try:
        from SenseFaceSmall.face_emotion_detection import generate_faces, load_model
    except ImportError as e:
        out["skipped"] = True
        out["skipped_reason"] = f"import_error:{e}"
        return out

    model = load_model()
    gray_roi = np.full((72, 72), 128, dtype=np.uint8)

    faces_aug = generate_faces(gray_roi)
    labels: list[int] = []
    for _ in range(stability_runs):
        scores = model.predict(faces_aug, verbose=0)
        label_index = int(np.argmax(np.sum(scores, axis=0).reshape(-1)))
        labels.append(label_index)

    out["stability_runs"] = stability_runs
    out["unique_argmax_count"] = len(set(labels))
    out["label_consistency_ratio"] = labels.count(labels[0]) / len(labels) if labels else 0.0

    lat_ms: list[float] = []
    for _ in range(timing_runs):
        t0 = time.perf_counter()
        model.predict(faces_aug, verbose=0)
        lat_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(lat_ms, dtype=np.float64)
    out["timing_runs"] = timing_runs
    out["latency_ms_mean"] = float(arr.mean())
    out["latency_ms_std"] = float(arr.std())
    out["latency_ms_p95"] = float(np.percentile(arr, 95))
    out["latency_ms_max"] = float(arr.max())
    return out
