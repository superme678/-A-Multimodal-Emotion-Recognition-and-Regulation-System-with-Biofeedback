"""语音情绪/语言标签解析（不加载 FunASR 模型）。"""
from __future__ import annotations

from typing import Any

from evaluation.asr_clean import clean_asr_text


def run_asr_tag_benchmark() -> dict[str, Any]:
    samples: list[tuple[str, str, str, str]] = [
        ("<|zh|><|HAPPY|>你好世界", "你好世界", "中文", "开心"),
        ("<|en|><|SAD|>hello", "hello", "英文", "伤心"),
        ("<|yue|><|ANGRY|>测试<|Speech|>", "测试", "粤语", "发怒"),
        ("plain no tags", "plain no tags", "", "中性"),
        ("<|NEUTRAL|><|zh|>只有中性", "只有中性", "中文", "中性"),
    ]
    ok = 0
    rows = []
    for raw, exp_text, exp_lang, exp_emo in samples:
        t, lang, emo = clean_asr_text(raw)
        row_ok = t == exp_text and lang == exp_lang and emo == exp_emo
        ok += int(row_ok)
        rows.append({"raw": raw[:60], "text": t, "lang": lang, "emo": emo, "pass": row_ok})

    return {
        "modality": "asr_tag_parse",
        "sample_count": len(samples),
        "pass_count": ok,
        "pass_rate": ok / len(samples) if samples else 0.0,
        "samples": rows,
    }
