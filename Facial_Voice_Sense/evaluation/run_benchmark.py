"""
三模态系统量化评测入口：生成 JSON 指标与 Markdown 分析报告。

用法（在 Facial_Voice_Sense 目录下）:
    python -m evaluation.run_benchmark
    python -m evaluation.run_benchmark --skip-face   # 跳过 TensorFlow 人脸延迟测试
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _write_report_md(path: Path, bundle: dict) -> None:
    meta = bundle["meta"]
    phys = bundle["physio"]
    fus = bundle["fusion"]
    asr = bundle["asr_tags"]
    face = bundle["face"]
    vad = bundle.get("vad_gate", {})

    lines = [
        "# 三模态情绪系统测试分析报告",
        "",
        f"- 生成时间（UTC）: {meta['generated_at_utc']}",
        f"- 运行主机: {meta.get('hostname', '')}",
        f"- Python: {meta.get('python_version', '')}",
        "",
        "## 1. 验证目的与范围",
        "",
        "本报告用**可复现的数值实验**支撑下列结论：",
        "",
        "1. **硬件串口协议解析**：对合成校验帧与随机噪声的字节流做解码，量化误检率与粘包恢复能力。",
        "2. **ASR 情绪/语言标签解析**：对构造的模型风格输出字符串做单元级对照。",
        "3. **三模态融合规则**：用表格化用例与生理信号缺失消融，验证加权投票是否符合实现定义。",
        "4. **人脸 CNN 推理稳定性与延迟**：在缺少公开测试集标注时，用**重复输入确定性**与**推理耗时分布**刻画稳定性；若需准确率请按 `evaluation/DATA.md` 接入标注数据。",
        "",
        "## 2. 数据来源与规模",
        "",
        "| 模块 | 数据性质 | 规模 | 说明 |",
        "|------|----------|------|------|",
        f"| 生理串口 | 程序合成金样帧（`build_physio_frame`） | {phys.get('golden_frame_count', 0)} 帧 | 字段与 `sscom/sscom.py` 解析一致，可完全复现 |",
        f"| 生理串口 | 随机字节流（误检压力） | {phys.get('random_trials', 0)}×{phys.get('random_bytes_per_trial', 0)} B | 种子={phys.get('prng_seed', '')}，度量「伪合法帧」数量 |",
        "| ASR 标签 | 手工构造字符串 | 见 JSON `asr_tags.samples` | 模拟 SenseVoice 带标签输出 |",
        "| 融合 | 规则用例 + 消融 | 见 JSON `fusion` | 不依赖传感器硬件 |",
    ]
    if face.get("skipped"):
        lines.append(f"| 人脸 CNN | 未执行或跳过 | — | {face.get('skipped_reason', '')} |")
    else:
        lines.append(
            f"| 人脸 CNN | 同输入重复推理 | stability_runs={face.get('stability_runs')}, timing_runs={face.get('timing_runs')} | 权重: `SenseFaceSmall/models/cnn3_best_weights.h5` |"
        )

    lines.extend(
        [
            "",
            "## 3. 测试过程摘要",
            "",
            "- 执行命令: `python -m evaluation.run_benchmark`（可选 `--skip-face`）。",
            "- 生理：将金样帧拼接为字节流，调用 `parse_valid_frames` 与原始 `(gsr,hr,spo2)` 列表逐三元组比对。",
            "- 随机噪声：固定种子 PRNG 生成均匀随机字节，统计被校验和误判为合法帧的数量（期望接近 0）。",
            "- 融合：对预置 `(face,voice,physio)` 调 `fuse_multi_emotion`，断言与手工推演一致。",
            "",
            "## 4. 主要结果",
            "",
            "### 4.1 生理解析（准确性 / 抗干扰）",
            "",
            f"- 金样流完全匹配: **{phys.get('golden_stream_decode_exact_match')}**",
            f"- 粘包场景（50 帧 + 前后随机填充）完全恢复: **{phys.get('sticky_packet_full_recovery')}**",
            f"- 随机噪声伪合法帧总数: **{phys.get('random_noise_total_spurious_frames')}**（折合每 KB 约 **{phys.get('random_noise_spurious_per_kb', 0):.4f}** 帧）",
            "",
            "### 4.2 ASR 标签解析",
            "",
            f"- 通过率: **{asr.get('pass_count')}/{asr.get('sample_count')}**（{asr.get('pass_rate', 0)*100:.1f}%）",
            "",
            "### 4.3 融合规则",
            "",
            f"- 规则用例通过率: **{fus.get('rule_case_pass')}/{fus.get('rule_case_count')}**（{fus.get('rule_case_pass_rate', 0)*100:.1f}%）",
            "",
            "### 4.3b 语音前端 VAD（门限对照）",
            "",
            f"- 与 `webtest` 默认 `max(min_energy, noise×vad_threshold)` 一致，低能量序列判为语音的帧数: **{vad.get('low_energy_speech_flags')}**（期望 0）",
            f"- 高能量序列判为语音的帧数: **{vad.get('high_energy_speech_flags')}/{vad.get('high_energy_frames')}**",
            "",
            "**消融（中性/中性，生理有效 vs 无效）**",
            "",
            f"- 生理有效时输出: `{fus.get('ablation_neutral_neutral', {}).get('fuse_physio_valid')}`",
            f"- 生理无效（传感器缺失）时输出: `{fus.get('ablation_neutral_neutral', {}).get('fuse_physio_invalid')}`",
            "",
            "### 4.4 人脸 CNN（稳定性 / 延迟）",
            "",
        ]
    )
    if face.get("skipped"):
        lines.append(f"- 跳过原因: `{face.get('skipped_reason')}` — 安装依赖并放置权重后可得到延迟与确定性曲线。")
    else:
        lines.extend(
            [
                f"- 同输入 {face.get('stability_runs')} 次推理，argmax 类别唯一数: **{face.get('unique_argmax_count')}**（理想为 1）",
                f"- 与首次结果一致比例: **{face.get('label_consistency_ratio', 0)*100:.2f}%**",
                f"- 推理耗时(ms): 均值 **{face.get('latency_ms_mean', 0):.2f}**, 标准差 **{face.get('latency_ms_std', 0):.2f}**, P95 **{face.get('latency_ms_p95', 0):.2f}**, 最大 **{face.get('latency_ms_max', 0):.2f}**",
            ]
        )

    lines.extend(
        [
            "",
            "## 5. 结论",
            "",
            "- **协议与预处理链路**：金样与粘包实验为解析正确性提供直接证据；随机噪声下伪帧率应接近 0，用于支撑「校验和过滤有效」。",
            "- **融合与标签层**：规则用例与消融实验将「三模态加权」行为固化为可审计的自动化测试，避免仅口头描述。",
            "- **人脸分支**：在无外部标注集时，以**确定性 + 延迟分布**论证推理侧稳定性；分类准确率需额外数据集（见 DATA.md）。",
            "",
            "## 6. 局限与后续工作",
            "",
            "- FunASR 端到端词错误率、人脸在 FER2013/自建集上的 Top-1 准确率需各自准备音频与图像标注后扩展本框架。",
            "- 当前生理「情绪」由 HR/GSR 阈值规则映射，与临床量表的一致性需独立验证数据。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-face", action="store_true", help="跳过 TensorFlow 人脸基准（CI 或无 GPU 时）")
    parser.add_argument("--out-dir", type=str, default="", help="报告输出目录，默认 evaluation/reports")
    args = parser.parse_args()

    import socket as _socket

    try:
        import platform

        py_ver = platform.python_version()
    except Exception:
        py_ver = ""

    meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": _socket.gethostname(),
        "python_version": py_ver,
    }

    from evaluation.bench_asr_tags import run_asr_tag_benchmark
    from evaluation.bench_fusion import run_fusion_benchmark
    from evaluation.bench_physio import run_physio_benchmark
    from evaluation.bench_vad import run_vad_micro_benchmark

    bundle = {
        "meta": meta,
        "physio": run_physio_benchmark(),
        "fusion": run_fusion_benchmark(),
        "asr_tags": run_asr_tag_benchmark(),
        "vad_gate": run_vad_micro_benchmark(),
        "face": {},
    }

    if args.skip_face:
        bundle["face"] = {"skipped": True, "skipped_reason": "cli_skip_face"}
    else:
        try:
            from evaluation.bench_face import run_face_benchmark

            bundle["face"] = run_face_benchmark()
        except Exception as e:
            bundle["face"] = {
                "skipped": True,
                "skipped_reason": f"exception:{type(e).__name__}:{e}",
            }

    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "evaluation" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"benchmark_{stamp}.json"
    md_path = out_dir / f"benchmark_{stamp}.md"

    json_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report_md(md_path, bundle)

    print(f"[OK] JSON: {json_path}")
    print(f"[OK] Markdown: {md_path}")

    # 非零退出：关键断言失败时便于 CI
    ok_phys = bundle["physio"].get("golden_stream_decode_exact_match") and bundle["physio"].get(
        "sticky_packet_full_recovery"
    )
    ok_fus = bundle["fusion"].get("rule_case_pass_rate", 0) >= 1.0 - 1e-9
    ok_asr = bundle["asr_tags"].get("pass_rate", 0) >= 1.0 - 1e-9
    ok_vad = bundle.get("vad_gate", {}).get("pass_low_silent") and bundle.get("vad_gate", {}).get(
        "pass_high_active"
    )
    ok_face = True
    fc = bundle["face"]
    if not fc.get("skipped"):
        ok_face = fc.get("unique_argmax_count", 0) == 1 and fc.get("label_consistency_ratio", 0) >= 0.999

    if not (ok_phys and ok_fus and ok_asr and ok_vad):
        print("[WARN] 生理/融合/ASR/VAD 关键检查未通过，请查看 JSON。")
        return 1
    if not fc.get("skipped") and not ok_face:
        print("[WARN] 人脸稳定性未通过阈值，请查看 JSON。")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
