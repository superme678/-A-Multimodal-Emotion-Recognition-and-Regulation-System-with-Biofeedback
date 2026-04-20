import argparse
import ctypes
import os
from pathlib import Path
import subprocess
import sys
import time
from collections import deque
from queue import Empty, Queue

from funasr import AutoModel
import numpy as np
import sounddevice as sd

BASE_DIR = Path(__file__).resolve().parent
SAMPLE_RATE = 16000


def setup_console_encoding():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass


def to_windows_short_path(path_str: str) -> str:
    if os.name != "nt":
        return path_str
    try:
        get_short = ctypes.windll.kernel32.GetShortPathNameW
        buffer = ctypes.create_unicode_buffer(260)
        result = get_short(path_str, buffer, 260)
        if result > 0:
            return buffer.value
    except Exception:
        pass
    return path_str


def resolve_model_path(model_path: str) -> str:
    if os.name != "nt":
        return model_path
    try:
        model_path.encode("ascii")
        return model_path
    except UnicodeEncodeError:
        pass

    src = Path(model_path)
    junction_root = Path(f"{src.drive}\\FVS_MODEL_ASCII")
    junction_path = junction_root / "SenseVoiceSmall"
    try:
        junction_root.mkdir(parents=True, exist_ok=True)
        if not junction_path.exists():
            subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(junction_path), str(src)],
                check=True,
                capture_output=True,
                text=True,
            )
        return str(junction_path)
    except Exception:
        return to_windows_short_path(model_path)


def clean_text(raw_text: str) -> tuple[str, str, str]:
    language = ""
    emotion = ""

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

    emotion_map = {
        "HAPPY": "开心",
        "SAD": "悲伤",
        "ANGRY": "愤怒",
        "FEARFUL": "害怕",
        "NEUTRAL": "中性",
        "DISGUSTED": "厌恶",
        "SURPRISED": "惊讶",
    }
    for tag, cn_name in emotion_map.items():
        if f"<|{tag}|>" in raw_text:
            emotion = cn_name
            break

    tags_to_remove = [
        "<|zh|>", "<|en|>", "<|ja|>", "<|ko|>", "<|yue|>",
        "<|HAPPY|>", "<|SAD|>", "<|ANGRY|>", "<|FEARFUL|>", "<|NEUTRAL|>",
        "<|DISGUSTED|>", "<|SURPRISED|>",
        "<|Speech|>", "<|Laughter|>", "<|Applause|>", "<|Cough|>",
        "<|Sneeze|>", "<|Cry|>", "<|Music|>",
        "<|/zh|>", "<|/en|>", "<|/ja|>", "<|/ko|>", "<|/yue|>",
        "<|woitn|>", "<|withitn|>",
    ]
    text = raw_text
    for tag in tags_to_remove:
        text = text.replace(tag, "")

    return text.strip(), language, emotion


def run_asr(model: AutoModel, samples: np.ndarray) -> tuple[str, str, str, float]:
    begin = time.time()
    out = model.generate(
        input=samples,
        cache={},
        language="auto",
        use_itn=False,
    )
    elapsed = time.time() - begin
    raw = out[0].get("text", "")
    text, language, emotion = clean_text(raw)
    return text, language, emotion, elapsed


def print_result(prefix: str, text: str, language: str, emotion: str, elapsed: float):
    if text:
        print(f"{prefix} {text}")
        if language:
            print(f"[LANG] {language}")
        if emotion:
            print(f"[EMO] {emotion}")
    else:
        print(f"{prefix} 未识别到有效语音")
    print(f"[TIME] 推理耗时 {elapsed:.3f}s")


def main():
    setup_console_encoding()

    parser = argparse.ArgumentParser(description="SenseVoice 实时语音识别（VAD + 流式）")
    parser.add_argument("--device", type=int, default=None, help="麦克风设备索引，默认系统设备")
    parser.add_argument("--once", action="store_true", help="只输出一句最终结果后退出")
    parser.add_argument("--frame-ms", type=int, default=30, help="VAD 帧长（毫秒），默认 30")
    parser.add_argument("--vad-threshold", type=float, default=2.2, help="能量阈值倍率（越大越严格）")
    parser.add_argument("--min-energy", type=float, default=0.003, help="最小能量阈值，默认 0.003")
    parser.add_argument("--end-silence-ms", type=int, default=700, help="静音多久判定句尾，默认 700ms")
    parser.add_argument("--min-speech-ms", type=int, default=350, help="最短语音长度，默认 350ms")
    parser.add_argument("--pre-roll-ms", type=int, default=300, help="前置保留时长，默认 300ms")
    parser.add_argument("--partial-ms", type=int, default=800, help="流式临时结果间隔，默认 800ms")
    parser.add_argument("--max-utterance-sec", type=float, default=15.0, help="单句最长秒数，默认 15s")
    parser.add_argument("--calibrate-sec", type=float, default=1.2, help="启动静音校准秒数，默认 1.2s")
    parser.add_argument("--debug-vad", action="store_true", help="打印每秒 VAD 能量与阈值")
    args = parser.parse_args()

    model_dir = resolve_model_path(str(BASE_DIR))
    print("[INFO] 正在加载模型...")
    model = AutoModel(
        model=model_dir,
        vad_model=None,
        punc_model=None,
        device="cpu",
        disable_update=True,
    )

    frame_samples = int(SAMPLE_RATE * args.frame_ms / 1000)
    end_silence_frames = max(1, int(args.end_silence_ms / args.frame_ms))
    min_speech_frames = max(1, int(args.min_speech_ms / args.frame_ms))
    pre_roll_frames = max(0, int(args.pre_roll_ms / args.frame_ms))
    partial_interval_frames = max(1, int(args.partial_ms / args.frame_ms))
    max_utterance_samples = int(args.max_utterance_sec * SAMPLE_RATE)

    audio_queue: Queue[np.ndarray] = Queue()
    pre_roll = deque(maxlen=pre_roll_frames)
    pending = np.array([], dtype=np.float32)
    utterance_frames: list[np.ndarray] = []
    in_speech = False
    silence_count = 0
    speech_count = 0
    frame_count_since_partial = 0
    noise_energy = 0.003
    last_partial_text = ""
    debug_frame_counter = 0
    debug_energy_sum = 0.0
    debug_threshold_sum = 0.0
    calibrated = False
    calibrate_energies: list[float] = []

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"[WARN] 录音状态异常: {status}", flush=True)
        mono = np.squeeze(indata).astype(np.float32)
        audio_queue.put(mono.copy())

    def finalize_utterance() -> bool:
        nonlocal utterance_frames, in_speech, silence_count, speech_count
        nonlocal frame_count_since_partial, last_partial_text
        if not utterance_frames:
            return False
        samples = np.concatenate(utterance_frames).astype(np.float32)
        utterance_frames = []
        in_speech = False
        silence_count = 0
        speech_count = 0
        frame_count_since_partial = 0
        last_partial_text = ""
        text, language, emotion, elapsed = run_asr(model, samples)
        print_result("[FINAL]", text, language, emotion, elapsed)
        return bool(text)

    print("[OK] 模型加载成功")
    print(f"[INFO] 采样率={SAMPLE_RATE}Hz, 帧长={args.frame_ms}ms")
    print("[INFO] 已启用 VAD 自动断句 + 流式增量识别")
    print(f"[INFO] 当前阈值参数: vad_threshold={args.vad_threshold}, min_energy={args.min_energy}")
    print(f"[INFO] 请保持 {args.calibrate_sec:.1f}s 安静，进行环境噪声校准")
    print("[INFO] 按 Ctrl+C 停止")

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=args.device,
            blocksize=frame_samples,
            callback=audio_callback,
        ):
            while True:
                try:
                    chunk = audio_queue.get(timeout=0.2)
                except Empty:
                    continue

                if pending.size == 0:
                    pending = chunk
                else:
                    pending = np.concatenate((pending, chunk))

                while pending.size >= frame_samples:
                    frame = pending[:frame_samples]
                    pending = pending[frame_samples:]

                    energy = float(np.mean(np.abs(frame)))
                    if not calibrated:
                        calibrate_energies.append(energy)
                        required = max(1, int(args.calibrate_sec * 1000 / args.frame_ms))
                        if len(calibrate_energies) >= required:
                            noise_energy = float(np.percentile(calibrate_energies, 80))
                            calibrated = True
                            print(f"[VAD] 校准完成: baseline={noise_energy:.5f}")
                        pre_roll.append(frame)
                        continue

                    if not in_speech:
                        noise_energy = 0.95 * noise_energy + 0.05 * energy
                    threshold = max(args.min_energy, noise_energy * args.vad_threshold)
                    is_speech = energy > threshold
                    if args.debug_vad:
                        debug_frame_counter += 1
                        debug_energy_sum += energy
                        debug_threshold_sum += threshold
                        if debug_frame_counter >= max(1, int(1000 / args.frame_ms)):
                            avg_energy = debug_energy_sum / debug_frame_counter
                            avg_threshold = debug_threshold_sum / debug_frame_counter
                            print(
                                f"[VAD-DEBUG] avg_energy={avg_energy:.5f}, avg_threshold={avg_threshold:.5f}, speech={is_speech}"
                            )
                            debug_frame_counter = 0
                            debug_energy_sum = 0.0
                            debug_threshold_sum = 0.0

                    if is_speech:
                        if not in_speech:
                            in_speech = True
                            silence_count = 0
                            speech_count = 0
                            utterance_frames = list(pre_roll)
                            print("\n[VAD] 检测到语音开始")
                        utterance_frames.append(frame)
                        speech_count += 1
                        silence_count = 0
                        frame_count_since_partial += 1

                        current_len = sum(len(f) for f in utterance_frames)
                        should_partial = (
                            speech_count >= min_speech_frames
                            and frame_count_since_partial >= partial_interval_frames
                        )
                        if should_partial:
                            frame_count_since_partial = 0
                            samples = np.concatenate(utterance_frames).astype(np.float32)
                            text, _, _, _ = run_asr(model, samples)
                            if text and text != last_partial_text:
                                last_partial_text = text
                                print(f"[PARTIAL] {text}")

                        if current_len >= max_utterance_samples:
                            print("[VAD] 达到单句最大时长，强制切句")
                            if finalize_utterance() and args.once:
                                return
                    else:
                        pre_roll.append(frame)
                        if in_speech:
                            utterance_frames.append(frame)
                            silence_count += 1
                            if silence_count >= end_silence_frames:
                                if speech_count >= min_speech_frames:
                                    print("[VAD] 检测到语音结束")
                                    if finalize_utterance() and args.once:
                                        return
                                else:
                                    in_speech = False
                                    silence_count = 0
                                    speech_count = 0
                                    utterance_frames = []
                                    frame_count_since_partial = 0
                                    last_partial_text = ""
    except KeyboardInterrupt:
        print("\n[INFO] 已停止实时识别")


if __name__ == "__main__":
    main()
