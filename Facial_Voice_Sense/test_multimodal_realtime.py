import argparse
import ctypes
import os
from pathlib import Path
from collections import deque
from collections import Counter
from queue import Empty, Queue
import subprocess
import sys
import threading
import time
import traceback

import cv2
from funasr import AutoModel
import numpy as np
import sounddevice as sd
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D, PReLU
from tensorflow.keras.models import Model

from SenseFaceSmall.blazeface import blaze_detect

BASE_DIR = Path(__file__).resolve().parent
SENSE_FACE_DIR = BASE_DIR / "SenseFaceSmall"
SENSE_VOICE_DIR = BASE_DIR / "SenseVoiceSmall"
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


def resolve_voice_model_path(model_path: str) -> str:
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


def clean_asr_text(raw_text: str) -> tuple[str, str]:
    language = ""
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
    return text.strip(), language


def create_face_model():
    input_layer = Input(shape=(48, 48, 1))
    x = Conv2D(32, (1, 1), strides=1, padding="same", activation="relu")(input_layer)
    x = Conv2D(64, (3, 3), strides=1, padding="same")(x)
    x = PReLU()(x)
    x = Conv2D(64, (5, 5), strides=1, padding="same")(x)
    x = PReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    x = Conv2D(64, (3, 3), strides=1, padding="same")(x)
    x = PReLU()(x)
    x = Conv2D(64, (5, 5), strides=1, padding="same")(x)
    x = PReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    x = Flatten()(x)
    x = Dense(2048, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(8, activation="softmax")(x)
    return Model(inputs=input_layer, outputs=x)


def generate_faces(face_img, img_size=48):
    face_img = face_img / 255.0
    face_img = cv2.resize(face_img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    resized_images = [
        face_img,
        face_img[2:45, :],
        face_img[1:47, :],
        cv2.flip(face_img[:, :], 1),
    ]
    for i in range(len(resized_images)):
        resized_images[i] = cv2.resize(resized_images[i], (img_size, img_size))
        resized_images[i] = np.expand_dims(resized_images[i], axis=-1)
    return np.array(resized_images)


def index_to_emotion(index: int):
    emotions = ["发怒", "厌恶", "恐惧", "开心", "伤心", "惊讶", "中性", "蔑视"]
    return emotions[index]


def face_loop(shared_state: dict, state_lock: threading.Lock, stop_event: threading.Event, camera_index: int, show_window: bool):
    model = create_face_model()
    model.load_weights(str(SENSE_FACE_DIR / "models" / "cnn3_best_weights.h5"))

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        with state_lock:
            shared_state["face_status"] = "camera_error"
        return

    window_name = "Face Stream (Q/ESC to close)"
    if show_window:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    last_emotion = "未检测到人脸"
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.02)
            continue

        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = blaze_detect(frame)
        if faces is not None and len(faces) > 0:
            largest = max(faces, key=lambda b: b[2] * b[3])
            x, y, w, h = largest
            x = max(0, x)
            y = max(0, y)
            w = min(w, gray.shape[1] - x)
            h = min(h, gray.shape[0] - y)
            roi = gray[y:y + h, x:x + w]
            if roi.size > 0:
                faces_aug = generate_faces(roi)
                scores = model.predict(faces_aug, verbose=0)
                label_index = int(np.argmax(np.sum(scores, axis=0).reshape(-1)))
                last_emotion = index_to_emotion(label_index)
                if show_window:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (40, 255, 40), 2)
                    cv2.putText(frame, last_emotion, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 255, 40), 2)

        with state_lock:
            shared_state["face_emotion"] = last_emotion
            shared_state["face_status"] = "ok"
            shared_state["face_ts"] = time.time()

        if show_window:
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                stop_event.set()
                break

    cap.release()
    if show_window:
        cv2.destroyAllWindows()


def main():
    setup_console_encoding()
    parser = argparse.ArgumentParser(description="面部情绪 + 流式语音识别 联合测试")
    parser.add_argument("--mic-device", type=int, default=None, help="麦克风设备索引")
    parser.add_argument("--cam-device", type=int, default=0, help="摄像头设备索引")
    parser.add_argument("--show-face-window", action="store_true", help="显示摄像头窗口")
    parser.add_argument("--once", action="store_true", help="输出一句后退出")
    parser.add_argument("--frame-ms", type=int, default=30)
    parser.add_argument("--vad-threshold", type=float, default=2.2)
    parser.add_argument("--min-energy", type=float, default=0.003)
    parser.add_argument("--end-silence-ms", type=int, default=700)
    parser.add_argument("--min-speech-ms", type=int, default=350)
    parser.add_argument("--pre-roll-ms", type=int, default=300)
    parser.add_argument("--partial-ms", type=int, default=800)
    parser.add_argument("--calibrate-sec", type=float, default=1.2)
    args = parser.parse_args()

    print("[INFO] 加载语音模型...")
    asr_model = AutoModel(
        model=resolve_voice_model_path(str(SENSE_VOICE_DIR)),
        vad_model=None,
        punc_model=None,
        device="cpu",
        disable_update=True,
    )
    print("[OK] 语音模型加载成功")

    shared_state = {"face_emotion": "初始化中", "face_status": "init", "face_ts": 0.0}
    state_lock = threading.Lock()
    stop_event = threading.Event()
    face_thread = threading.Thread(
        target=face_loop,
        args=(shared_state, state_lock, stop_event, args.cam_device, args.show_face_window),
        daemon=True,
    )
    face_thread.start()
    print("[INFO] 人脸线程已启动")

    frame_samples = int(SAMPLE_RATE * args.frame_ms / 1000)
    end_silence_frames = max(1, int(args.end_silence_ms / args.frame_ms))
    min_speech_frames = max(1, int(args.min_speech_ms / args.frame_ms))
    pre_roll_frames = max(0, int(args.pre_roll_ms / args.frame_ms))
    partial_interval_frames = max(1, int(args.partial_ms / args.frame_ms))

    audio_queue: Queue[np.ndarray] = Queue()
    pre_roll = deque(maxlen=pre_roll_frames)
    pending = np.array([], dtype=np.float32)
    utterance_frames: list[np.ndarray] = []
    in_speech = False
    silence_count = 0
    speech_count = 0
    frame_since_partial = 0
    noise_energy = args.min_energy
    calibrated = False
    calibrate_energies: list[float] = []
    face_history = deque()
    last_partial_text = ""
    stop_reason = "unknown"

    def get_face_emotion():
        with state_lock:
            status = shared_state["face_status"]
            emotion = shared_state["face_emotion"]
            ts = shared_state["face_ts"]
        if status == "camera_error":
            return "摄像头异常"

        now = time.time()
        if emotion and status == "ok" and ts > 0:
            face_history.append((ts, emotion))

        # Keep only the last 1 second.
        while face_history and now - face_history[0][0] > 1.0:
            face_history.popleft()

        if not face_history:
            return emotion

        recent_emotions = [item[1] for item in face_history]
        mode_emotion = Counter(recent_emotions).most_common(1)[0][0]
        return mode_emotion

    def print_asr(prefix: str, text: str, language: str, elapsed: float):
        face_emotion = get_face_emotion()
        if text:
            print(f"{prefix} {text}")
        else:
            print(f"{prefix} 未识别到有效语音")
        if language:
            print(f"[LANG] {language}")
        print(f"[FACE] {face_emotion}")
        print(f"[TIME] 推理耗时 {elapsed:.3f}s")

    def do_asr(samples: np.ndarray):
        begin = time.time()
        out = asr_model.generate(input=samples, cache={}, language="auto", use_itn=False)
        elapsed = time.time() - begin
        text, language = clean_asr_text(out[0].get("text", ""))
        return text, language, elapsed

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"[WARN] 麦克风状态异常: {status}")
        audio_queue.put(np.squeeze(indata).astype(np.float32).copy())

    print("[INFO] 联合识别已启动，按 Ctrl+C 结束")
    print("[INFO] 等待语音输入...")
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=args.mic_device,
            blocksize=frame_samples,
            callback=audio_callback,
        ):
            while not stop_event.is_set():
                try:
                    chunk = audio_queue.get(timeout=0.2)
                except Empty:
                    continue

                pending = np.concatenate((pending, chunk)) if pending.size else chunk
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
                            print(f"[VAD] 校准完成 baseline={noise_energy:.5f}")
                        pre_roll.append(frame)
                        continue

                    if not in_speech:
                        noise_energy = 0.95 * noise_energy + 0.05 * energy
                    threshold = max(args.min_energy, noise_energy * args.vad_threshold)
                    is_speech = energy > threshold

                    if is_speech:
                        if not in_speech:
                            in_speech = True
                            silence_count = 0
                            speech_count = 0
                            frame_since_partial = 0
                            utterance_frames = list(pre_roll)
                            print("\n[VAD] 语音开始")
                        utterance_frames.append(frame)
                        speech_count += 1
                        frame_since_partial += 1
                        silence_count = 0

                        if speech_count >= min_speech_frames and frame_since_partial >= partial_interval_frames:
                            frame_since_partial = 0
                            samples = np.concatenate(utterance_frames).astype(np.float32)
                            text, language, elapsed = do_asr(samples)
                            if text and text != last_partial_text:
                                last_partial_text = text
                                print_asr("[PARTIAL]", text, language, elapsed)
                    else:
                        pre_roll.append(frame)
                        if in_speech:
                            utterance_frames.append(frame)
                            silence_count += 1
                            if silence_count >= end_silence_frames:
                                if speech_count >= min_speech_frames:
                                    print("[VAD] 语音结束")
                                    samples = np.concatenate(utterance_frames).astype(np.float32)
                                    text, language, elapsed = do_asr(samples)
                                    print_asr("[FINAL]", text, language, elapsed)
                                    if args.once:
                                        stop_reason = "once_mode_completed"
                                        stop_event.set()
                                        break
                                in_speech = False
                                silence_count = 0
                                speech_count = 0
                                frame_since_partial = 0
                                utterance_frames = []
                                last_partial_text = ""
    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
    except Exception as e:
        stop_reason = f"runtime_exception: {e}"
        print(f"[ERROR] 运行异常: {e}")
        traceback.print_exc()
    finally:
        if stop_reason == "unknown" and stop_event.is_set():
            stop_reason = "stop_event_set"
        stop_event.set()
        face_thread.join(timeout=1.5)
        print(f"\n[INFO] 联合识别已停止 (reason={stop_reason})")


if __name__ == "__main__":
    main()
