import argparse
import ctypes
import os
from pathlib import Path
from collections import deque, Counter
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

# ===================== 生理信号配置（代码2核心功能） =====================
# 请修改为你的DAT文件路径和TXT保存路径
DAT_FILE = r"D:\Facial-and-Speech-Dual-Modal-Emotion-Sensing-and-Regulation-System-master\Facial_Voice_Sense\sscom\ReceivedTofile-COM5-2026_4_19_21-07-53.DAT"
SAVE_TXT = r"D:\Facial-and-Speech-Dual-Modal-Emotion-Sensing-and-Regulation-System-master\Facial_Voice_Sense\sscom\手环实时解析结果.txt"
FIXED_FRAME_LEN = 19
FRAME_HEAD = 0xFA
FRAME_TAIL = 0xAF
# ========================================================================

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

def clean_asr_text(raw_text: str) -> tuple[str, str, str]:
    language = ""
    voice_emotion = "中性"  # 默认中性
    # 新增：FunASR情绪标签映射（包含EMO_UNKNOWN）
    emotion_map = {
        "<|HAPPY|>": "开心", "<|SAD|>": "伤心", "<|ANGRY|>": "发怒",
        "<|FEARFUL|>": "恐惧", "<|NEUTRAL|>": "中性", "<|DISGUSTED|>": "厌恶",
        "<|SURPRISED|>": "惊讶", "<|EMO_UNKNOWN|>": "中性"  # 新增：未知情绪统一归为中性
    }

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

    # 提取语音情绪（优先匹配标签）
    found_emotion = False
    for tag, emo in emotion_map.items():
        if tag in raw_text:
            voice_emotion = emo
            found_emotion = True
            break
    # 若未识别到任何情绪标签，默认中性
    if not found_emotion:
        voice_emotion = "中性"

    # 移除所有标签
    tags_to_remove = [
        "<|zh|>", "<|en|>", "<|ja|>", "<|ko|>", "<|yue|>",
        "<|HAPPY|>", "<|SAD|>", "<|ANGRY|>", "<|FEARFUL|>", "<|NEUTRAL|>",
        "<|DISGUSTED|>", "<|SURPRISED|>", "<|EMO_UNKNOWN|>",  # 新增：移除EMO_UNKNOWN
        "<|Speech|>", "<|Laughter|>", "<|Applause|>", "<|Cough|>",
        "<|Sneeze|>", "<|Cry|>", "<|Music|>",
        "<|/zh|>", "<|/en|>", "<|/ja|>", "<|/ko|>", "<|/yue|>",
        "<|woitn|>", "<|withitn|>",
    ]
    text = raw_text
    for tag in tags_to_remove:
        text = text.replace(tag, "")
    return text.strip(), language, voice_emotion

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

# ===================== 代码2：生理数据解析函数 =====================
def parse_valid_frames(buffer):
    """解析完整数据帧，返回结果+剩余残帧缓存"""
    valid_frames = []
    i = 0
    total_len = len(buffer)
    while i <= total_len - FIXED_FRAME_LEN:
        if buffer[i] != FRAME_HEAD:
            i += 1
            continue

        frame = buffer[i:i + FIXED_FRAME_LEN]
        if frame[-1] != FRAME_TAIL:
            i += 1
            continue

        # 校验和验证
        data_sum = sum(frame[1:17])
        calc_checksum = data_sum % 256
        real_checksum = frame[17]
        if calc_checksum != real_checksum:
            i += 1
            continue

        # GSR计算
        gsr_raw = frame[1] * 256 + frame[2]
        gsr_voltage = round((gsr_raw / 4095) * 3.3, 4)

        # 加速度计算
        def calc_acc(high, low):
            raw = high * 256 + low
            signed = raw if raw < 32768 else raw - 65536
            return round((signed / 32768) * 2, 4)

        acc_x = calc_acc(frame[3], frame[4])
        acc_y = calc_acc(frame[5], frame[6])
        acc_z = calc_acc(frame[7], frame[8])

        # 陀螺仪计算
        def calc_gyro(high, low):
            raw = high * 256 + low
            signed = raw if raw < 32768 else raw - 65536
            return round((signed / 32768) * 250, 4)

        gyro_x = calc_gyro(frame[9], frame[10])
        gyro_y = calc_gyro(frame[11], frame[12])
        gyro_z = calc_gyro(frame[13], frame[14])

        # 心率、血氧
        heart_rate = frame[15]
        spo2 = frame[16]

        valid_frames.append([gsr_raw, gsr_voltage, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, heart_rate, spo2])
        i += FIXED_FRAME_LEN

    return valid_frames, buffer[i:]
# =================================================================

# ===================== 新增：生理信号线程 =====================
def physio_loop(shared_state: dict, state_lock: threading.Lock, stop_event: threading.Event):
    cache_buffer = b''
    # 初始化TXT文件
    try:
        header = "=" * 120 + "\n序号 | GSR(原始值/电压) | 加速度X(g) | 加速度Y(g) | 加速度Z(g) | 陀螺仪X(°/s) | 陀螺仪Y(°/s) | 陀螺仪Z(°/s) | 心率 | 血氧\n" + "=" * 120
        with open(SAVE_TXT, "w", encoding="utf-8") as f:
            f.write(header + "\n")
    except:
        pass

    packet_index = 0
    last_physio = {
        "gsr_volt": 0.0, "heart_rate": 0, "spo2": 0,
        "acc_x":0.0, "acc_y":0.0, "acc_z":0.0,
        "gyro_x":0.0, "gyro_y":0.0, "gyro_z":0.0
    }

    while not stop_event.is_set():
        try:
            # 读取DAT文件
            if os.path.exists(DAT_FILE):
                with open(DAT_FILE, 'rb') as f:
                    new_data = f.read()

                if new_data:
                    cache_buffer += new_data
                    valid_frames, cache_buffer = parse_valid_frames(cache_buffer)

                    # 更新最新生理数据
                    for data in valid_frames:
                        packet_index += 1
                        gsr_raw, gsr_volt, acc_x, acc_y, acc_z, gx, gy, gz, hr, spo2 = data
                        last_physio = {
                            "gsr_volt": gsr_volt, "heart_rate": hr, "spo2": spo2,
                            "acc_x": acc_x, "acc_y": acc_y, "acc_z": acc_z,
                            "gyro_x": gx, "gyro_y": gy, "gyro_z": gz
                        }

                        # 写入TXT
                        line = f"{packet_index:>2} | {gsr_raw}({gsr_volt}V) | {acc_x:>9} | {acc_y:>9} | {acc_z:>9} | {gx:>10} | {gy:>10} | {gz:>10} | {hr:>2}bpm | {spo2:>2}%"
                        with open(SAVE_TXT, "a", encoding="utf-8") as f:
                            f.write(line + "\n")

                    # 清空DAT文件
                    with open(DAT_FILE, 'wb') as f:
                        f.truncate(0)
                        f.seek(0)

            # 线程安全更新共享状态
            with state_lock:
                shared_state["physio_data"] = last_physio
                shared_state["physio_status"] = "ok"
                shared_state["physio_ts"] = time.time()

        except Exception as e:
            with state_lock:
                shared_state["physio_status"] = f"error: {str(e)}"

        time.sleep(0.01)

    # 程序退出清空文件
    try:
        with open(SAVE_TXT, 'wb') as f:
            f.truncate(0)
        with open(DAT_FILE, 'wb') as f:
            f.truncate(0)
    except:
        pass
# =================================================================

# ===================== 新增：多模态情绪融合核心算法（动态权重版） =====================
def fuse_multi_emotion(face_emo: str, voice_emo: str, physio: dict) -> str:
    """
    三重情绪融合：动态权重调整（学术标准）
    基准：生理65% + 人脸25% + 语音10%
    自动适配：生理异常 / 无人脸 / 语音无效
    """
    hr = physio.get("heart_rate", 0)
    gsr = physio.get("gsr_volt", 0.0)
    spo2 = physio.get("spo2", 0)

    # 1. 生理信号判断情绪（核心依据）
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

    # ===================== 动态权重核心代码 =====================
    # 基准票数（总20票 = 100%，13=65%，5=25%，2=10%）
    physio_weight = 13
    face_weight = 5
    voice_weight = 2

    # 条件1：生理数据异常（手环未连接/数据无效）→ 降低生理权重
    if hr <= 0 or gsr <= 0 or spo2 <= 0:
        physio_weight = 4
        face_weight = 12
        voice_weight = 4

    # 条件2：未检测到人脸 / 初始化 → 清零人脸权重
    if face_emo in ["未检测到人脸", "初始化中", "摄像头异常"]:
        face_weight = 0
        # 权重重新分配
        total = physio_weight + voice_weight
        if total > 0:
            physio_weight = int(physio_weight / total * 18)
            voice_weight = 2

    # 条件3：语音为默认中性 → 降低语音权重
    if voice_emo == "NEUTRAL" or voice_emo == "中性":
        voice_weight = 1

    # 权重保底，避免全0
    physio_weight = max(physio_weight, 1)
    face_weight = max(face_weight, 0)
    voice_weight = max(voice_weight, 1)
    # ============================================================

    # 加权投票
    vote = Counter()
    vote.update([physio_emo] * physio_weight)
    if face_weight > 0:
        vote.update([face_emo] * face_weight)
    vote.update([voice_emo] * voice_weight)

    # 返回最终情绪
    return vote.most_common(1)[0][0] if vote else "中性"
# =================================================================

def main():
    setup_console_encoding()
    parser = argparse.ArgumentParser(description="面部情绪 + 语音 + 生理信号 三模态情绪识别")
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

    # ===================== 共享状态：新增生理数据 =====================
    shared_state = {
        "face_emotion": "初始化中", "face_status": "init", "face_ts": 0.0,
        "physio_data": {}, "physio_status": "init", "physio_ts": 0.0,
        "voice_emotion": "中性"
    }
    state_lock = threading.Lock()
    stop_event = threading.Event()
    # =================================================================

    # 启动人脸线程
    face_thread = threading.Thread(
        target=face_loop,
        args=(shared_state, state_lock, stop_event, args.cam_device, args.show_face_window),
        daemon=True,
    )
    face_thread.start()
    print("[INFO] 人脸线程已启动")

    # ===================== 启动生理信号线程 =====================
    physio_thread = threading.Thread(
        target=physio_loop,
        args=(shared_state, state_lock, stop_event),
        daemon=True,
    )
    physio_thread.start()
    print("[INFO] 生理信号线程已启动")
    # ==============================================================

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

    # ===================== 修改：获取三模态融合情绪 =====================
    def get_multi_modal_emotion():
        with state_lock:
            # 读取所有状态
            face_status = shared_state["face_status"]
            face_emo = shared_state["face_emotion"]
            voice_emo = shared_state["voice_emotion"]
            physio_data = shared_state.get("physio_data", {})
            physio_status = shared_state.get("physio_status", "init")

        # 异常处理
        if face_status == "camera_error":
            return "摄像头异常"
        if not physio_data or "error" in physio_status:
            return face_emo

        # 融合情绪
        final_emo = fuse_multi_emotion(face_emo, voice_emo, physio_data)
        return final_emo
    # =================================================================

    # ===================== 修改：打印所有数据 =====================
    def print_asr(prefix: str, text: str, language: str, elapsed: float):
        final_emotion = get_multi_modal_emotion()
        with state_lock:
            physio = shared_state.get("physio_data", {})
            face_emo = shared_state["face_emotion"]
            voice_emo = shared_state["voice_emotion"]

        print(f"\n{prefix} {text if text else '未识别到有效语音'}")
        if language:
            print(f"🌐 语言：{language}")
        print(f"🎭 人脸情绪：{face_emo}")
        print(f"🗣️ 语音情绪：{voice_emo}")
        print(f"❤️ 生理数据：心率={physio.get('heart_rate',0)}bpm 血氧={physio.get('spo2',0)}% GSR={physio.get('gsr_volt',0)}V")
        print(f"🔥 最终融合情绪：{final_emotion}")
        print(f"⏱️ 推理耗时：{elapsed:.3f}s")
    # ==============================================================

    def do_asr(samples: np.ndarray):
        begin = time.time()
        out = asr_model.generate(input=samples, cache={}, language="auto", use_itn=False)
        elapsed = time.time() - begin
        text, language, voice_emotion = clean_asr_text(out[0].get("text", ""))
        # 更新语音情绪到共享状态
        with state_lock:
            shared_state["voice_emotion"] = voice_emotion
        return text, language, elapsed

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"[WARN] 麦克风状态异常: {status}")
        audio_queue.put(np.squeeze(indata).astype(np.float32).copy())

    print("[INFO] 三模态情绪识别已启动，按 Ctrl+C 结束")
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
        physio_thread.join(timeout=1.5)
        print(f"\n[INFO] 三模态识别已停止 (reason={stop_reason})")


if __name__ == "__main__":
    main()