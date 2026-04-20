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
import json
import base64
from datetime import datetime

import cv2
from funasr import AutoModel
import numpy as np
import sounddevice as sd
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D, PReLU
from tensorflow.keras.models import Model
from flask import Flask, request, Response
from flask_cors import CORS

# 尝试导入blaze_detect，如果失败则使用备用方案
try:
    from SenseFaceSmall.blazeface import blaze_detect
except ImportError:
    print("[WARN] SenseFaceSmall 未找到，使用OpenCV级联分类器备用")
    blaze_detect = None

# Flask App
app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

BASE_DIR = Path(__file__).resolve().parent
SENSE_FACE_DIR = BASE_DIR / "SenseFaceSmall"
SENSE_VOICE_DIR = BASE_DIR / "SenseVoiceSmall"
SAMPLE_RATE = 16000

# ===================== 生理信号配置 =====================
DAT_FILE = r"D:\AI_project\Facial-and-Speech-Dual-Modal-Emotion-Sensing-and-Regulation-System-master\Facial_Voice_Sense\sscom\ReceivedTofile-COM3-2026_4_20_22-19-30.DAT"
SAVE_TXT = r"D:\AI_project\Facial-and-Speech-Dual-Modal-Emotion-Sensing-and-Regulation-System-master\Facial_Voice_Sense\sscom\手环实时解析结果.txt"
FIXED_FRAME_LEN = 19
FRAME_HEAD = 0xFA
FRAME_TAIL = 0xAF


# ===================== 全局状态管理 =====================
class SystemState:
    def __init__(self):
        self.lock = threading.Lock()
        self.data = {
            "face_emotion": "初始化中",
            "face_status": "init",
            "face_ts": 0.0,
            "face_frame": None,
            "physio_data": {},
            "physio_status": "init",
            "physio_ts": 0.0,
            "voice_emotion": "中性",
            "voice_text": "",
            "voice_language": "",
            "final_emotion": "中性",
            "system_status": "running",
            "logs": deque(maxlen=100)
        }
        self.stop_event = threading.Event()
        self.asr_model = None

    def update(self, key, value):
        with self.lock:
            self.data[key] = value
            self.data[f"{key}_ts"] = time.time()

    def get_all(self):
        with self.lock:
            # 创建副本并转换deque为list
            result = {}
            for key, value in self.data.items():
                if key == "logs":
                    result[key] = list(value)  # deque转list
                elif isinstance(value, np.integer):
                    result[key] = int(value)
                elif isinstance(value, np.floating):
                    result[key] = float(value)
                elif isinstance(value, np.ndarray):
                    result[key] = value.tolist()
                elif isinstance(value, dict):
                    # 处理嵌套dict中的numpy类型
                    result[key] = self._convert_numpy_types(value)
                else:
                    result[key] = value
            return result

    def _convert_numpy_types(self, obj):
        """递归转换numpy类型为Python原生类型"""
        if isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def add_log(self, message, level="INFO"):
        with self.lock:
            self.data["logs"].append({
                "time": datetime.now().isoformat(),
                "level": level,
                "message": message
            })


system_state = SystemState()


# ===================== 工具函数 =====================
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
                check=True, capture_output=True, text=True,
            )
        return str(junction_path)
    except Exception:
        return to_windows_short_path(model_path)


def clean_asr_text(raw_text: str) -> tuple[str, str, str]:
    language = ""
    voice_emotion = "中性"
    emotion_map = {
        "<|HAPPY|>": "开心", "<|SAD|>": "伤心", "<|ANGRY|>": "发怒",
        "<|FEARFUL|>": "恐惧", "<|NEUTRAL|>": "中性", "<|DISGUSTED|>": "厌恶",
        "<|SURPRISED|>": "惊讶", "<|EMO_UNKNOWN|>": "中性"
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

    found_emotion = False
    for tag, emo in emotion_map.items():
        if tag in raw_text:
            voice_emotion = emo
            found_emotion = True
            break
    if not found_emotion:
        voice_emotion = "中性"

    tags_to_remove = [
        "<|zh|>", "<|en|>", "<|ja|>", "<|ko|>", "<|yue|>",
        "<|HAPPY|>", "<|SAD|>", "<|ANGRY|>", "<|FEARFUL|>", "<|NEUTRAL|>",
        "<|DISGUSTED|>", "<|SURPRISED|>", "<|EMO_UNKNOWN|>",
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


# 备用：使用OpenCV级联分类器检测人脸
def detect_faces_opencv(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return faces


# ===================== 生理数据解析 =====================
def parse_valid_frames(buffer):
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

        data_sum = sum(frame[1:17])
        calc_checksum = data_sum % 256
        real_checksum = frame[17]
        if calc_checksum != real_checksum:
            i += 1
            continue

        gsr_raw = frame[1] * 256 + frame[2]
        gsr_voltage = round((gsr_raw / 4095) * 3.3, 4)

        def calc_acc(high, low):
            raw = high * 256 + low
            signed = raw if raw < 32768 else raw - 65536
            return round((signed / 32768) * 2, 4)

        acc_x = calc_acc(frame[3], frame[4])
        acc_y = calc_acc(frame[5], frame[6])
        acc_z = calc_acc(frame[7], frame[8])

        def calc_gyro(high, low):
            raw = high * 256 + low
            signed = raw if raw < 32768 else raw - 65536
            return round((signed / 32768) * 250, 4)

        gyro_x = calc_gyro(frame[9], frame[10])
        gyro_y = calc_gyro(frame[11], frame[12])
        gyro_z = calc_gyro(frame[13], frame[14])

        heart_rate = frame[15]
        spo2 = frame[16]

        valid_frames.append([gsr_raw, gsr_voltage, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, heart_rate, spo2])
        i += FIXED_FRAME_LEN

    return valid_frames, buffer[i:]


# ===================== 情绪融合算法 =====================
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

    physio_weight = 9
    face_weight = 5
    voice_weight = 6

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


# ===================== 人脸检测线程 =====================
def face_loop(camera_index: int):
    model = create_face_model()
    weights_path = str(SENSE_FACE_DIR / "models" / "cnn3_best_weights.h5")
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
    else:
        system_state.add_log("人脸模型权重文件未找到，使用随机权重", "WARN")

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        system_state.update("face_status", "camera_error")
        system_state.add_log("摄像头打开失败", "ERROR")
        return

    system_state.add_log("人脸检测线程启动成功")
    last_emotion = "未检测到人脸"

    while not system_state.stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.02)
            continue

        frame = cv2.resize(frame, (640, 480))

        # 检测人脸
        if blaze_detect:
            faces = blaze_detect(frame)
        else:
            faces = detect_faces_opencv(frame)

        display_frame = frame.copy()

        if faces is not None and len(faces) > 0:
            if blaze_detect:
                largest = max(faces, key=lambda b: b[2] * b[3])
                x, y, w, h = largest
            else:
                x, y, w, h = faces[0]

            x = max(0, x)
            y = max(0, y)
            w = min(w, frame.shape[1] - x)
            h = min(h, frame.shape[0] - y)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            roi = gray[y:y + h, x:x + w]

            if roi.size > 0:
                try:
                    faces_aug = generate_faces(roi)
                    scores = model.predict(faces_aug, verbose=0)
                    label_index = int(np.argmax(np.sum(scores, axis=0).reshape(-1)))
                    last_emotion = index_to_emotion(label_index)
                except Exception as e:
                    last_emotion = "检测错误"

                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (40, 255, 40), 2)
                cv2.putText(display_frame, last_emotion, (x, max(20, y - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 255, 40), 2)

        # 编码为base64
        _, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        system_state.update("face_emotion", last_emotion)
        system_state.update("face_status", "ok")
        system_state.update("face_frame", f"data:image/jpeg;base64,{img_base64}")

        time.sleep(0.05)

    cap.release()
    system_state.add_log("人脸检测线程已停止")


# ===================== 生理信号线程 =====================
def physio_loop():
    cache_buffer = b''
    try:
        header = "=" * 120 + "\n序号 | GSR(原始值/电压) | 加速度X(g) | 加速度Y(g) | 加速度Z(g) | 陀螺仪X(°/s) | 陀螺仪Y(°/s) | 陀螺仪Z(°/s) | 心率 | 血氧\n" + "=" * 120
        with open(SAVE_TXT, "w", encoding="utf-8") as f:
            f.write(header + "\n")
    except Exception as e:
        system_state.add_log(f"初始化生理数据文件失败: {e}", "WARN")

    packet_index = 0
    last_physio = {
        "gsr_raw": 0, "gsr_volt": 0.0, "heart_rate": 0, "spo2": 0,
        "acc_x": 0.0, "acc_y": 0.0, "acc_z": 0.0,
        "gyro_x": 0.0, "gyro_y": 0.0, "gyro_z": 0.0
    }

    system_state.add_log("生理信号线程启动成功")

    while not system_state.stop_event.is_set():
        try:
            if os.path.exists(DAT_FILE):
                with open(DAT_FILE, 'rb') as f:
                    new_data = f.read()

                if new_data:
                    cache_buffer += new_data
                    valid_frames, cache_buffer = parse_valid_frames(cache_buffer)

                    for data in valid_frames:
                        packet_index += 1
                        gsr_raw, gsr_volt, acc_x, acc_y, acc_z, gx, gy, gz, hr, spo2 = data
                        last_physio = {
                            "gsr_raw": int(gsr_raw),
                            "gsr_volt": float(gsr_volt),
                            "heart_rate": int(hr),
                            "spo2": int(spo2),
                            "acc_x": float(acc_x),
                            "acc_y": float(acc_y),
                            "acc_z": float(acc_z),
                            "gyro_x": float(gx),
                            "gyro_y": float(gy),
                            "gyro_z": float(gz)
                        }

                        line = f"{packet_index:>2} | {gsr_raw}({gsr_volt}V) | {acc_x:>9} | {acc_y:>9} | {acc_z:>9} | {gx:>10} | {gy:>10} | {gz:>10} | {hr:>2}bpm | {spo2:>2}%"
                        try:
                            with open(SAVE_TXT, "a", encoding="utf-8") as f:
                                f.write(line + "\n")
                        except:
                            pass

                    try:
                        with open(DAT_FILE, 'wb') as f:
                            f.truncate(0)
                            f.seek(0)
                    except:
                        pass

            system_state.update("physio_data", last_physio)
            system_state.update("physio_status", "ok")

        except Exception as e:
            system_state.update("physio_status", f"error: {str(e)}")
            system_state.add_log(f"生理信号错误: {str(e)}", "ERROR")

        time.sleep(0.01)

    try:
        with open(SAVE_TXT, 'wb') as f:
            f.truncate(0)
        with open(DAT_FILE, 'wb') as f:
            f.truncate(0)
    except:
        pass
    system_state.add_log("生理信号线程已停止")


# ===================== 语音处理线程 =====================
def voice_loop(mic_device=None):
    frame_ms = 30
    vad_threshold = 2.2
    min_energy = 0.003
    end_silence_ms = 700
    min_speech_ms = 350
    pre_roll_ms = 300
    partial_ms = 800
    calibrate_sec = 1.2

    frame_samples = int(SAMPLE_RATE * frame_ms / 1000)
    end_silence_frames = max(1, int(end_silence_ms / frame_ms))
    min_speech_frames = max(1, int(min_speech_ms / frame_ms))
    pre_roll_frames = max(0, int(pre_roll_ms / frame_ms))
    partial_interval_frames = max(1, int(partial_ms / frame_ms))

    audio_queue: Queue[np.ndarray] = Queue()
    pre_roll = deque(maxlen=pre_roll_frames)
    pending = np.array([], dtype=np.float32)
    utterance_frames: list[np.ndarray] = []
    in_speech = False
    silence_count = 0
    speech_count = 0
    frame_since_partial = 0
    noise_energy = min_energy
    calibrated = False
    calibrate_energies: list[float] = []
    last_partial_text = ""

    def do_asr(samples: np.ndarray):
        begin = time.time()
        try:
            out = system_state.asr_model.generate(input=samples, cache={}, language="auto", use_itn=False)
            elapsed = time.time() - begin
            text, language, voice_emotion = clean_asr_text(out[0].get("text", ""))
            system_state.update("voice_emotion", voice_emotion)
            return text, language, elapsed, voice_emotion
        except Exception as e:
            return "", "", 0, "中性"

    def audio_callback(indata, frames, time_info, status):
        if status:
            system_state.add_log(f"麦克风状态异常: {status}", "WARN")
        audio_queue.put(np.squeeze(indata).astype(np.float32).copy())

    system_state.add_log("语音处理线程启动成功")

    try:
        with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                device=mic_device,
                blocksize=frame_samples,
                callback=audio_callback,
        ):
            while not system_state.stop_event.is_set():
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
                        required = max(1, int(calibrate_sec * 1000 / frame_ms))
                        if len(calibrate_energies) >= required:
                            noise_energy = float(np.percentile(calibrate_energies, 80))
                            calibrated = True
                            system_state.add_log(f"VAD校准完成 baseline={noise_energy:.5f}")
                        pre_roll.append(frame)
                        continue

                    if not in_speech:
                        noise_energy = 0.95 * noise_energy + 0.05 * energy
                    threshold = max(min_energy, noise_energy * vad_threshold)
                    is_speech = energy > threshold

                    if is_speech:
                        if not in_speech:
                            in_speech = True
                            silence_count = 0
                            speech_count = 0
                            frame_since_partial = 0
                            utterance_frames = list(pre_roll)
                            system_state.add_log("语音开始检测")
                        utterance_frames.append(frame)
                        speech_count += 1
                        frame_since_partial += 1
                        silence_count = 0

                        if speech_count >= min_speech_frames and frame_since_partial >= partial_interval_frames:
                            frame_since_partial = 0
                            samples = np.concatenate(utterance_frames).astype(np.float32)
                            text, language, elapsed, voice_emo = do_asr(samples)
                            if text and text != last_partial_text:
                                last_partial_text = text
                                system_state.update("voice_text", text)
                                system_state.update("voice_language", language)
                                system_state.add_log(f"[部分识别] {text} ({voice_emo})")
                    else:
                        pre_roll.append(frame)
                        if in_speech:
                            utterance_frames.append(frame)
                            silence_count += 1
                            if silence_count >= end_silence_frames:
                                if speech_count >= min_speech_frames:
                                    system_state.add_log("语音结束，处理中...")
                                    samples = np.concatenate(utterance_frames).astype(np.float32)
                                    text, language, elapsed, voice_emo = do_asr(samples)
                                    system_state.update("voice_text", text)
                                    system_state.update("voice_language", language)
                                    system_state.add_log(f"[最终识别] {text} ({voice_emo})")

                                    face_emo = system_state.data.get("face_emotion", "中性")
                                    physio = system_state.data.get("physio_data", {})
                                    final_emo = fuse_multi_emotion(face_emo, voice_emo, physio)
                                    system_state.update("final_emotion", final_emo)

                                in_speech = False
                                silence_count = 0
                                speech_count = 0
                                frame_since_partial = 0
                                utterance_frames = []
                                last_partial_text = ""
    except Exception as e:
        system_state.add_log(f"语音处理错误: {str(e)}", "ERROR")
        traceback.print_exc()
    finally:
        system_state.add_log("语音处理线程已停止")


# ===================== Flask API 路由 =====================
@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/api/status')
def api_status():
    try:
        data = system_state.get_all()
        return Response(json.dumps(data, ensure_ascii=False),
                        mimetype='application/json; charset=utf-8')
    except Exception as e:
        error_data = {"error": str(e), "traceback": traceback.format_exc()}
        return Response(json.dumps(error_data, ensure_ascii=False),
                        status=500, mimetype='application/json')


@app.route('/api/logs')
def api_logs():
    try:
        with system_state.lock:
            logs = list(system_state.data["logs"])
        return Response(json.dumps(logs, ensure_ascii=False),
                        mimetype='application/json; charset=utf-8')
    except Exception as e:
        return Response(json.dumps({"error": str(e)}, ensure_ascii=False),
                        status=500, mimetype='application/json')


@app.route('/api/config', methods=['GET', 'POST'])
def api_config():
    if request.method == 'POST':
        config = request.json
        return Response(json.dumps({"success": True, "config": config}, ensure_ascii=False),
                        mimetype='application/json')
    return Response(json.dumps({
        "dat_file": DAT_FILE,
        "save_txt": SAVE_TXT,
        "sample_rate": SAMPLE_RATE
    }, ensure_ascii=False), mimetype='application/json')


@app.route('/api/control', methods=['POST'])
def api_control():
    try:
        action = request.json.get('action')
        if action == 'stop':
            system_state.stop_event.set()
            return Response(json.dumps({"success": True, "message": "系统停止中..."}, ensure_ascii=False),
                            mimetype='application/json')
        elif action == 'restart':
            system_state.stop_event.clear()
            return Response(json.dumps({"success": True, "message": "系统重启中..."}, ensure_ascii=False),
                            mimetype='application/json')
        return Response(json.dumps({"success": False, "message": "未知操作"}, ensure_ascii=False),
                        mimetype='application/json')
    except Exception as e:
        return Response(json.dumps({"error": str(e)}, ensure_ascii=False),
                        status=500, mimetype='application/json')


# ===================== 主函数 =====================
def main():
    setup_console_encoding()

    parser = argparse.ArgumentParser(description="三模态情绪识别 API 服务")
    parser.add_argument("--mic-device", type=int, default=None)
    parser.add_argument("--cam-device", type=int, default=0)
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    # 创建static目录
    static_dir = BASE_DIR / "static"
    static_dir.mkdir(exist_ok=True)

    print("[INFO] 加载语音模型...")
    try:
        system_state.asr_model = AutoModel(
            model=resolve_voice_model_path(str(SENSE_VOICE_DIR)),
            vad_model=None,
            punc_model=None,
            device="cpu",
            disable_update=True,
        )
        print("[OK] 语音模型加载成功")
        system_state.add_log("语音模型加载成功")
    except Exception as e:
        print(f"[ERROR] 语音模型加载失败: {e}")
        system_state.add_log(f"语音模型加载失败: {e}", "ERROR")
        # 继续运行，只是语音功能不可用

    # 启动后台线程
    face_thread = threading.Thread(target=face_loop, args=(args.cam_device,), daemon=True)
    physio_thread = threading.Thread(target=physio_loop, daemon=True)
    voice_thread = threading.Thread(target=voice_loop, args=(args.mic_device,), daemon=True)

    face_thread.start()
    physio_thread.start()
    voice_thread.start()

    time.sleep(1)

    print(f"[INFO] API 服务启动于 http://{args.host}:{args.port}")
    print(f"[INFO] 前端访问: http://localhost:{args.port}")
    system_state.add_log(f"API 服务启动于 http://{args.host}:{args.port}")

    try:
        app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        print("\n[INFO] 正在关闭服务...")
    finally:
        system_state.stop_event.set()
        face_thread.join(timeout=2)
        physio_thread.join(timeout=2)
        voice_thread.join(timeout=2)
        print("[INFO] 服务已停止")


if __name__ == "__main__":
    main()