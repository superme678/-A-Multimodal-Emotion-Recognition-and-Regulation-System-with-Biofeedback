"""
author: Assistant
datetime: 2026/3/26
 desc: 摄像头实时检测人脸情绪
"""
import os
import argparse
import sys
import cv2
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.layers import PReLU
from blazeface import blaze_detect

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=int, default=0, help="data source, 0 for camera 1 for video")
parser.add_argument("--video_path", type=str, default=None, help="video file path when source=1")
opt = parser.parse_args()

WINDOW_NAME = "Face Emotion Detection (ESC/Q to exit)"


def _setup_console_encoding():
    """Avoid garbled Chinese logs on Windows terminals."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

def create_model():
    """
    创建情绪识别模型
    :return:
    """
    # input
    input_layer = Input(shape=(48, 48, 1))
    x = Conv2D(32, (1, 1), strides=1, padding='same', activation='relu')(input_layer)
    # block1
    x = Conv2D(64, (3, 3), strides=1, padding='same')(x)
    x = PReLU()(x)
    x = Conv2D(64, (5, 5), strides=1, padding='same')(x)
    x = PReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    # block2
    x = Conv2D(64, (3, 3), strides=1, padding='same')(x)
    x = PReLU()(x)
    x = Conv2D(64, (5, 5), strides=1, padding='same')(x)
    x = PReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    # fc
    x = Flatten()(x)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(8, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=x)
    return model

def load_model():
    """
    加载本地模型
    :return:
    """
    model = create_model()
    # 构建到模型文件的路径
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'cnn3_best_weights.h5')
    model.load_weights(model_path)
    return model

def generate_faces(face_img, img_size=48):
    """
    将探测到的人脸进行增广
    :param face_img: 灰度化的单个人脸图
    :param img_size: 目标图片大小
    :return:
    """

    face_img = face_img / 255.
    face_img = cv2.resize(face_img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    resized_images = list()
    resized_images.append(face_img)
    resized_images.append(face_img[2:45, :])
    resized_images.append(face_img[1:47, :])
    resized_images.append(cv2.flip(face_img[:, :], 1))

    for i in range(len(resized_images)):
        resized_images[i] = cv2.resize(resized_images[i], (img_size, img_size))
        resized_images[i] = np.expand_dims(resized_images[i], axis=-1)
    resized_images = np.array(resized_images)
    return resized_images

def index2emotion(index=0, kind='cn'):
    """
    根据表情下标返回表情字符串
    :param index:
    :return:
    """
    emotions = {
        '发怒': 'anger',
        '厌恶': 'disgust',
        '恐惧': 'fear',
        '开心': 'happy',
        '伤心': 'sad',
        '惊讶': 'surprised',
        '中性': 'neutral',
        '蔑视': 'contempt'
    }
    if kind == 'cn':
        return list(emotions.keys())[index]
    else:
        return list(emotions.values())[index]

def cv2_img_add_text(img, text, left, top, text_color=(0, 255, 0), text_size=20):
    """
    :param img:
    :param text:
    :param left:
    :param top:
    :param text_color:
    :param text_size
    :return:
    """
    from PIL import Image, ImageDraw, ImageFont

    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    font_text = ImageFont.truetype(
        os.path.join(os.path.dirname(__file__), 'assets', 'simsun.ttc'), text_size, encoding="utf-8")  # 使用宋体
    draw.text((left, top), text, text_color, font=font_text)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def detect_face(img):
    """
    检测人脸
    :param img:
    :return:
    """
    # 使用BlazeFace检测器
    faces = blaze_detect(img)
    return faces

def predict_expression():
    """
    实时预测
    :return:
    """
    # 参数设置
    model = load_model()

    border_color = (0, 0, 0)  # 黑框框
    font_color = (255, 255, 255)  # 白字字

    _setup_console_encoding()

    # 1. 根据参数选择视频源
    if opt.source == 0:
        # 使用摄像头
        # CAP_DSHOW is usually more stable on Windows and reduces black screen issues.
        capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture = cv2.VideoCapture(0)
        source_info = "摄像头"
    elif opt.source == 1:
        # 使用视频文件
        if opt.video_path is None:
            print("错误：使用视频模式时需要提供 --video_path 参数")
            return
        if not os.path.exists(opt.video_path):
            print(f"错误：视频文件不存在 - {opt.video_path}")
            return
        capture = cv2.VideoCapture(opt.video_path)
        source_info = f"视频文件: {opt.video_path}"
    else:
        print(f"错误：不支持的源类型 - {opt.source}")
        return

    # 2. 检查视频源是否成功打开
    if not capture.isOpened():
        print(f"错误：无法打开视频源 '{source_info}'")
        return

    print(f"已成功打开: {source_info}")
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    while True:
        # User clicked the window close button.
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

        # 3. 读取帧
        ret, frame = capture.read()
        if not ret:
            # 如果 ret 为 False，说明视频已结束或读取失败
            print("视频已结束或无法读取帧。")
            break

        # 调整帧大小
        frame = cv2.resize(frame, (800, 600))
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 灰度化

        # 人脸检测
        faces = detect_face(frame)

        if faces is not None and len(faces) > 0:
            for (x, y, w, h) in faces:
                x = max(0, x)
                y = max(0, y)
                w = min(w, frame_gray.shape[1] - x)
                h = min(h, frame_gray.shape[0] - y)
                face = frame_gray[y: y + h, x: x + w]
                # 确保人脸区域有效
                if face.size > 0:
                    faces_aug = generate_faces(face)
                    results = model.predict(faces_aug)
                    result_sum = np.sum(results, axis=0).reshape(-1)
                    label_index = np.argmax(result_sum, axis=0)
                    emotion = index2emotion(label_index)
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), border_color, thickness=2)
                    frame = cv2_img_add_text(frame, emotion, x + 30, y + 30, font_color, 20)

        # 显示帧
        cv2.imshow(WINDOW_NAME, frame)

        # 退出条件
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q'), ord('Q')):  # ESC / Q
            break

    # 释放资源
    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    predict_expression()
