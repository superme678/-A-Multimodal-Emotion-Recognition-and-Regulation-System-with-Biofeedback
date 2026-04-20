"""
环境检查脚本 - 检查项目所需的所有依赖和环境配置
"""
import sys
import os

def check_python_version():
    """检查Python版本"""
    print("=" * 60)
    print("Python版本检查")
    print("=" * 60)
    version = sys.version_info
    print(f"当前Python版本: {version.major}.{version.minor}.{version.micro}")

    if version.major == 3 and version.minor >= 7:
        print("[OK] Python版本满足要求 (>= 3.7)")
        return True
    else:
        print("[ERROR] Python版本过低，建议使用 Python 3.7+")
        return False

def check_package(package_name, import_name=None):
    """检查单个包是否安装"""
    if import_name is None:
        import_name = package_name

    try:
        __import__(import_name)
        print(f"[OK] {package_name}")
        return True
    except ImportError as e:
        print(f"[ERROR] {package_name} - 未安装 ({e})")
        return False

def check_tensorflow():
    """特别检查TensorFlow（因为可能有GPU版本）"""
    try:
        import tensorflow as tf
        print(f"[OK] tensorflow - 版本: {tf.__version__}")

        # 检查是否有GPU支持
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"  └─ GPU设备: {len(gpus)} 个可用")
            for gpu in gpus:
                print(f"     • {gpu.name}")
        else:
            print("  └─ 使用CPU模式")
        return True
    except ImportError as e:
        print(f"[ERROR] tensorflow - 未安装 ({e})")
        return False

def check_funasr():
    """特别检查FunASR"""
    try:
        import funasr
        print(f"[OK] funasr - 已安装")
        return True
    except ImportError as e:
        print(f"[ERROR] funasr - 未安装 ({e})")
        return False

def check_model_files():
    """检查模型文件是否存在"""
    print("\n" + "=" * 60)
    print("模型文件检查")
    print("=" * 60)

    base_path = os.path.dirname(os.path.abspath(__file__))

    # SenseFaceSmall 模型文件
    face_models = [
        "SenseFaceSmall/models/cnn3_best_weights.h5",
        "SenseFaceSmall/blazeface/weights/face_detection_front.tflite",
        "SenseFaceSmall/blazeface/weights/face_detection_back.tflite",
        "SenseFaceSmall/assets/simsun.ttc"
    ]

    # SenseVoiceSmall 模型文件
    voice_models = [
        "SenseVoiceSmall/model.pt",
        "SenseVoiceSmall/config.yaml",
        "SenseVoiceSmall/chn_jpn_yue_eng_ko_spectok.bpe.model",
        "SenseVoiceSmall/am.mvn",
        "SenseVoiceSmall/tokens.json"
    ]

    all_exist = True

    print("\n[人脸情绪检测模型]")
    for model_path in face_models:
        full_path = os.path.join(base_path, model_path)
        if os.path.exists(full_path):
            size = os.path.getsize(full_path) / (1024 * 1024)  # MB
            print(f"[OK] {model_path} ({size:.2f} MB)")
        else:
            print(f"[ERROR] {model_path} - 文件不存在")
            all_exist = False

    print("\n[语音识别模型]")
    for model_path in voice_models:
        full_path = os.path.join(base_path, model_path)
        if os.path.exists(full_path):
            size = os.path.getsize(full_path) / (1024 * 1024)  # MB
            print(f"[OK] {model_path} ({size:.2f} MB)")
        else:
            print(f"[ERROR] {model_path} - 文件不存在")
            all_exist = False

    return all_exist

def check_audio_test_file():
    """检查测试音频文件"""
    print("\n" + "=" * 60)
    print("测试文件检查")
    print("=" * 60)

    base_path = os.path.dirname(os.path.abspath(__file__))
    test_files = [
        "talk_material/personal_voice_1.m4a",
        "SenseVoiceSmall/example/zh.mp3",
        "SenseVoiceSmall/example/en.mp3",
        "SenseVoiceSmall/example/ja.mp3",
        "SenseVoiceSmall/example/ko.mp3",
        "SenseVoiceSmall/example/yue.mp3"
    ]

    for file_path in test_files:
        full_path = os.path.join(base_path, file_path)
        if os.path.exists(full_path):
            size = os.path.getsize(full_path) / 1024  # KB
            print(f"[OK] {file_path} ({size:.2f} KB)")
        else:
            print(f"[ERROR] {file_path} - 文件不存在")

def generate_requirements():
    """生成requirements.txt内容"""
    print("\n" + "=" * 60)
    print("建议的依赖安装命令")
    print("=" * 60)
    print("""
请运行以下命令安装所需依赖：

# 基础依赖
pip install opencv-python numpy Pillow

# TensorFlow (选择其一)
pip install tensorflow              # CPU版本
# 或
pip install tensorflow-gpu          # GPU版本

# FunASR及相关依赖
pip install funasr soundfile sounddevice

# 如果安装FunASR遇到问题，可以尝试：
pip install funasr modelscope huggingface_hub

注意：
1. FunASR可能需要较新的Python版本 (>= 3.8)
2. 如果使用GPU，需要确保安装了CUDA和cuDNN
3. sounddevice在某些系统上可能需要额外安装 PortAudio
   - Windows: 通常会自动安装
   - Linux: sudo apt-get install portaudio19-dev
   - macOS: brew install portaudio
    """)

def main():
    print("\n" + "=" * 60)
    print("Facial Voice Sense - 环境检查工具")
    print("=" * 60 + "\n")

    # 1. 检查Python版本
    python_ok = check_python_version()

    # 2. 检查基础依赖
    print("\n" + "=" * 60)
    print("基础依赖包检查")
    print("=" * 60)

    packages = [
        ("opencv-python", "cv2"),
        ("numpy", "numpy"),
        ("Pillow", "PIL"),
    ]

    basic_ok = True
    for pkg_name, import_name in packages:
        if not check_package(pkg_name, import_name):
            basic_ok = False

    # 3. 检查TensorFlow
    print("\n" + "=" * 60)
    print("深度学习框架检查")
    print("=" * 60)
    tf_ok = check_tensorflow()

    # 4. 检查FunASR
    print("\n" + "=" * 60)
    print("语音识别框架检查")
    print("=" * 60)
    funasr_ok = check_funasr()

    # 5. 检查音频处理库
    print("\n" + "=" * 60)
    print("音频处理库检查")
    print("=" * 60)
    audio_ok = True
    if not check_package("soundfile"):
        audio_ok = False
    if not check_package("sounddevice"):
        audio_ok = False

    # 6. 检查模型文件
    models_ok = check_model_files()

    # 7. 检查测试文件
    check_audio_test_file()

    # 8. 总结
    print("\n" + "=" * 60)
    print("环境检查总结")
    print("=" * 60)

    issues = []
    if not python_ok:
        issues.append("Python版本过低")
    if not basic_ok:
        issues.append("缺少基础依赖包")
    if not tf_ok:
        issues.append("缺少TensorFlow")
    if not funasr_ok:
        issues.append("缺少FunASR")
    if not audio_ok:
        issues.append("缺少音频处理库")
    if not models_ok:
        issues.append("缺少模型文件")

    if issues:
        print(f"\n[WARN] 发现 {len(issues)} 个问题:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        print("\n请根据上述提示安装缺失的依赖或文件。")
        generate_requirements()
    else:
        print("\n[OK] 所有环境检查通过！可以正常运行项目。")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()