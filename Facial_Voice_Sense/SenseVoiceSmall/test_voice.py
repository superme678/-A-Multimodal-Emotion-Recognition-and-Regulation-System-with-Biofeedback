import sys
import os
from pathlib import Path
import ctypes
import subprocess
from funasr import AutoModel
import numpy as np
import soundfile as sf
import time

# ========== 配置项 ==========
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = str(BASE_DIR)
SUPPORTED_FORMATS = ['.wav', '.mp3', '.m4a', '.flac', '.aac', '.ogg', '.wma']

# ========== 全局模型变量 ==========
_model = None


def _setup_console_encoding():
    """避免 Windows 控制台输出中文/符号乱码。"""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass


def _to_windows_short_path(path_str: str) -> str:
    """Return 8.3 short path on Windows to avoid non-ASCII path issues."""
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


def _resolve_model_path(model_path: str) -> str:
    """Resolve a Windows-friendly model path for FunASR."""
    path_str = str(model_path)
    if os.name != "nt":
        return path_str

    # Keep original path when it is already ASCII-only.
    try:
        path_str.encode("ascii")
        return path_str
    except UnicodeEncodeError:
        pass

    src = Path(path_str)
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
        # Fallback to short path when junction creation is unavailable.
        return _to_windows_short_path(path_str)


def init_model(model_path=MODEL_PATH):
    """
    初始化模型（只需调用一次）
    
    参数:
        model_path: 模型路径
    
    返回:
        bool: 是否成功
    """
    global _model
    try:
        print("正在加载模型...")
        model_path = _resolve_model_path(str(model_path))
        _model = AutoModel(
            model=model_path,
            vad_model=None,
            punc_model=None,
            device="cpu",
            disable_update=True
        )
        print("[OK] 模型加载成功")
        return True
    except Exception as e:
        print(f"[ERROR] 模型加载失败：{e}")
        return False


def recognize_audio(audio_path):
    """
    识别单个音频文件的主接口
    
    参数:
        audio_path: 音频文件路径 (str)
    
    返回:
        dict: 识别结果
            - success (bool): 是否成功
            - text (str): 识别文本
            - language (str): 语言类型
            - emotion (str): 情感标签
            - event (str): 事件类型
            - duration (float): 音频时长（秒）
            - infer_time (float): 推理耗时（秒）
            - error (str): 错误信息（如果失败）
    
    示例:
        >>> result = recognize_audio("test.wav")
        >>> if result['success']:
        ...     print(f"识别结果：{result['text']}")
    """
    global _model
    
    result = {
        'success': False,
        'text': '',
        'language': '',
        'emotion': '',
        'event': '',
        'duration': 0,
        'infer_time': 0,
        'error': ''
    }
    
    # 检查模型是否已加载
    if _model is None:
        if not init_model():
            result['error'] = '模型未初始化'
            return result
    
    # 检查文件是否存在
    if not Path(audio_path).exists():
        result['error'] = '文件不存在'
        return result
    
    # 检查文件格式
    file_ext = Path(audio_path).suffix.lower()
    if file_ext not in SUPPORTED_FORMATS:
        result['error'] = f'不支持的格式：{file_ext}'
        return result
    
    try:
        # 加载音频
        data, sr = sf.read(audio_path)
        
        # 转单声道
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        
        # 重采样到 16kHz
        if sr != 16000:
            data = np.interp(
                np.arange(0, len(data), sr/16000),
                np.arange(0, len(data)),
                data
            )
        
        # 归一化
        data = data.astype(np.float32)
        
        # 计算时长
        duration = len(data) / 16000
        result['duration'] = duration
        
        # 执行识别
        start_time = time.time()
        recognize_result = _model.generate(
            input=data,
            cache={},
            language="auto",
            use_itn=False
        )
        end_time = time.time()
        
        # 解析结果
        raw_text = recognize_result[0]['text']
        
        # 提取语言标签
        if '<|zh|>' in raw_text:
            result['language'] = '中文'
        elif '<|en|>' in raw_text:
            result['language'] = '英文'
        elif '<|ja|>' in raw_text:
            result['language'] = '日文'
        elif '<|ko|>' in raw_text:
            result['language'] = '韩文'
        elif '<|yue|>' in raw_text:
            result['language'] = '粤语'
        
        # 提取情感标签
        emotion_map = {
            'HAPPY': '开心', 'SAD': '悲伤', 'ANGRY': '愤怒',
            'FEARFUL': '害怕', 'NEUTRAL': '中性', 'DISGUSTED': '厌恶',
            'SURPRISED': '惊讶'
        }
        for emo_cn, emo_en in emotion_map.items():
            if f'<|{emo_en}|>' in raw_text or f'<|{emo_en.upper()}|>' in raw_text:
                result['emotion'] = emo_en
                break
        
        # 提取事件标签
        event_map = {
            'Speech': '语音', 'Laughter': '笑声', 'Applause': '掌声',
            'Cough': '咳嗽', 'Sneeze': '喷嚏', 'Cry': '哭声', 'Music': '音乐'
        }
        for event_cn, event_en in event_map.items():
            if f'<|{event_en}|>' in raw_text:
                result['event'] = event_cn
                break
        
        # 清理文本
        tags_to_remove = [
            '<|zh|>', '<|en|>', '<|ja|>', '<|ko|>', '<|yue|>',
            '<|HAPPY|>', '<|SAD|>', '<|ANGRY|>', '<|FEARFUL|>', '<|NEUTRAL|>',
            '<|DISGUSTED|>', '<|SURPRISED|>',
            '<|Speech|>', '<|Laughter|>', '<|Applause|>', '<|Cough|>',
            '<|Sneeze|>', '<|Cry|>', '<|Music|>',
            '<|/zh|>', '<|/en|>', '<|/ja|>', '<|/ko|>', '<|/yue|>',
            '<|/HAPPY|>', '<|/SAD|>', '<|/ANGRY|>', '<|/FEARFUL|>', '<|/NEUTRAL|>',
            '<|/Speech|>', '<|/Laughter|>', '<|/Applause|>', '<|/Cough|>',
            '<|/Sneeze|>', '<|/Cry|>', '<|/Music|>',
            '<|woitn|>', '<|withitn|>'
        ]
        
        clean_txt = raw_text
        for tag in tags_to_remove:
            clean_txt = clean_txt.replace(tag, '')
        
        result['text'] = clean_txt.strip()
        result['infer_time'] = end_time - start_time
        result['success'] = True
        
    except Exception as e:
        result['error'] = f'处理失败：{e}'
        import traceback
        traceback.print_exc()
    
    return result


def get_input_file_path():
    """
    获取用户输入的音频文件路径
    
    返回:
        str: 音频文件路径，如果用户取消则返回 None
    """
    if len(sys.argv) > 1:
        # 从命令行参数获取
        target_path = sys.argv[1]
        print(f"使用命令行参数：{target_path}")
    else:
        # 提示用户输入
        print("\n请输入音频文件路径:")
        print("示例：D:\\AI_project\\iic\\talk_material\\voice.m4a")
        print("(直接回车使用默认测试文件)")
        
        target_path = input("> ").strip().strip('"')
        
        if not target_path:
            # 使用默认测试文件
            default_test = r"D:\AI_project\iic\SenseVoiceSmall\example\zh.mp3"
            if Path(default_test).exists():
                target_path = default_test
                print(f"使用默认测试文件：{default_test}")
            else:
                print("[ERROR] 默认测试文件不存在")
                return None
    
    # 验证路径
    target_path = Path(target_path)
    
    if not target_path.exists():
        print(f"[ERROR] 文件不存在：{target_path}")
        return None
    
    if target_path.is_dir():
        print(f"[ERROR] 错误：请输入文件路径，不是目录")
        return None
    
    return str(target_path)


if __name__ == "__main__":
    _setup_console_encoding()
    print("=" * 60)
    print("SenseVoice 语音识别系统")
    print("=" * 60)
    
    # 初始化模型
    if not init_model():
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("使用说明")
    print("=" * 60)
    print("用法：python test_voice.py <音频文件路径>")
    print("=" * 60)
    print(f"支持的格式：{', '.join(SUPPORTED_FORMATS)}")
    print("=" * 60)
    
    # 获取输入文件路径
    audio_path = get_input_file_path()

    if audio_path is None:
        print("\n未选择文件，程序退出")
        sys.exit(0)
    
    # 处理单个文件
    print(f"\n正在处理：{Path(audio_path).name}")
    result = recognize_audio(audio_path)
    
    if result['success']:
        print("\n" + "=" * 60)
        print("识别成功")
        print("=" * 60)
        print(f"文本：{result['text']}")
        print(f"语言：{result['language']}")
        print(f"情感：{result['emotion']}")
        print(f"事件：{result['event']}")
        print(f"时长：{result['duration']:.2f}秒")
        print(f"推理耗时：{result['infer_time']:.3f}秒")
        print(f"实时率：{result['infer_time']/result['duration']:.2f}x")
    else:
        print(f"\n[ERROR] 识别失败：{result['error']}")
