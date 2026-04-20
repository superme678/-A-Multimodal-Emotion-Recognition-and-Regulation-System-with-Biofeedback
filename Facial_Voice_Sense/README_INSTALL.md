# Facial Voice Sense - 虚拟环境配置安装指南

## 快速开始

### Windows系统

```powershell
# 1. 创建虚拟环境
python -m venv .venv

# 2. 激活虚拟环境
.venv\Scripts\Activate.ps1

# 3. 升级pip
python -m pip install --upgrade pip

# 4. 安装依赖
pip install -r requirements.txt

# 5. 验证安装
python check_environment.py
```

### Linux/macOS系统

```bash
# 1. 创建虚拟环境
python3 -m venv .venv

# 2. 激活虚拟环境
source .venv/bin/activate

# 3. 升级pip
python -m pip install --upgrade pip

# 4. 安装依赖
pip install -r requirements.txt

# 5. 验证安装
python check_environment.py
```

## 详细安装步骤

### 1. 环境准备

#### 1.1 检查Python版本
```bash
python --version
```
确保Python版本 >= 3.7，推荐使用Python 3.8或3.9

#### 1.2 检查pip版本
```bash
pip --version
```
确保pip版本 >= 20.0

### 2. 创建虚拟环境

#### Windows PowerShell
```powershell
# 在项目根目录下执行
python -m venv .venv
```

#### Windows CMD
```cmd
python -m venv .venv
```

#### Linux/macOS
```bash
python3 -m venv .venv
```

### 3. 激活虚拟环境

#### Windows PowerShell
```powershell
.venv\Scripts\Activate.ps1
```

**注意**: 如果遇到执行策略错误，请运行：
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### Windows CMD
```cmd
.venv\Scripts\activate.bat
```

#### Linux/macOS
```bash
source .venv/bin/activate
```

**验证虚拟环境是否激活**：
- Windows: 命令行前面应该显示 `(.venv)`
- Linux/macOS: 命令行前面应该显示 `(.venv)`

### 4. 升级pip和安装工具

```bash
# 升级pip
python -m pip install --upgrade pip

# 安装wheel（加速包安装）
pip install wheel
```

### 5. 安装项目依赖

#### 5.1 安装所有依赖
```bash
pip install -r requirements.txt
```

#### 5.2 分步安装（如果遇到问题）

**步骤1: 安装基础依赖**
```bash
pip install numpy Pillow opencv-python
```

**步骤2: 安装深度学习框架**
```bash
# CPU版本（推荐）
pip install tensorflow

# GPU版本（如果有NVIDIA显卡）
pip install tensorflow-gpu
```

**步骤3: 安装音频处理库**
```bash
pip install sounddevice soundfile librosa
```

**步骤4: 安装语音识别框架**
```bash
pip install funasr
```

**步骤5: 安装其他依赖**
```bash
pip install jieba hydra-core omegaconf h5py filelock joblib
```

### 6. 验证安装

#### 6.1 运行环境检查脚本
```bash
python check_environment.py
```

#### 6.2 手动验证关键包

**验证TensorFlow**
```bash
python -c "import tensorflow as tf; print(f'TensorFlow版本: {tf.__version__}')"
```

**验证OpenCV**
```bash
python -c "import cv2; print(f'OpenCV版本: {cv2.__version__}')"
```

**验证FunASR**
```bash
python -c "import funasr; print('FunASR安装成功')"
```

**验证音频处理库**
```bash
python -c "import sounddevice, soundfile; print('音频库安装成功')"
```

### 7. 下载模型文件

#### 7.1 人脸情绪识别模型
确保以下文件存在于 `SenseFaceSmall/` 目录：
- `models/cnn3_best_weights.h5`
- `blazeface/weights/face_detection_front.tflite`
- `blazeface/weights/face_detection_back.tflite`
- `assets/simsun.ttc`

#### 7.2 语音识别模型
确保以下文件存在于 `SenseVoiceSmall/` 目录：
- `model.pt`
- `config.yaml`
- `chn_jpn_yue_eng_ko_spectok.bpe.model`
- `am.mvn`
- `tokens.json`

## 常见问题解决

### 问题1: PowerShell执行策略错误

**错误信息**：
```
无法加载文件 Activate.ps1，因为在此系统上禁止运行脚本
```

**解决方案**：
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 问题2: pip安装速度慢

**解决方案**：使用国内镜像源

```bash
# 清华源
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 阿里源
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# 中科大源
pip install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple/
```

**永久配置镜像源**：
```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### 问题3: TensorFlow安装失败

**解决方案**：使用预编译的wheel包

```bash
# 查看可用的TensorFlow版本
pip search tensorflow

# 安装特定版本
pip install tensorflow==2.10.0
```

### 问题4: FunASR安装问题

**解决方案**：
```bash
# 方法1: 使用官方源
pip install funasr

# 方法2: 从源码安装
git clone https://github.com/alibaba-damo-academy/FunASR.git
cd FunASR
pip install -e .
```

### 问题5: sounddevice安装失败

**Windows系统**：
```bash
# 通常会自动安装，如果失败可以尝试
pip install sounddevice --no-binary sounddevice
```

**Linux系统**：
```bash
# 安装PortAudio开发库
sudo apt-get install portaudio19-dev
pip install sounddevice
```

**macOS系统**：
```bash
# 安装PortAudio
brew install portaudio
pip install sounddevice
```

### 问题6: 虚拟环境无法激活

**Windows系统**：
```powershell
# 检查虚拟环境路径
Get-ChildItem .venv\Scripts\

# 手动激活
$env:VIRTUAL_ENV = "d:\AI_project\FVS(备份)\Facial_Voice_Sense\.venv"
$env:PATH = "$env:VIRTUAL_ENV\Scripts;$env:PATH"
```

**Linux/macOS系统**：
```bash
# 检查虚拟环境路径
ls -la .venv/bin/

# 手动激活
source .venv/bin/activate
```

## 环境管理

### 退出虚拟环境

```bash
deactivate
```

### 重新激活虚拟环境

**Windows**:
```powershell
.venv\Scripts\Activate.ps1
```

**Linux/macOS**:
```bash
source .venv/bin/activate
```

### 删除虚拟环境

**Windows**:
```powershell
# 先退出虚拟环境
deactivate

# 删除虚拟环境目录
Remove-Item -Recurse -Force .venv
```

**Linux/macOS**:
```bash
# 先退出虚拟环境
deactivate

# 删除虚拟环境目录
rm -rf .venv
```

### 导出当前环境依赖

```bash
pip freeze > requirements_installed.txt
```

### 比较依赖差异

```bash
pip freeze | diff - requirements.txt -
```

## 性能优化

### 1. 使用GPU加速

如果有NVIDIA显卡，可以安装GPU版本的TensorFlow：

```bash
# 卸载CPU版本
pip uninstall tensorflow

# 安装GPU版本
pip install tensorflow-gpu
```

**注意**: 需要预先安装CUDA和cuDNN。

### 2. 优化pip安装速度

```bash
# 使用缓存
pip install -r requirements.txt --cache-dir ./pip_cache

# 并行安装（需要pip-tools）
pip install pip-tools
pip-compile requirements.txt
```

## 测试环境

### 测试人脸情绪检测
```bash
python SenseFaceSmall/face_emotion_detection.py --source 0
```

### 测试语音识别
```bash
python SenseVoiceSmall/test_voice.py "SenseVoiceSmall/example/zh.mp3"
```

### 测试实时语音识别
```bash
python SenseVoiceSmall/test_voice_realtime.py
```

### 测试多模态联合识别
```bash
python test_multimodal_realtime.py --show-face-window
```

## 项目目录结构

```
Facial_Voice_Sense/
├── .venv/                          # 虚拟环境目录
├── SenseFaceSmall/                  # 人脸情绪检测模块
│   ├── face_emotion_detection.py
│   ├── models/
│   ├── blazeface/
│   └── assets/
├── SenseVoiceSmall/                 # 语音识别模块
│   ├── test_voice.py
│   ├── test_voice_realtime.py
│   ├── model.pt
│   └── config.yaml
├── test_multimodal_realtime.py       # 多模态联合识别
├── check_environment.py             # 环境检查工具
├── requirements.txt                 # 依赖配置文件
└── README_INSTALL.md               # 本安装指南
```

## 技术支持

如果遇到安装问题，请按以下步骤排查：

1. 运行环境检查脚本：`python check_environment.py`
2. 检查Python版本：`python --version`
3. 检查pip版本：`pip --version`
4. 查看详细错误信息
5. 参考本文档的"常见问题解决"部分

## 版本信息

- Python: 3.7+
- TensorFlow: 2.10.0+
- OpenCV: 4.5.0+
- FunASR: 1.0.0+

## 更新日志

### v1.0 (2026-04-12)
- 初始版本
- 支持人脸情绪检测
- 支持多语言语音识别
- 支持多模态联合识别