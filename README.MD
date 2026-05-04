本文件分为两个部分：分别为面部识别与语音识别
面部识别模型：
运行命令：python face_emotion_detection.py可独立测试
语音识别模型：
运行命令：python test_voice.py进行文件格式下的音频文件测试
运行命令：python test_voice_real_time.py进行实时语音识别测试

运行命令：python test_multimodal_realtime.py 进行面部与语音双模态测试

三模态（含串口生理）量化测试与报告：在 `Facial_Voice_Sense` 目录执行
`python -m evaluation.run_benchmark`（可选 `--skip-face`），见 `Facial_Voice_Sense/evaluation/` 与生成的 `evaluation/reports/*.md`。
