# 扩展评测数据（可选）

当前 `python -m evaluation.run_benchmark` 已内置：

- **生理**：合成 `build_physio_frame` 金样 + 随机字节误检压力（固定种子）。
- **融合 / ASR 标签**：构造用例。

若审稿或课题要求 **人脸表情分类准确率**，请自备标注数据并扩展脚本，例如：

1. **FER-2013**（公开）：将 `train.csv` 置于 `evaluation_data/fer2013/`，图片解压为 `evaluation_data/fer2013/images/`，再实现按索引加载 48×48 灰度图与标签，调用 `SenseFaceSmall.face_emotion_detection.load_model()` 批量 `predict` 计算 Top-1 准确率与本项目 8 类映射表（注意 FER 为 7 类常用子集时需合并类别）。
2. **自建采集**：用同一摄像头协议录制短视频，人工帧级标注后统计混淆矩阵。

将上述脚本存为 `evaluation/bench_face_labeled.py` 并在 `run_benchmark.py` 中按需 `import` 即可把结果并入 JSON/Markdown 报告。
