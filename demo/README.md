# Demo 脚本

这些脚本用于学习和检查数据读取、6 通道输入、特征尺寸和一次前向传播，不用于正式训练或论文指标统计。

示例（在项目根目录运行）：

```bash
python demo/read.py --data-root <DATASET_ROOT> --split train --file-name video1_frame_00432.jpg
python demo/tensor.py --data-root <DATASET_ROOT> --split train --file-name video1_frame_00432.jpg
python demo/show.py --data-root <DATASET_ROOT> --split train --file-name video1_frame_00432.jpg
python demo/ratio2.py --data-root <DATASET_ROOT> --split train --file-name video1_frame_00432.jpg
```

输出默认写入 `runs/demo/`，不会写回原始数据集目录。
