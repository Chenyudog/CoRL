# LIBERO + π0.5 手写工作区

把自己的 Python 代码放在 `code/`。用 VS Code 打开本目录时，选择 `.venv/bin/python` 作为解释器。

```bash
cd ~/CoRL/LIBERO_pi05
source env.sh
```

| 路径 | 内容 |
| --- | --- |
| `openpi/` | 移入本工作区的官方 openpi 源码，提交 `215abfb`，含原版 LIBERO 子模块 |
| `pi05_libero/` | 已有的官方 LIBERO 微调权重，已转换为 PyTorch，附归一化统计 |
| `openpi/examples/libero/main.py` | 官方仿真示例，适合对照 observations 和动作的接口 |

`pi05_libero/` 链接到已有模型文件。原 openpi 路径保留了兼容链接。自己的实现写在 `code/`。

工作区根目录的小型 `.venv` 复用随 openpi 一起移入的 `openpi/.venv`，其中包含 PyTorch、CUDA 用户态库、Transformers、JAX 等。仿真依赖放在或链接到 `.sim-deps/`，tokenizer 通过硬链接复用。工作区不包含 LIBERO 演示数据，原版仿真场景和初始状态仍随 openpi 保留。

源码和依赖在同一文件系统内移动，未复制第二份。推理时仍需给模型分配内存和显存。在这台 8 GB 显卡上，之前已用 CPU 分担视觉编码跑通一条闭环，全 GPU 设置遇到过显存不足。
