# LIBERO + π0.5 手写工作区

把自己的 Python 代码放在 `code/`。用 VS Code 打开本目录时，选择 `.venv/bin/python` 作为解释器。

```bash
cd ~/CoRL/LIBERO_pi05
source env.sh
```

| 路径 | 内容 |
| --- | --- |
| `openpi/` | 已有的官方 openpi 源码，提交 `215abfb`，含原版 LIBERO 子模块 |
| `pi05_libero/` | 已有的官方 LIBERO 微调权重，已转换为 PyTorch，附归一化统计 |
| `libero_data/` | 已有 LIBERO 演示数据 |
| `openpi/examples/libero/main.py` | 官方仿真示例，适合对照 observations 和动作的接口 |

三个资源目录都是链接。修改链接指向的源码或文件会影响原目录，自己的实现写在 `code/`。这里的小型 `.venv` 复用现有依赖，原依赖的路径或版本变动会影响此环境。缓存使用硬链接复用 tokenizer 文件。

创建目录和 Python 环境几乎不占额外磁盘空间。推理时仍需给模型分配内存和显存。在这台 8 GB 显卡上，之前已用 CPU 分担视觉编码跑通一条闭环，全 GPU 设置遇到过显存不足。
