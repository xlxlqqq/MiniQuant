第一阶段 — 一个 Linear

目标: 实现 Linear 层的量化/反量化、序列化接口。
任务: quant 原语（int8.py/int4.py/pack.py）、单元测试、快速数值回归（相对 FP32）。
产出: 精简 demo 脚本验证相对误差和推理速度。
第二阶段 — 一个 Transformer Block

目标: 实现自注意力 + MLP（含 LayerNorm）以及对应量化流程。
任务: Block 实现、融合量化路径、端到端单块准确性和性能测试（单步延时）。
产出: Block-level 基准与数值比对脚本。
第三阶段 — TinyLlama / 0.5B

目标: 从头构建小模型配置并跑通训练/推理/量化流程。
任务: model 定义 (models/qwen2.py 风格)、权重加载/转换、calibration 数据集采集、量化并评估 PPL。
产出: 可复现的实验目录与 baseline 结果。
第四阶段 — Qwen2.5-0.5B

目标: 把 Qwen2.5 的小尺度变体接入并验证精度/速度。
任务: 调整 tokenizer/模型接口，运行 experiments/exp_int8.py / exp_int4.py，记录 PPL 与推理延时（TTFT）。
产出: 精度-性能曲线与配置建议。
第五阶段 — Qwen2.5-1.5B

目标: 扩展到中等规模，检查量化鲁棒性和显存占用。
任务: 分组/通道量化 (per_group.py/per_channel.py)、实验 group-size 扫描、内存/吞吐基准。
产出: group size vs accuracy/VRAM 折线图与参数表。
最后 — Qwen2.5-7B

目标: 全尺度集成，优化推理内核（CPU/CUDA/MUSA）并量化到生产级别。
任务:
实现/集成 GPTQ/AWQ（algorithms/gptq.py、awq.py），
编写/优化 kernels/（先 CPU 原型，再 CUDA、MUSA），
运行端到端基准：PPL、TTFT、TPS、VRAM。
产出: 最终报告（accuracy vs throughput）, 可复现的 benchmark 脚本和配置。
横向任务（跨阶段）

校准: calibration/collect_activation.py 与 calibration_dataset.py（自动化采样与缓存）。
Bench: benchmark/accuracy.py / latency.py / memory.py（统一 API）。
测试与 CI: 单元测试、端到端回归、性能基线自动化。
文档: 使用说明和复现实验步骤（README.md 与示例命令）。