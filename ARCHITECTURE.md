# Qwen3-Medical-SFT 项目架构与代码逻辑

## 一、项目概览

本项目是一个基于 **Qwen3** 大模型的医疗领域 SFT（Supervised Fine-Tuning）微调项目，目标是将通用大语言模型微调为具备 **R1 推理风格**（带 `<think>` 思考链）的医学问答专家。

### 技术栈

| 组件 | 技术选型 | 说明 |
|------|---------|------|
| 基座模型 | Qwen3-1.7B / Qwen3-8B | 阿里通义千问第三代 |
| 模型下载 | ModelScope `snapshot_download` | 从魔搭社区拉取模型权重 |
| 训练框架 | HuggingFace Transformers `Trainer` | 标准训练循环 |
| 高效微调 | PEFT (LoRA / QLoRA) | 低秩适配，仅训练 <1% 参数 |
| 量化 | BitsAndBytes NF4 4-bit | QLoRA 所需的基座量化 |
| 实验追踪 | SwanLab | 训练 loss 曲线 + 样本预测可视化 |
| 推理服务 | FastAPI + Uvicorn | RESTful API，支持流式/非流式 |
| 数据集 | `krisfu/delicate_medical_r1_data` | 医疗问答 + 思考过程 |

---

## 二、项目文件结构

```
Qwen3-Medical-SFT/
├── data.py              # 数据下载与预处理
├── train.py             # 全参数微调（Qwen3-1.7B）
├── train_lora.py        # QLoRA 微调（Qwen3-8B）
├── train.ipynb          # Jupyter Notebook 版训练脚本
├── inference.py         # 全参数微调模型推理
├── inference_lora.py    # QLoRA 微调模型推理
├── server.py            # FastAPI 推理服务
├── requirements.txt     # Python 依赖
├── README.md            # 项目说明（中文）
├── README_EN.md         # 项目说明（英文）
├── DEPLOY.md            # 部署文档
└── readme_images/       # README 插图
```

---

## 三、数据流水线 (`data.py`)

### 流程

```
ModelScope 数据集
       │
       ▼
 MsDataset.load()     ← 从魔搭加载 krisfu/delicate_medical_r1_data
       │
       ▼
 random.shuffle()     ← seed=42 保证可复现
       │
       ▼
 90/10 划分           ← 训练集 / 验证集
       │
       ▼
 train.jsonl + val.jsonl   ← 输出 JSONL 格式
```

### 数据格式

原始数据每条包含三个字段：

```json
{
  "question": "患者的医学问题",
  "think": "医学推理思考过程",
  "answer": "最终回答"
}
```

---

## 四、训练流水线

### 4.1 数据预处理（`train.py` / `train_lora.py` 共用逻辑）

两个训练脚本共享相同的数据处理逻辑，分为两步：

#### 步骤 1：格式转换 (`dataset_jsonl_transfer`)

将原始 JSONL 转换为 instruction-input-output 三元组格式：

```
原始: { question, think, answer }
     │
     ▼
转换: {
  instruction: "你是一个医学专家...",
  input: question,
  output: "<think>{think}</think> \n {answer}"
}
```

关键设计：output 中用 `<think>...</think>` 标签包裹思考过程，训练模型学会 R1 推理风格。

#### 步骤 2：Token 化 (`process_func`)

将文本转换为模型可消费的 token 序列：

```
[system prompt tokens] + [user input tokens] + [assistant response tokens] + [pad]
         │                       │                        │
    labels = -100           labels = -100           labels = response_ids
    （不计算 loss）          （不计算 loss）          （计算 loss）
```

关键细节：
- 使用 Qwen 的 ChatML 格式（`<|im_start|>` / `<|im_end|>`）
- instruction 部分的 labels 设为 `-100`（交叉熵忽略），只在 response 部分计算 loss
- 超过 `MAX_LENGTH=2048` 的序列会被截断

### 4.2 全参数微调 (`train.py`)

| 配置项 | 值 |
|--------|-----|
| 模型 | Qwen3-1.7B |
| 精度 | bfloat16 |
| batch size | 1 × 4 (grad accum) = 4 |
| 学习率 | 1e-4 |
| Epoch | 2 |
| 显存需求 | ~32GB |

全参数微调更新模型**所有权重**，效果最优但成本最高。

### 4.3 QLoRA 微调 (`train_lora.py`)

| 配置项 | 值 |
|--------|-----|
| 模型 | Qwen3-8B |
| 量化 | NF4 4-bit + 双重量化 |
| LoRA rank | 16, alpha=32, dropout=0.05 |
| LoRA 目标层 | q/k/v/o/gate/up/down_proj（全部 7 个投影层） |
| batch size | 2 × 8 (grad accum) = 16 |
| 学习率 | 2e-4, cosine scheduler, 3% warmup |
| 优化器 | paged_adamw_8bit |
| 显存需求 | ~13-15GB |

QLoRA 的核心思想：
1. **BitsAndBytes 4-bit 量化**：基座模型以 NF4 格式加载，大幅减少显存占用
2. **LoRA 低秩适配**：冻结原始权重，只训练注入的低秩矩阵（参数量 <1%）
3. 最终效果接近全参数微调，但显存需求降低约 50%

### 4.4 训练后评估

两个训练脚本末尾都会取验证集前 3 条做主观评估：
```
验证集样本 → 模型推理 → 打印输出 → SwanLab 记录
```

---

## 五、推理模块

### 5.1 全参数推理 (`inference.py`)

```
加载原始 tokenizer
    +
加载微调后 checkpoint（完整权重）
    │
    ▼
apply_chat_template → generate → decode → 输出
```

兼容 MPS (Apple Silicon) / CUDA / CPU 三种设备。

### 5.2 QLoRA 推理 (`inference_lora.py`)

```
加载 4-bit 量化基座模型（配置必须与训练时一致）
    +
PeftModel.from_pretrained() 加载 LoRA adapter
    │
    ▼
model.eval() → 推理
```

关键注意：量化配置（`BitsAndBytesConfig`）**必须与训练时完全一致**，否则权重无法正确匹配。

---

## 六、API 服务 (`server.py`)

### 双模式架构

服务支持通过 `--mode` 参数切换两种加载模式：

```
                    ┌──────────────────────────────────┐
                    │        FastAPI Application        │
                    │                                    │
 HTTP Request ──►   │  POST /chat      → 非流式推理      │
                    │  POST /chat/stream → SSE 流式      │
                    │  GET  /health    → 健康检查+模式    │
                    │                                    │
                    │  ┌────────────────────────────┐   │
                    │  │   inference_lock (Mutex)    │   │
                    │  │                            │   │
                    │  │  --mode lora               │   │
                    │  │  ┌──────────────────────┐  │   │
                    │  │  │ 基座模型 (NF4 4-bit)  │  │   │
                    │  │  │ + QLoRA Adapter       │  │   │
                    │  │  └──────────────────────┘  │   │
                    │  │                            │   │
                    │  │  --mode full               │   │
                    │  │  ┌──────────────────────┐  │   │
                    │  │  │ 完整 checkpoint 权重   │  │   │
                    │  │  │ (bfloat16 直接加载)   │  │   │
                    │  │  └──────────────────────┘  │   │
                    │  └────────────────────────────┘   │
                    └──────────────────────────────────┘
```

### 启动方式

```bash
# QLoRA 模式（默认）— Qwen3-8B + 4-bit 量化 + LoRA adapter
python server.py --mode lora

# 全参微调模式 — Qwen3-1.7B 完整 checkpoint
python server.py --mode full

# 自定义模型路径和 checkpoint
python server.py --mode full --model-path ./Qwen/Qwen3-1.7B --checkpoint ./output/Qwen3-1.7B/checkpoint-1084
```

### 模型加载逻辑

| 参数 | `--mode lora`（默认） | `--mode full` |
|------|----------------------|---------------|
| 默认 model-path | `./Qwen/Qwen3-8B` | `./Qwen/Qwen3-1.7B` |
| 默认 checkpoint | `./output/Qwen3-8B/checkpoint-best` | `./output/Qwen3-1.7B/checkpoint-1084` |
| 加载方式 | BitsAndBytes 4-bit 量化基座 + `PeftModel` adapter | `AutoModelForCausalLM` 直接加载 checkpoint |
| 推理显存 | ~6-7GB | ~4-5GB |

### 核心设计

1. **双模式加载**：`lifespan` 根据 `--mode` 参数调用 `load_model_lora()` 或 `load_model_full()`，上层推理逻辑完全共用
2. **Lifespan 管理**：FastAPI 的 `lifespan` 上下文管理器负责启动时加载模型、关闭时释放显存
3. **线程安全**：`inference_lock` (threading.Lock) 保证单卡环境下推理请求串行执行，避免 GPU 显存冲突
4. **流式输出**：使用 Transformers 的 `TextIteratorStreamer`，在独立线程中运行 `model.generate()`，主线程通过迭代器逐 token 返回 SSE 事件
5. **R1 响应解析**：`parse_thinking_response()` 将 `<think>...</think>` 标签拆分为思考过程和最终回答，API 返回结构化的 `thinking` + `answer` 字段

### API 端点

| 端点 | 方法 | 功能 |
|------|------|------|
| `/chat` | POST | 非流式问答，返回完整 JSON |
| `/chat/stream` | POST | SSE 流式输出，逐 token 推送 |
| `/health` | GET | 健康检查 + GPU 显存状态 + 当前模式 |

---

## 七、端到端工作流

```
┌──────────┐    ┌──────────┐    ┌──────────────────┐    ┌──────────────┐    ┌──────────┐
│  data.py │───►│ train.py │───►│ inference.py     │    │              │    │  客户端   │
│ 数据准备  │    │ 全参微调  │    │ 本地推理测试      │    │  server.py   │◄───│ curl/Web │
│          │    │    或     │───►│       或         │───►│  API 服务    │    │          │
│          │    │train_lora│    │ inference_lora.py │    │              │    │          │
└──────────┘    └──────────┘    └──────────────────┘    └──────────────┘    └──────────┘

  Phase 1          Phase 2           Phase 3               Phase 4
  数据准备          模型训练           推理验证              生产部署
```

### 各阶段说明

1. **数据准备**：从 ModelScope 下载医疗问答数据集，划分训练集/验证集
2. **模型训练**：选择全参微调（小模型 1.7B）或 QLoRA（大模型 8B），使用 Trainer API 训练
3. **推理验证**：加载 checkpoint，用测试样本验证模型输出质量
4. **生产部署**：启动 FastAPI 服务，对外提供 HTTP API

---

## 八、关键技术决策

### 为什么用 R1 推理风格？

传统 SFT 只训练模型直接给出答案。R1 风格在答案前加入 `<think>` 思考链，让模型"先想后说"：
- 提升复杂医学问题的推理准确性
- 思考过程可审计，增强可信度
- 对应 DeepSeek-R1 的技术路线

### 为什么同时提供全参微调和 QLoRA？

| | 全参微调 | QLoRA |
|--|---------|-------|
| 适用模型 | 较小模型 (1.7B) | 较大模型 (8B) |
| 显存需求 | ~32GB | ~13-15GB |
| 训练效果 | 略优 | 接近全参 |
| 推理部署 | 直接加载 | 需加载基座 + adapter |

用户可根据自身算力条件选择方案。

### 为什么用 ChatML 格式而非 `apply_chat_template`？

训练阶段手动拼接 ChatML token（`<|im_start|>` / `<|im_end|>`）是为了精确控制 labels 的 -100 mask，确保 loss 只在 response 部分计算。推理阶段则用 `apply_chat_template` 简化处理。
