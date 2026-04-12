# Qwen3-8B 医疗 QLoRA 微调 — 部署使用方案

## 一、服务器环境要求

| 项目 | 要求 |
|---|---|
| 系统 | Ubuntu 22.04 |
| GPU | NVIDIA RTX 4090 x1 (24GB) |
| CPU | 16 核 |
| 内存 | 64GB |
| CUDA | 12.1+ |
| Python | 3.10+ |
| 磁盘 | 至少 50GB 可用空间（模型 ~16GB + 数据集 + checkpoint） |

## 二、环境搭建

### 2.1 系统依赖

```bash
sudo apt update && sudo apt install -y python3-venv python3-pip git
```

### 2.2 克隆项目

```bash
git clone https://github.com/davis-you/Qwen3-Medical-SFT.git
cd Qwen3-Medical-SFT
```

### 2.3 创建虚拟环境

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2.4 安装 PyTorch（CUDA 12.1）

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

> 如果服务器 CUDA 版本不同，参考 https://pytorch.org/get-started/locally/ 选择对应版本。

### 2.5 安装项目依赖

```bash
pip install -r requirements.txt
```

## 三、数据准备

```bash
python data.py
```

输出：
```
数据集已分割完成：
训练集大小：xxx
验证集大小：xxx
```

会自动从 ModelScope 下载 `krisfu/delicate_medical_r1_data` 医疗问答数据集，生成 `train.jsonl` 和 `val.jsonl`。

## 四、微调训练

项目提供两种微调方案，根据算力条件选择：

| 方案 | 脚本 | 模型 | 显存需求 | 训练时间 |
|------|------|------|---------|---------|
| **方案 A：QLoRA 微调** | `train_lora.py` | Qwen3-8B | ~13-15GB | ~2-4 小时 |
| **方案 B：全参微调** | `train.py` | Qwen3-1.7B | ~32GB | ~1-2 小时 |

### 方案 A：QLoRA 微调（Qwen3-8B）

#### A.1 开始训练

```bash
python train_lora.py
```

#### A.2 训练参数说明

| 参数 | 值 | 说明 |
|---|---|---|
| 基座模型 | Qwen3-8B | 从 ModelScope 自动下载 |
| 量化方式 | NF4 4-bit + 双重量化 | QLoRA，显存 ~13-15GB |
| LoRA rank | 16 | 适配 8B 模型容量 |
| LoRA target | 全部 7 个投影层 | q/k/v/o/gate/up/down_proj |
| 有效 batch size | 16 | batch=2 × grad_accum=8 |
| 学习率 | 2e-4 | cosine scheduler + 3% warmup |
| 优化器 | paged_adamw_8bit | 显存友好 |
| Epoch | 2 | |
| 序列长度 | 2048 | |

#### A.3 训练产物

```
output/Qwen3-8B/
├── checkpoint-200/    ← LoRA adapter 权重
├── checkpoint-400/
├── checkpoint-600/
└── ...
```

> 注意：QLoRA 的 checkpoint 只包含 LoRA adapter 权重（~几十 MB），推理时需要同时加载基座模型 + adapter。

### 方案 B：全参微调（Qwen3-1.7B）

#### B.1 开始训练

```bash
python train.py
```

#### B.2 训练参数说明

| 参数 | 值 | 说明 |
|---|---|---|
| 基座模型 | Qwen3-1.7B | 从 ModelScope 自动下载 |
| 微调方式 | 全参数微调 | 更新所有模型权重 |
| 有效 batch size | 4 | batch=1 × grad_accum=4 |
| 学习率 | 1e-4 | 线性 scheduler |
| 精度 | bfloat16 | |
| Epoch | 2 | |
| 序列长度 | 2048 | |

#### B.3 训练产物

```
output/Qwen3-1.7B/
├── checkpoint-400/    ← 完整模型权重
├── checkpoint-800/
├── checkpoint-1084/
└── ...
```

> 注意：全参微调的 checkpoint 包含完整模型权重（~3.4GB），推理时直接加载 checkpoint 即可。

### 通用说明

#### 显存监控

训练过程中可另开终端监控：

```bash
watch -n 1 nvidia-smi
```

预期显存占用：
- **QLoRA (8B)**：~13-15GB
- **全参微调 (1.7B)**：~32GB

#### 训练可视化

项目集成了 SwanLab，训练过程中会自动记录 loss 曲线和样本预测。首次运行会要求登录 SwanLab 账号（可选跳过）。

#### 两种方案效果对比

经测试全参数微调效果略优于 QLoRA，但 QLoRA 可以训练更大的模型（8B vs 1.7B），综合效果 QLoRA 8B 更有优势。

## 五、推理测试

### 方案 A：QLoRA 推理

编辑 `inference_lora.py`，将 `LORA_CHECKPOINT` 改为实际的 checkpoint 路径：

```python
LORA_CHECKPOINT = "./output/Qwen3-8B/checkpoint-400"  # 改为你的实际路径
```

```bash
python inference_lora.py
```

会加载 4-bit 量化基座模型 + QLoRA adapter，对一个糖尿病问题进行推理测试。

### 方案 B：全参微调推理

编辑 `inference.py`，将 checkpoint 路径改为实际路径：

```python
model = AutoModelForCausalLM.from_pretrained("./output/Qwen3-1.7B/checkpoint-1084", ...)
```

```bash
python inference.py
```

直接加载完整 checkpoint 权重进行推理，兼容 CUDA / MPS (Apple Silicon) / CPU。

## 六、启动 API 服务

API 服务支持两种模式，通过 `--mode` 参数切换：

| 模式 | 说明 | 显存占用 |
|------|------|---------|
| `lora`（默认） | Qwen3-8B + 4-bit 量化 + QLoRA adapter | ~6-7GB |
| `full` | Qwen3-1.7B 全参微调 checkpoint 直接加载 | ~4-5GB |

### 6.1 QLoRA 模式启动（默认）

```bash
python server.py --mode lora --port 8000 --checkpoint ./output/Qwen3-8B/checkpoint-400
```

启动后日志：
```
INFO:     Starting server: mode=lora, model=./Qwen/Qwen3-8B, checkpoint=./output/Qwen3-8B/checkpoint-400
INFO:     [lora] Loading tokenizer from ./Qwen/Qwen3-8B
INFO:     [lora] Loading base model in 4-bit from ./Qwen/Qwen3-8B
INFO:     [lora] Loading QLoRA adapter from ./output/Qwen3-8B/checkpoint-400
INFO:     [lora] Model loaded successfully
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 6.1.1 全参微调模式启动

```bash
python server.py --mode full --port 8000 --checkpoint ./output/Qwen3-1.7B/checkpoint-1084
```

启动后日志：
```
INFO:     Starting server: mode=full, model=./Qwen/Qwen3-1.7B, checkpoint=./output/Qwen3-1.7B/checkpoint-1084
INFO:     [full] Loading tokenizer from ./Qwen/Qwen3-1.7B
INFO:     [full] Loading model from ./output/Qwen3-1.7B/checkpoint-1084
INFO:     [full] Model loaded successfully
INFO:     Uvicorn running on http://0.0.0.0:8000
```

也可以自定义模型路径：
```bash
python server.py --mode full --model-path ./Qwen/Qwen3-1.7B --checkpoint ./output/Qwen3-1.7B/checkpoint-1084 --port 8080
```

### 6.2 API 端点

#### POST /chat — 非流式问答

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "糖尿病患者饮食需要注意什么？",
    "temperature": 0.7,
    "top_p": 0.9
  }'
```

响应示例：
```json
{
  "question": "糖尿病患者饮食需要注意什么？",
  "response": "<think>...思考过程...</think>\n最终回答...",
  "thinking": "...思考过程...",
  "answer": "最终回答...",
  "elapsed_seconds": 12.34
}
```

#### POST /chat/stream — 流式问答（SSE）

```bash
curl -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "高血压的治疗方案有哪些？"}' \
  --no-buffer
```

返回 Server-Sent Events 流，逐 token 输出。

#### GET /health — 健康检查

```bash
curl http://localhost:8000/health
```

响应：
```json
{
  "status": "healthy",
  "mode": "lora",
  "model_loaded": true,
  "gpu_memory_allocated_gb": 5.82,
  "gpu_memory_reserved_gb": 6.14,
  "gpu_total_memory_gb": 24.0
}
```

### 6.3 请求参数说明

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `question` | string | 必填 | 医学问题（1-4096 字符） |
| `system_prompt` | string | 医学专家提示词 | 自定义系统提示词 |
| `max_tokens` | int | 2048 | 最大生成 token 数 |
| `temperature` | float | 0.7 | 采样温度（0=贪婪） |
| `top_p` | float | 0.9 | 核采样概率 |
| `stream` | bool | false | 是否流式（用 /chat/stream 更好） |

### 6.4 后台运行（生产部署）

```bash
# QLoRA 模式后台运行
nohup python server.py --mode lora --port 8000 --checkpoint ./output/Qwen3-8B/checkpoint-400 \
  > server.log 2>&1 &

# 全参微调模式后台运行
# nohup python server.py --mode full --port 8000 --checkpoint ./output/Qwen3-1.7B/checkpoint-1084 \
#   > server.log 2>&1 &

# 查看日志
tail -f server.log

# 停止服务
kill $(pgrep -f "server.py")
```

或使用 systemd 管理：

```bash
sudo tee /etc/systemd/system/medical-api.service << 'EOF'
[Unit]
Description=Qwen3 Medical API
After=network.target

[Service]
User=你的用户名
WorkingDirectory=/path/to/Qwen3-Medical-SFT
ExecStart=/path/to/Qwen3-Medical-SFT/venv/bin/python server.py --mode lora --port 8000 --checkpoint ./output/Qwen3-8B/checkpoint-400
Restart=on-failure
RestartSec=10
Environment=CUDA_VISIBLE_DEVICES=0

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable medical-api
sudo systemctl start medical-api
sudo systemctl status medical-api
```

## 七、完整流程速查

```
1. 环境搭建       → pip install torch + pip install -r requirements.txt
2. 数据准备       → python data.py

方案 A — QLoRA 微调 (Qwen3-8B):
3a. 训练          → python train_lora.py                              （约 2-4 小时）
4a. 推理测试      → python inference_lora.py
5a. 启动 API      → python server.py --mode lora --port 8000

方案 B — 全参微调 (Qwen3-1.7B):
3b. 训练          → python train.py                                   （约 1-2 小时）
4b. 推理测试      → python inference.py
5b. 启动 API      → python server.py --mode full --port 8000

6. 调用接口       → curl POST /chat
```

## 八、常见问题

### Q: 训练时 OOM 怎么办？
将 `train_lora.py` 中 `per_device_train_batch_size` 改为 `1`，或将 `MAX_LENGTH` 降为 `1024`。

### Q: bitsandbytes 报 CUDA 版本错误？
确认 PyTorch CUDA 版本和系统 CUDA toolkit 一致：
```bash
python -c "import torch; print(torch.version.cuda)"
nvidia-smi  # 查看驱动 CUDA 版本
```

### Q: 怎么选最佳 checkpoint？
查看训练 log 中 eval_loss 最低的 checkpoint，或在 SwanLab 上对比各 checkpoint 的 loss 曲线。

### Q: API 能否支持并发？
单卡部署下，`inference_lock` 会自动序列化请求（一次只处理一个），其他请求排队等待。如需更高 QPS，考虑多卡部署或使用 vLLM 推理框架。
