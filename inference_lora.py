import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ===================== 配置区 =====================
MODEL_LOCAL_PATH = "./Qwen/Qwen3-8B"
LORA_CHECKPOINT = "./output/Qwen3-8B/checkpoint-best"  # 改为实际 checkpoint 路径
# ==================================================


def predict(messages, model, tokenizer):
    device = "cuda"
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=2048)
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


# 4-bit 量化配置（必须与训练时一致）
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# 加载 4-bit 量化基座模型
tokenizer = AutoTokenizer.from_pretrained(MODEL_LOCAL_PATH, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_LOCAL_PATH,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
)

# 加载 QLoRA adapter
model = PeftModel.from_pretrained(model, model_id=LORA_CHECKPOINT)
model.eval()

# 测试推理
test_texts = {
    'instruction': "你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。",
    'input': "医生，我最近被诊断为糖尿病，听说碳水化合物的选择很重要，我应该选择什么样的碳水化合物呢？"
}

messages = [
    {"role": "system", "content": test_texts['instruction']},
    {"role": "user", "content": test_texts['input']}
]

response = predict(messages, model, tokenizer)
print(response)
