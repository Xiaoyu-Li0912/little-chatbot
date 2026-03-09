# config.py

# ================= 基础路径配置 =================
# 基座模型路径 (HuggingFace ID 或 本地路径)
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# 数据集路径
DATA_PATH = "data/finetune_v2.jsonl"

# 模型输出路径 (建议加上日期或参数标识，方便区分)
OUTPUT_DIR = "output/Qwen2.5-LoRA-v1"

# 名称
USER_NAME = "XY. Lee"
TARGET_NAME = "小明"
# ================= 训练超参数 (Training Arguments) =================
# 这些参数会直接传给 Transformers 的 TrainingArguments
TRAIN_ARGS = {
    "per_device_train_batch_size": 2,      # 批次大小 (显存不够就减小)
    "gradient_accumulation_steps": 8,      # 梯度累积 (显存小就调大，保持 Batch*Accumulation ~= 16/32)
    "learning_rate": 2e-4,                 # 学习率 (LoRA通常 1e-4 ~ 5e-4)
    "num_train_epochs": 3,                 # 训练轮数
    "logging_steps": 10,                   # 打印日志频率
    "save_steps": 100,                     # 保存模型频率
    "save_total_limit": 2,                 # 最多保留几个 checkpoint
    "fp16": False,                         # 是否使用 float16
    "bf16": True,                          # 是否使用 bfloat16 (30系/40系显卡推荐 True)
    "dataloader_num_workers": 0,           # Windows设为0，Linux可设为4
    "group_by_length": True,               # 提高训练效率
    "report_to": "none",                   # 不上传wandb
}

# ================= LoRA 参数配置 =================
LORA_CONFIG = {
    "r": 16,                               # LoRA 秩 (Rank)，越大参数越多，显存需求越高
    "lora_alpha": 32,                      # 缩放系数，通常是 r 的 2 倍
    "lora_dropout": 0.1,                   # 防止过拟合
    "target_modules": [                    # 要微调的模块 (全量微调线性层效果最好)
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj"
    ],
    "task_type": "CAUSAL_LM",
}

# ================= 数据参数 =================
MAX_SEQ_LENGTH = 1024
# 单条数据的最大 token 长度

# ================= 系统提示词 (System Prompt) =================
# 把它放在配置里，保证训练和推理使用同一套人设
SYSTEM_PROMPT = (
    "你现在是XY.Lee的朋友，你的名字是小明。你是一个性格比较直率、有点暴躁、喜欢互损的男生。"
    "你喜欢玩怪物猎人、永劫无间和我的世界。"
    "请用小明的语气回复XY. Lee的消息。"
)





