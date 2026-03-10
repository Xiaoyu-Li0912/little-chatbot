import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
import config  # 导入你的配置文件

def main():
    print(f"🔥 正在加载 Tokenizer: {config.MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1. 准备数据处理函数 (不依赖 SFTTrainer 的黑盒)
    def process_func(example):
        MAX_LENGTH = config.MAX_SEQ_LENGTH
        
        instruction = config.SYSTEM_PROMPT
        input_text = example['input']
        output_text = example['output']

        # A. 构建包含整个对话的消息
        full_messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": input_text},
            {"role": "assistant", "content": output_text}
        ]
        
        # B. 构建只包含“提问”部分的消息
        prompt_messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": input_text}
        ]

        # 1. 将全对话转为文本
        full_text = tokenizer.apply_chat_template(full_messages, tokenize=False)
        # 2. 将提问转为文本 (加上 add_generation_prompt=True 引导出 assistant)
        prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)

        # 3. 分别进行 Tokenize
        full_tokens = tokenizer(full_text, truncation=True, max_length=MAX_LENGTH)
        prompt_tokens = tokenizer(prompt_text, truncation=True, max_length=MAX_LENGTH)

        # 4. 构建 Labels (核心中的核心)
        input_ids = full_tokens["input_ids"]
        prompt_len = len(prompt_tokens["input_ids"])
        
        # 将前面的 prompt 部分全部遮罩为 -100
        labels = [-100] * prompt_len + input_ids[prompt_len:]
        
        # 为了防止截断导致的长度不一致，做一下对齐
        labels = labels[:MAX_LENGTH]

        return {
            "input_ids": input_ids,
            "attention_mask": full_tokens["attention_mask"],
            "labels": labels
        }

    # 2. 加载并处理数据集
    print(f"📂 正在加载数据: {config.DATA_PATH} ...")
    raw_dataset = load_dataset("json", data_files=config.DATA_PATH, split="train")
    
    print("⚙️  正在预处理数据 (Tokenizing)...")
    tokenized_dataset = raw_dataset.map(
        process_func,
        remove_columns=raw_dataset.column_names # 移除原始文本列，只保留 token IDs
    )

    # 3. 加载模型
    print("🧠 正在加载基座模型...")
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        torch_dtype=torch.bfloat16, # 4060 显卡完美支持 bf16
        device_map="auto",
        trust_remote_code=True
    )
    
    # 开启梯度检查点 (省显存神器)
    model.enable_input_require_grads()

    # 4. 配置 LoRA (显存不够必须用这个)
    print("🔧 配置 LoRA...")
    peft_config = LoraConfig(**config.LORA_CONFIG)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters() # 打印一下有多少参数会被训练

    # 5. 配置训练参数 (使用最基础的 TrainingArguments)
    args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        **config.TRAIN_ARGS
    )

    # 6. 初始化标准 Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    # 7. 开始训练
    print("🚀 开始训练! (Relax, this will work)")
    trainer.train()

    # 8. 保存
    print(f"💾 保存模型至: {config.OUTPUT_DIR}")
    trainer.model.save_pretrained(config.OUTPUT_DIR)
    tokenizer.save_pretrained(config.OUTPUT_DIR)

if __name__ == "__main__":

    main()
