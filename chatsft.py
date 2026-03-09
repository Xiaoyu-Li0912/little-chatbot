import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import config  # 导入配置

def main():
    print(f"正在加载基座模型: {config.MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        torch_dtype=torch.bfloat16 if config.TRAIN_ARGS.get("bf16") else torch.float16,
        device_map="auto"
    )

    print(f"正在加载 LoRA 权重: {config.OUTPUT_DIR} ...")
    try:
        model = PeftModel.from_pretrained(base_model, config.OUTPUT_DIR)
    except Exception as e:
        print(f"加载 LoRA 失败，请检查路径。错误: {e}")
        return

    model.eval()

    print("="*50)
    print("刘嘉岩 (AI Clone) 已上线！输入 'exit' 退出")
    print("="*50)

    while True:
        user_input = input(f"\n{config.USER_NAME}: ") # 使用 config 里的名字
        if user_input.lower() in ["exit", "quit", "退出"]:
            break

        # 使用 config 里的 System Prompt，确保和训练时一致
        messages = [
            {"role": "system", "content": config.SYSTEM_PROMPT},
            {"role": "user", "content": user_input}
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1
            )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        print(f"{config.TARGET_NAME}: {response}") # 使用 config 里的名字

if __name__ == "__main__":
    # 为了让 chat.py 也能用到 config.py 里的 USER_NAME 和 TARGET_NAME
    # 你可能需要在 config.py 里补上这两个变量定义：
    # USER_NAME = 'XY. Lee'
    # TARGET_NAME = '刘嘉岩'
    main()