import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import config  # 导入配置，复用里面的路径和System Prompt

def main():
    print("=" * 50)
    print(f"🧪 正在加载原始基座模型: {config.MODEL_ID}")
    print("⚠️  注意：此模式不加载任何微调权重，仅测试模型原生能力")
    print("=" * 50)

    # 1. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID, trust_remote_code=True)

    # 2. 加载基座模型 (不加载 LoRA/Peft)
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        torch_dtype=torch.bfloat16, # 4060 显卡用这个精度最高且快
        device_map="auto",
        trust_remote_code=True
    )
    model.eval() # 开启评估模式

    # 3. 准备 System Prompt
    # 我们故意使用和训练时一样的人设，看看没训练过的模型能不能演出来
    system_prompt = config.SYSTEM_PROMPT
    
    print("\n✅ 模型加载完毕！")
    print(f"🎭 当前人设指令: {system_prompt}")
    print("-" * 50)

    # 4. 对话循环
    while True:
        user_input = input(f"\n{config.USER_NAME}: ")
        if user_input.lower() in ["exit", "quit", "退出"]:
            break

        # 构建对话消息
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]

        # 模板处理
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 编码并移动到 GPU
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # 生成
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=512,
                temperature=0.7,  # 稍微有一点随机性
                top_p=0.9,
                repetition_penalty=1.1
            )

        # 解码
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        print(f"{config.TARGET_NAME} (原始版): {response}")

if __name__ == "__main__":
    main()