import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import gradio as gr
import config  # 依赖你的 config.py

print("="*50)
print("🚀 正在启动 Web 服务，请稍候...")
print("="*50)

# 1. 全局加载模型和 Tokenizer (只加载一次)
print(f"📦 正在加载基座模型: {config.MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    config.MODEL_ID,
    torch_dtype=torch.bfloat16 if config.TRAIN_ARGS.get("bf16") else torch.float16,
    device_map="auto"
)

print(f"🧩 正在挂载 LoRA 权重: {config.OUTPUT_DIR}")
model = PeftModel.from_pretrained(base_model, config.OUTPUT_DIR)
model.eval()

# 2. 定义核心的对话推理函数
def chat_with_model(message, history):
    """
    message: 当前用户输入的文字
    history: 之前的对话历史，格式为 [[用户话语1, AI话语1], [用户话语2, AI话语2], ...]
    """
    # 构建包含系统人设的初始消息
    messages = [{"role": "system", "content": config.SYSTEM_PROMPT}]
    
    # 拼接历史对话记录 (让模型有记忆)
    for user_msg, ai_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": ai_msg})
        
    # 加入当前最新的一句话
    messages.append({"role": "user", "content": message})

    # 将消息列表转换为模型需要的文本格式
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # 编码并送入 GPU
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # 模型生成回复
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            temperature=0.3,       # 保持较低的温度以维持逻辑
            top_p=0.85,
            repetition_penalty=1.1 # 防止复读机
        )

    # 截取新生成的部分并解码
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response

# 3. 构建 Gradio 网页界面
demo = gr.ChatInterface(
    fn=chat_with_model,              # 绑定我们上面写的对话函数
    title=f"🤖 {config.TARGET_NAME} 的 AI 数字分身", # 网页大标题
    description=f"我是用 {config.TARGET_NAME} 的真实聊天记录训练出来的微调大模型。\n你可以试着问我：'v我50' 或者 '去打永劫无间吗？'",
    theme="soft",                    # UI 主题
    retry_btn="🔄 重新生成",
    undo_btn="↩️ 撤销上一句",
    clear_btn="🗑️ 清空聊天记录",
)

if __name__ == "__main__":
    # 启动 Web 服务
    # server_name="0.0.0.0" 允许局域网内的其他设备访问
    # share=True 允许生成一个公网链接给外网同学访问
    demo.launch(server_name="0.0.0.0", 
                server_port=7860, 
                share=False,
                auth=("xiaoming", "xm123456"))
