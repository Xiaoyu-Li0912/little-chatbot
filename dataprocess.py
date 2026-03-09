import pandas as pd
import json
import re
from datetime import datetime, timedelta

# ================= 配置区域 =================
INPUT_FILE = r'data\chat.csv'
OUTPUT_FILE = r'data\finetune_v2.jsonl'

TARGET_NAME = '刘嘉岩'  # 模仿对象
USER_NAME = 'XY. Lee'   # 用户

# 【关键配置】时间阈值（单位：秒）
# 1. 合并阈值：同一个人在 180秒(3分钟) 内连续发的消息，合并为一句话
MERGE_INTERVAL = 600 
# 2. 回复阈值：如果刘嘉岩的回复距离你上一句话超过 600秒(10分钟)，
#    则认为他开启了新话题，不作为训练数据（防止错位匹配）
REPLY_INTERVAL = 6000

SYSTEM_PROMPT = (
    "你现在是刘嘉岩。你是一个性格直率、有点暴躁、喜欢互损的男生。"
    "你喜欢玩怪物猎人、永劫无间和我的世界，经常帮XY.Lee扫码登录游戏账号。"
    "你的口头禅包括'cnm'、'sb'、'OK?'、'。。。。'、'再发个'。"
    "请用刘嘉岩的语气回复XY. Lee的消息。"
)
# ===========================================

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.strip()
    
    # 过滤规则
    if text in ["[图片]", "[视频]", "[文件]", "[位置]"]: return ""
    if text.startswith("[语音通话]") or text.startswith("[视频通话]"): return ""
    if "撤回了一条消息" in text or "拍了拍" in text or "邀请您组队" in text: return ""
    if text.startswith("ⓘ"): return "" # 系统消息
    
    # 处理语音转文字保留内容
    if "[语音转文字]" in text:
        text = text.replace("[语音转文字]", "").strip()
    # 过滤失败的语音
    if "转文字失败" in text or "未知错误" in text: return ""
    
    # 过滤表情包代码
    if text.startswith("[动画表情]"): return ""
    
    # 特殊标记
    if "[转账]" in text: return "（发起转账）"
    if "[转账收款]" in text: return "（收款）"
    
    return text

def process_chat_with_time(file_path):
    # 1. 读取数据
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except:
        df = pd.read_csv(file_path, encoding='gbk')

    # 2. 时间解析
    # 你的CSV时间格式是 "2024-06-13T09:44:24.000Z"
    df['dt'] = pd.to_datetime(df['CreateTime'])
    
    # 3. 清洗文本
    df['clean_msg'] = df['msg'].apply(clean_text)
    df = df[df['clean_msg'] != ""] # 删掉空行
    
    if df.empty: return []

    # ================= 第一步：基于时间 + 说话人 合并消息 =================
    # 目标：生成一个 merged_talks 列表
    # 结构：[{'role': 'XY. Lee', 'content': '...', 'end_time': timestamp}, ...]
    
    merged_talks = []
    
    # 初始化缓冲
    buffer_role = df.iloc[0]['talker']
    buffer_msgs = [df.iloc[0]['clean_msg']]
    buffer_end_time = df.iloc[0]['dt']
    
    for i in range(1, len(df)):
        curr_row = df.iloc[i]
        curr_role = curr_row['talker']
        curr_msg = curr_row['clean_msg']
        curr_time = curr_row['dt']
        
        # 计算距离上一条消息的时间差（秒）
        time_diff = (curr_time - buffer_end_time).total_seconds()
        
        # 合并条件：同一个人 且 时间间隔在阈值内
        if curr_role == buffer_role and time_diff <= MERGE_INTERVAL:
            buffer_msgs.append(curr_msg)
            buffer_end_time = curr_time # 更新这一轮的结束时间
        else:
            # 结算上一轮
            merged_talks.append({
                "role": buffer_role,
                "content": "，".join(buffer_msgs),
                "start_time": buffer_end_time - timedelta(seconds=time_diff), # 估算开始时间，虽然用处不大
                "end_time": buffer_end_time
            })
            
            # 开启新一轮
            buffer_role = curr_role
            buffer_msgs = [curr_msg]
            buffer_end_time = curr_time
            
    # 加入最后一段
    merged_talks.append({
        "role": buffer_role,
        "content": "，".join(buffer_msgs),
        "end_time": buffer_end_time
    })

    # ================= 第二步：构建问答对 (QA Pairs) =================
    finetune_data = []
    
    for i in range(len(merged_talks) - 1):
        question_block = merged_talks[i]
        answer_block = merged_talks[i+1]
        
        # 核心逻辑检查：
        # 1. 必须是 XY. Lee 问，刘嘉岩 答
        if question_block['role'] != USER_NAME or answer_block['role'] != TARGET_NAME:
            continue
            
        # 2. 【重点】回复时间检查
        # 如果刘嘉岩回复的时间，距离你说完话的时间超过了 10分钟(REPLY_INTERVAL)
        # 说明这可能不是回复，而是他睡醒了发起的“新话题”
        # 这种情况下，你的上一句话不应该作为 Input
        time_gap = (answer_block['end_time'] - question_block['end_time']).total_seconds()
        
        if time_gap > REPLY_INTERVAL:
            # print(f"跳过对话（间隔过长 {int(time_gap/60)}分钟）：\n问：{question_block['content']}\n答：{answer_block['content']}")
            continue
            
        # 3. 构造数据
        finetune_data.append({
            "instruction": SYSTEM_PROMPT,
            "input": question_block['content'],
            "output": answer_block['content']
        })

    return finetune_data

# ================= 运行 =================
if __name__ == "__main__":
    print(f"正在处理 {INPUT_FILE}，启用时间感知逻辑...")
    data = process_chat_with_time(INPUT_FILE)
    
    print(f"处理完成！生成有效对话对：{len(data)} 条")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for entry in data:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')
            
    print(f"预览前 5 条数据：")
    for item in data[:5]:
        print(f"Q: {item['input']}")
        print(f"A: {item['output']}")

        print("-" * 30)
