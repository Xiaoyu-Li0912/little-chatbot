import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

# 1. 初始化 ChromaDB (创建一个本地文件夹存数据库)
client = chromadb.PersistentClient(path="./data/chroma_db")

# 2. 设置嵌入模型 (BGE-M3 是非常优秀的中文模型)
# 它会自动下载模型，第一次运行会稍慢
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-m3" 
)

collection = client.get_or_create_collection(
    name="xiaoming_memory", 
    embedding_function=embedding_func
)

# 3. 读取 CSV 数据
df = pd.read_csv('data/chat.csv', encoding='utf-8')
# 只处理说话内容，且排除掉图片/语音等无意义信息
df = df[df['type_name'] == 'text'].dropna(subset=['msg'])

# 4. 写入 ChromaDB
documents = df['msg'].tolist()
metadatas = [{"role": row['talker'], "time": str(row['CreateTime'])} for _, row in df.iterrows()]
ids = [str(i) for i in range(len(df))]

print(f"正在将 {len(documents)} 条聊天记录存入向量数据库...")
collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids
)
print("✅ 向量数据库构建完毕！")