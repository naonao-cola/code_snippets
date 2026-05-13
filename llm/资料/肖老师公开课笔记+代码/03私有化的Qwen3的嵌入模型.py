


from sentence_transformers import SentenceTransformer


# 需要安装： pip install sentence-transformers
# Requires transformers>=4.51.0
# Requires sentence-transformers>=2.7.0
# Load the model


# model = SentenceTransformer("Qwen/Qwen3-Embedding-4B")
qwen3_embedding = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

resp = qwen3_embedding.encode(
    ['I like large language models.',
     '今天的天气非常不错！'
     ]
)

print(resp[0])
print(len(resp[0]))
