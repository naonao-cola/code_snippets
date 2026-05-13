from langchain.embeddings import HuggingFaceBgeEmbeddings


#  pip install langchain-huggingface
from langchain_huggingface import HuggingFaceEmbeddings


model_name = "BAAI/bge-small-zh-v1.5"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity

#  第一次运行，会自动下载模型（去huggingface上下载），下载到hf默认的缓存目录。
#  可以通过修改环境变量：HF_HOME=指定目录

# bge_hf_embedding = HuggingFaceBgeEmbeddings(
#     model_name=model_name,
#     model_kwargs=model_kwargs,
#     encode_kwargs=encode_kwargs
# )

bge_hf_embedding = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)


resp = bge_hf_embedding.embed_documents(
    ['I like large language models.',
     '今天的天气非常不错！'
     ]
)

print(resp[0])
print(len(resp[0]))

