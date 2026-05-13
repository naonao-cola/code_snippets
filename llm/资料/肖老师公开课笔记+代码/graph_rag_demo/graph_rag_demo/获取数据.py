import json
from typing import List

from langchain_community.document_loaders import WikipediaLoader, JSONLoader
from langchain_core.documents import Document

# 加载中文维基百科中关于"阿里巴巴"的文档，限制最多2篇
docs: List[Document] = WikipediaLoader(query="马云", lang="zh", load_max_docs=3).load()
# docs = WikipediaLoader(query="沃伦·巴菲特", lang="zh", load_max_docs=3).load()

# 查看加载的文档数量
print(f"加载的文档数量: {len(docs)}")


# 保存Document到JSON
def save_docs_to_json(docs, output_path):
    doc_dicts = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(doc_dicts, f, ensure_ascii=False, indent=4)

def metadata_func(record: dict, metadata: dict) -> dict:

    metadata["title"] = record.get("metadata").get("title")
    metadata["summary"] = record.get("metadata").get("summary")
    metadata["source"] = record.get("metadata").get("source")

    return metadata

# 从JSON加载Document
def load_docs_from_json(file_path):
    loader = JSONLoader(
        file_path=file_path,
        jq_schema=".[]",
        content_key="page_content",
        metadata_func=metadata_func,
        # text_content=False
    )
    return loader.load()


save_docs_to_json(docs, "alibaba_docs.json")
loaded_docs = load_docs_from_json("alibaba_docs.json")
#
# 查看第一篇文档的元数据（如标题、来源URL等）
print("元数据:", loaded_docs[0].metadata)

# 查看第一篇文档的前400个字符内容
print("内容预览:", loaded_docs[0].page_content[:400])