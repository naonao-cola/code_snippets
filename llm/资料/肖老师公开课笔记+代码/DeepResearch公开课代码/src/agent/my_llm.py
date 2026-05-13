# 大模型
from langchain_openai import ChatOpenAI

from agent.env_utils import ALIBABA_API_KEY, ALIBABA_BASE_URL

# llm = ChatOpenAI(
#     model="deepseek-v3.2",
#     temperature=1.1,
#     openai_api_key=ALIBABA_API_KEY,
#     openai_api_base=ALIBABA_BASE_URL,
# )

llm = ChatOpenAI(
    model="glm-5",
    temperature=1.0,
    openai_api_key=ALIBABA_API_KEY,
    openai_api_base=ALIBABA_BASE_URL,
)