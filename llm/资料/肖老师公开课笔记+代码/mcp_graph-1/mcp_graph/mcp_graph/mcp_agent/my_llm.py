from langchain_openai import ChatOpenAI
from zhipuai import ZhipuAI

from env_utils import DEEPSEEK_API_KEY, ZHIPU_API_KEY

zhipuai_client = ZhipuAI(api_key=ZHIPU_API_KEY)  # 填写您自己的APIKey

#
# llm = ChatOpenAI(  # zhipuai的
#     temperature=0,
#     model='glm-4-air-250414',
#     api_key=ZHIPU_API_KEY,
#     base_url="https://open.bigmodel.cn/api/paas/v4/")


# llm = ChatOpenAI(  # openai的
#     temperature=0,
#     model='gpt-4o-mini',
#     api_key=OPENAI_API_KEY,
#     base_url="https://xiaoai.plus/v1")


# llm = ChatOpenAI(
#     temperature=0.5,
#     model='deepseek-chat',
#     api_key=DEEPSEEK_API_KEY,
#     base_url="https://api.deepseek.com")


llm = ChatOpenAI(  # 私有化部署的大模型
    temperature=0,
    model="qwen3-8b",
    openai_api_key="XXXX",
    openai_api_base="http://localhost:6006/v1",
    extra_body={"chat_template_kwargs": {"enable_thinking": True}},
)