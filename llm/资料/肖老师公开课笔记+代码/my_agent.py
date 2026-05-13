import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from env_utils import ZHIPU_API_KEY

# python -m vllm.entrypoints.openai.api_server
# --model /root/autodl-tmp/models/Qwen/Qwen3-8B
# --served-model-name qwen3-8b
# --max-model-len 8k
# --host 0.0.0.0
# --port 6006
# --dtype bfloat16   --gpu-memory-utilization 0.8   --enable-auto-tool-choice   --tool-call-parser hermes
llm = ChatOpenAI(
    model='qwen3-8b',
    api_key='xxxxxxxx',
    base_url='http://localhost:6006/v1',
    extra_body={'chat_template_kwargs': {'enable_thinking': False}},
)


# 连接MCP的配置
java_mcp_server_config = {  # 调用JAVA  MCP服务器中的各种工具，
    "url": "http://127.0.0.1:8086/sse",  # sse是数据通信的机制
    "transport": "sse"
}

python_mcp_server_config = {  # 连接Python  MCP服务器的配置
    "url": "http://127.0.0.1:8000/streamable",
    "transport": "streamable_http"  # MCP的通信机制
}

zhipu_mcp_server_config = {  # 连接 智普的MCP服务端配置
    "url": "https://open.bigmodel.cn/api/mcp/web_search/sse?Authorization="+ZHIPU_API_KEY,
    "transport": "sse"  # MCP的通信机制
}


client = MultiServerMCPClient(  # MCP的客户端
    {
        'java_tool': java_mcp_server_config,
        # 'python_tool': python_mcp_server_config,
        'zhipu_tool': zhipu_mcp_server_config,
    }
)


async def create_agent():
    """构建一个基于MCP的智能体"""
    tools = await client.get_tools()  # 拿到所有的MCP工具

    print(tools)

    # 创建智能体
    agent = create_react_agent(llm, tools)

    resp = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "计算一下(3 + 5) x 12的结果"}]}
    )
    print(resp.get("messages")[-1].content)

    resp2 = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "今天，上海的天气怎么样？"}]})
    print(resp2.get('messages')[-1].content)



if __name__ == '__main__':
    asyncio.run(create_agent())