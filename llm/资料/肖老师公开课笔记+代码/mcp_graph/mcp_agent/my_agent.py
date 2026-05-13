import asyncio
from contextlib import asynccontextmanager

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

from mcp_agent.my_llm import llm

# mcp_server_config = {
#     "url": "http://localhost:8000/sse",
#     "transport": "sse"
# }

mcp_server_config = {  # 连接MCP服务区的配置
    "url": "http://127.0.0.1:8000/streamable",
    "transport": "streamable_http"
}


@asynccontextmanager
async def make_agent():
    """生成一个智能体(langgraph)"""
    # 必须在异步环境下连接MCP服务端： with：自动的释放资源。
    async with MultiServerMCPClient({'lx_mcp': mcp_server_config}) as client:
        tools = client.get_tools()
        print(tools)
        # 创建：一个智能体
        agent = create_react_agent(llm, tools=tools)
        yield agent


async def main():
    """在异步环境下，创建智能体，并执行"""
    async with make_agent() as agent:
        # resp = await agent.ainvoke({'messages': '计算一下(3+6)的结果'})
        # resp = await agent.ainvoke({'messages': '计算一下(3 + 5) x 12的结果'})
        resp = await agent.ainvoke({'messages': '今天的北京的天气怎么样？'})
        print(resp.get('messages')[-1].content)


if __name__ == '__main__':
    asyncio.run(main())