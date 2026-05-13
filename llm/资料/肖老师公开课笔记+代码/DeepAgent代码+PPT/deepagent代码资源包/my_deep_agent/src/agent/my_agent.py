import asyncio

from deepagents import CompiledSubAgent, create_deep_agent
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_mcp_adapters.client import MultiServerMCPClient

from agent.mcp_tool_config import gaode_mcp_server_config, my12306_mcp_server_config, fenxi_mcp_server_config
from agent.my_llm import llm, system_prompt, DEFAULT_SUMMARY_PROMPT, SUMMARY_PREFIX

# 开发多步骤的，多子智能体的负责的一个Agent项目
# 路线规划，火车票查询，数据分析、爬取网页等各种能力的agent

# llm, tools, 提示词

# 创建一个mcp的客户端
mcp_client = MultiServerMCPClient({
    "gaode": gaode_mcp_server_config,
    "12306": my12306_mcp_server_config,
    "fenxi": fenxi_mcp_server_config,
})



async def create_my_agent():
    # 创建多个agent：这些都是子智能体。把这些子智能体合并
    gaode_tools = await mcp_client.get_tools(server_name="gaode")
    railway_tools = await mcp_client.get_tools(server_name="12306")
    fenxi_tools = await mcp_client.get_tools(server_name="fenxi")

    print(f'所有高德地图工具：{gaode_tools}')
    print(f'所有高德地图工具的数量：{len(gaode_tools)}')

    gaode_assistant = create_agent(  # 创建一个简单智能体（子智能体）
        model=llm,
        tools=gaode_tools,
        system_prompt='您是一位高德地图的子Agent，负责查询天气、地图信息和规划行程路线。',
        middleware=[
            SummarizationMiddleware(  # 内置的中间件（1、内置。2、内置的装饰器中间件（半自定义）3、全自定义）
                        summary_prompt=DEFAULT_SUMMARY_PROMPT,
                        summary_prefix=SUMMARY_PREFIX,
                        max_tokens_before_summary=4000,  # 触发的条件：超过这个数量，则进行总结
                        messages_to_keep=3,  # 保留的最近消息的数量
                        model=llm,
                    ),
        ]
    )

    # 创建了一个子智能体
    gaode_subagent = CompiledSubAgent(name='gaode_assistant',
                                      description='专门处理查询天气、地图信息和规划行程路线的智能体',
                                      runnable=gaode_assistant)

    railway_assistant = create_agent(  # 创建一个简单智能体（子智能体）
        model=llm,
        tools=railway_tools,
        system_prompt='您是一位12306铁路查询的子Agent，负责查询火车站、高铁站的信息和查询各种火车、高铁票。',
    )

    # 创建了一个子智能体
    railway_subagent = CompiledSubAgent(name='railway_assistant',
                                        description='专门查询火车站、高铁站的信息和查询各种火车、高铁票的智能体',
                                        runnable=railway_assistant)

    subagents = [gaode_subagent, railway_subagent]

    return create_deep_agent(model=llm, subagents=subagents, system_prompt=system_prompt)

deep_agent = asyncio.run(create_my_agent())

# 1、自己写一个界面，把agent运行起来
# 2、采用FastAPI（后端框架）把agent部署到后端服务里面，对外提供Restful的API接口