import asyncio
import uuid
from typing import List, Dict

import gradio as gr
from gradio import ChatMessage
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from langgraph_swarm import create_handoff_tool, create_swarm

from my_llm import llm

# 配置MCP的地址而已，通信协议
# 高德地图MCP服务端（工具的配置）
gaode_mcp_server_config = {  # 高德地图MCP服务端 里面有各种高德给你提供公交、地铁、公交、驾车、步行、骑行、POI搜索、IP定位、逆地理编码、云图服务、云图审图、云图审
    "url": "https://mcp.amap.com/mcp?key=01467c30e6e6d2cca314e3b657de3fbd",
    "transport": "streamable_http",
}

# 网络爬虫MCP服务端（工具的配置）
# fetch_mcp_server_config = {
#     "url": "https://mcp.api-inference.modelscope.net/f59fe300817843/sse",
#     "transport": "sse",
# }

# 12306的MCP服务端（工具的配置）
my12306_mcp_server_config = {
    "url": "https://mcp.api-inference.modelscope.net/285e1bb9420d43/sse",
    "transport": "sse",
}



# 创建一个MCP的客户端得到这些工具
mcp_client = MultiServerMCPClient({
    "gaode_mcp_server": gaode_mcp_server_config,
    # "fetch_mcp_server": fetch_mcp_server_config,
    "my12306_mcp_server": my12306_mcp_server_config,
})

transfer_to_gaode_assistant = create_handoff_tool(
    agent_name="gaode_assistant",
    description="将用户转接到高德地图智能助手。",
)
transfer_to_railway_assistant = create_handoff_tool(
    agent_name="railway_assistant",
    description="将用户转接到12306铁路查询智能助手。",
)


async def create_agent():
    gaode_mcp_tools = await mcp_client.get_tools(server_name='gaode_mcp_server')
    railway_mcp_tools = await mcp_client.get_tools(server_name='my12306_mcp_server')
    print(f'所有的高德地图工具: {gaode_mcp_tools}')
    print(f'所有的12306工具: {railway_mcp_tools}')

    gaode_assistant = create_react_agent(
        model=llm,
        tools=[*gaode_mcp_tools, transfer_to_railway_assistant],
        prompt="您是一位高德地图智能助手，负责查询天气、地图信息和规划行程路线。",
        name="gaode_assistant"
    )


    railway_assistant = create_react_agent(
        model=llm,
        tools=[*railway_mcp_tools, transfer_to_gaode_assistant],
        prompt="您是一位12306铁路查询智能助手，负责查询火车站、高铁站的信息和查询各种火车、高铁票。",
        name="railway_assistant"
    )

    swarm = create_swarm(
        agents=[gaode_assistant, railway_assistant],
        default_active_agent="railway_assistant"
    ).compile(checkpointer=InMemorySaver())
    return swarm


agent = asyncio.run(create_agent())

# 执行智能体
config = {
    "configurable": {
        # 检查点由session_id访问
        "thread_id": str(uuid.uuid4()),  # 可以用 用户名作为会话
    }
}

# res = agent.invoke(
#     input={'messages': [HumanMessage(content='你好！')]},
#     config=config
# )
#
# print(res)

# 开发一个界面：输入一个任务，然后输出一个结果
def add_message(chat_history, user_message):
    if user_message:
        chat_history.append({"role": "user", "content": user_message})
    return chat_history, gr.Textbox(value=None, interactive=False)


async def submit_messages(chat_history: List[Dict]):
    """流式处理消息的核心函数"""
    user_input = chat_history[-1]['content']
    current_state = agent.get_state(config)
    full_response = ""  # 累积完整响应
    tool_calls = []  # 记录工具调用

    # 处理中断恢复或正常消息
    inputs = Command(resume={'answer': user_input}) if current_state.next else {
        'messages': [HumanMessage(content=user_input)]}

    async for chunk in agent.astream(
            inputs,
            config,
            stream_mode=["messages", "updates"],  # 同时监听消息和状态更新
    ):
        if 'messages' in chunk:
            for message in chunk[1]:
                if hasattr(message, "pretty_repr"):
                    msg_repr = message.pretty_repr(html=True)
                    print(msg_repr)  # 输出消息的表示形式
                else:
                    print(message)
                # 处理AI消息流式输出
                if isinstance(message, AIMessage) and message.content:
                    full_response += message.content
                    # 更新最后一条消息而非追加
                    if chat_history and isinstance(chat_history[-1], ChatMessage) and 'title' not in chat_history[-1].metadata:
                        chat_history[-1].content = full_response
                    else:
                        chat_history.append(ChatMessage(role="assistant", content=message.content))
                    yield chat_history

                # 处理工具调用消息
                elif isinstance(message, ToolMessage):
                    tool_msg = f"🔧 调用工具: {message.name}\n{message.content}"
                    chat_history.append(ChatMessage(role="assistant", content=tool_msg,
                                        metadata={"title": f"🛠️ Used tool {message.name}"}))
                    yield chat_history

    # 检查新中断
    current_state = agent.get_state(config)
    if current_state.next:
        interrupt_msg = current_state.interrupts[0].value
        # chat_history.append({'role': 'assistant', 'content': interrupt_msg})
        chat_history.append(ChatMessage(role="assistant", content=interrupt_msg))
        yield chat_history


# 创建Gradio界面
with gr.Blocks(
        title='我的智能小秘书',
        theme=gr.themes.Soft(),
        css=".system {color: #666; font-style: italic;}"  # 自定义系统消息样式
) as demo:
    # 聊天历史记录组件
    chatbot = gr.Chatbot(
        type="messages",
        height=500,
        render_markdown=True,  # 支持Markdown格式
        line_breaks=False  # 禁用自动换行符
    )

    # 输入组件
    chat_input = gr.Textbox(
        placeholder="请输入您的消息...",
        label="用户输入",
        max_lines=5,
        container=False
    )

    # 控制按钮
    with gr.Row():
        submit_btn = gr.Button("发送", variant="primary")
        clear_btn = gr.Button("清空对话")

    # 消息提交处理链
    msg_handler = chat_input.submit(
        fn=add_message,
        inputs=[chatbot, chat_input],
        outputs=[chatbot, chat_input],
        queue=False
    ).then(
        fn=submit_messages,
        inputs=chatbot,
        outputs=chatbot,
        api_name="chat_stream"  # API端点名称
    )

    # 按钮点击处理链
    btn_handler = submit_btn.click(
        fn=add_message,
        inputs=[chatbot, chat_input],
        outputs=[chatbot, chat_input],
        queue=False
    ).then(
        fn=submit_messages,
        inputs=chatbot,
        outputs=chatbot
    )

    # 清空对话
    clear_btn.click(
        fn=lambda: [],
        inputs=None,
        outputs=chatbot,
        queue=False
    )

    # 重置输入框状态
    msg_handler.then(
        lambda: gr.Textbox(interactive=True),
        None,
        [chat_input]
    )
    btn_handler.then(
        lambda: gr.Textbox(interactive=True),
        None,
        [chat_input]
    )

if __name__ == '__main__':
    demo.launch()