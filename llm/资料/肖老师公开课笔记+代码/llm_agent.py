import asyncio
import uuid
from typing import List, Dict
import gradio as gr
from gradio import ChatMessage
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command

from my_agent.env_utils import ZHIPU_API_KEY
from my_agent.my_llm import llm

# 配置MCP的地址而已，通信协议

# 高德地图MCP服务端（工具的配置）
gaode_mcp_server_config = {
    "url": "https://mcp.amap.com/mcp?key=01467c30e6e6d2cca314e3b657de3fbd",
    "transport": "streamable_http",
}

# 网络搜索MCP服务端（工具的配置）
zhipuai_mcp_server_config = {
    "url": "https://open.bigmodel.cn/api/mcp/web_search/sse?Authorization="+ZHIPU_API_KEY,
    "transport": "sse",
}


# 网络爬虫MCP服务端（工具的配置）
fetch_mcp_server_config = {
    "url": "https://mcp.api-inference.modelscope.net/566d884334dd48/sse",
    "transport": "sse",
}

# 12306的MCP服务端（工具的配置）
my12306_mcp_server_config = {
    "url": "https://mcp.api-inference.modelscope.net/351e1101f6fc43/sse",
    "transport": "sse",
}

# 数据分析的MCP服务端（工具的配置）
chart_mcp_server_config = {
    "url": "https://mcp.api-inference.modelscope.net/037a9fc870d14a/sse",
    "transport": "sse",
}

# 创建MCP客户端
mcp_client = MultiServerMCPClient(
    {
        "zhipuai_mcp_server_config": zhipuai_mcp_server_config,
        "gaode_mcp_server_config": gaode_mcp_server_config,
        "fetch_mcp_server_config": fetch_mcp_server_config,
        "my12306_mcp_server_config": my12306_mcp_server_config,
        "chart_mcp_server_config": chart_mcp_server_config,
    }
)

# 开发智能体
async def create_agent():
    # 获取MCP的所有tools
    mcp_tools = await mcp_client.get_tools()

    # print(len(mcp_tools))  # 打印tools的个数 : 55个
    # print(mcp_tools[-2:]) # 打印最后两个tools
    return create_react_agent(
        llm,
        tools=mcp_tools,
        prompt='你是一个智能助手，尽可能的调用工具回答用户问题',
        checkpointer=InMemorySaver()  # 创建一个内存的保存器： 保存对话上下文
    )


agent = asyncio.run(create_agent())
# 配置参数，包含会话ID
config = {
    "configurable": {
        # 检查点由session_id访问
        "thread_id": str(uuid.uuid4()),
    }
}



# res = agent.invoke(input={'messages': [HumanMessage(content='你好！')]}, config=config)
# print(res)

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