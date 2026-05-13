from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState, ToolRuntime
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.types import Command
from pydantic import BaseModel, Field
from typing_extensions import runtime

from agent.graph1.my_state import CustomState
from agent.llm.my_llm import llm


@tool
def get_user_name(runtime: ToolRuntime) -> Command:
    """获取用户的所有信息，包括：性别，年龄等"""
    user_name = runtime.config['configurable'].get('user_name', 'zs')
    print(f"调用工具， 传入的用户名是: {user_name}")
    # 模拟
    return Command(update={
        "username": user_name,  # 更新state中的用户名
        "messages": [
            ToolMessage(
                content=f'成功得到当前用户名：{user_name}',
                tool_call_id=runtime.tool_call_id,
            )
        ]
    })


# @tool
# def get_user_name(tool_call_id: Annotated[str, InjectedToolCallId], config: RunnableConfig) -> Command:
#     """获取用户的所有信息，包括：性别，年龄等"""
#     user_name = config['configurable'].get('user_name', 'zs')
#     print(f"调用工具， 传入的用户名是: {user_name}")
#     # 模拟
#     return Command(update={
#         "username": user_name,  # 更新state中的用户名
#         "messages": [
#             ToolMessage(
#                 content=f'成功得到当前用户名：{user_name}',
#                 tool_call_id=tool_call_id,
#             )
#         ]
#     })


@tool
def greet_user(runtime: ToolRuntime) -> str:
    """在获取用户的用户名后，给当前用户一个祝福语句"""
    print(f"调用祝福语工具")
    print(runtime.state)
    user_name = runtime.state.get('username')  # 从状态中获取用户名
    return f'祝贺你：{user_name}！'

#
# @tool
# def greet_user(state: Annotated[CustomState, InjectedState]) -> str:
#     """在获取用户的用户名后，给当前用户一个祝福语句"""
#     print(f"调用祝福语工具")
#     user_name = state['username']  # 从状态中获取用户名
#     return f'祝贺你：{user_name}！'


@tool
def get_user_info_by_name(config: RunnableConfig) -> dict:
    """获取用户的所有信息，包括：性别，年龄等"""
    user_name = config['configurable'].get('user_name', 'zs')
    print(f"调用工具， 传入的用户名是: {user_name}")
    # 模拟
    return {'username': user_name, 'sex': '男', 'age': 18}

prompt = (  # 外层的模板
    PromptTemplate.from_template("帮我生成一个简短的，关于{topic}的报幕词。")
    + ", 要求： 1、内容搞笑一点；"
    + "2、输出的内容采用{language}。"
)


chain = prompt | llm | StrOutputParser()


class ToolArgs(BaseModel):
    topic: str = Field(description="报幕词的主题")
    language: str = Field(description="报幕词采用的语言")


runnable_tool = chain.as_tool(
    name='chain_tool',
    description='这是一个专门生成报幕词的工具',
    args_schema=ToolArgs,
)