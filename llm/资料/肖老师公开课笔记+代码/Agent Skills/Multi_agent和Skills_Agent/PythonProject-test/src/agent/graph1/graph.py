from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt.chat_agent_executor import AgentState
from pydantic import BaseModel, Field

from agent.graph1.my_state import CustomState
from agent.graph1.tool_demo9 import get_user_name, greet_user, get_user_info_by_name, runnable_tool
from agent.llm.my_llm import llm


class Response(BaseModel):
    """Respond to the user with this"""

    username: str = Field(description="The name of user")



def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


# 动态系统提示词
@dynamic_prompt
def prompt(request: ModelRequest) -> str:
    user_name = request.runtime.context['user_name']            # config["configurable"].get('user_name', 'zs')
    system_message = f'你是一个智能助手, 只能调用工具回答问题, 当前用户的名字是: {user_name}'
    print(system_message)

    return system_message


graph = create_agent(
    llm,
    # tools=[get_user_info_by_name, greet_user],
    tools=[runnable_tool, get_user_name, greet_user, get_user_info_by_name],
    # middleware=[prompt],
    system_prompt='你是一个智能助手, 只能调用合适的工具后，再回答问题。',
    # state_schema=CustomState,
    # response_format=Response
)

# result = graph.stream(
#     input={"messages": [{"role": "user", "content": "计算一下（3+5）*12的结果?"}]},
#     config={"configurable": {"user_name": "老肖"}},
#     stream_mode='messages-tuple'
# )
#
# for chunk in result:
#     print(chunk)

# result = graph.stream(
#     input={
#         "messages": [{
#             "role": "user",
#             "content": "给当前用户一个祝福语?",
#         }],
#     },
#     config={"configurable": {"user_name": "老肖"}},
#     stream_mode='messages-tuple'
# )
#
# for chunk in result:
#     print(chunk)
