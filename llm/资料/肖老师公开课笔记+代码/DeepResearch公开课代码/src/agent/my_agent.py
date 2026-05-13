import os
from datetime import datetime

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

from agent.my_llm import llm
from agent.my_tools import zhipu_web_search, think_tool
from agent.prompt import RESEARCH_WORKFLOW_INSTRUCTIONS, SUBAGENT_DELEGATION_INSTRUCTIONS, RESEARCHER_INSTRUCTIONS

# 限制设置
max_concurrent_research_units = 3
max_researcher_iterations = 3

# 获取当前日期
current_date = datetime.now().strftime("%Y-%m-%d")
print(f"[系统] 当前系统日期: {current_date}")

# 提示词，用于主Agent（注意；RESEARCHER_INSTRUCTIONS 仅用于子智能体）
INSTRUCTIONS = (
    RESEARCH_WORKFLOW_INSTRUCTIONS
    + "\n\n"
    + "=" * 80
    + "\n\n"
    + SUBAGENT_DELEGATION_INSTRUCTIONS.format(
        max_concurrent_research_units=max_concurrent_research_units,
        max_researcher_iterations=max_researcher_iterations,
    )
)


# 创建一个临时目录作为代理的“沙盒”
temp_workspace = "./agent_workspace"
os.makedirs(temp_workspace, exist_ok=True)

# 创建主Agent
agent = create_deep_agent(
    model=llm,
    tools=[zhipu_web_search, think_tool],
    system_prompt=INSTRUCTIONS,
    backend=FilesystemBackend(
        root_dir=temp_workspace,
        virtual_mode=True
    ),
    subagents=[
        {
            "name": "sub_research",
            "description": "将研究任务委派给子研究智能体。一次只给这个研究员一个主题。",
            "system_prompt": RESEARCHER_INSTRUCTIONS.format(date=current_date),
            "tools": [zhipu_web_search, think_tool],
        }
    ]
)