import asyncio
import os
import sys
from pathlib import Path
from typing import AsyncIterator

from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from env_utils import ALIBABA_API_KEY, ALIBABA_BASE_URL


def get_python_executable():
    """获取当前Python解释器的完整路径"""
    python_exe = sys.executable
    print(f"当前Python解释器: {python_exe}")
    return python_exe


# 大模型
llm = ChatOpenAI(
    model_name="deepseek-v3.2",
    temperature=1.1,
    api_key=ALIBABA_API_KEY,
    base_url=ALIBABA_BASE_URL
)


async def main():
    SKILLS_ROOT = Path("./teaching_skills")
    workspace_dir = Path("./").absolute()


    checkpointer = InMemorySaver()

    # 本地沙箱
    backend = LocalShellBackend(
        root_dir=".",  # 将Agent的文件系统访问限制在当前目录下
        virtual_mode=True,  # 启用虚拟模式，规范化路径，阻止使用 `..` 和 `~` 等越界访问
        # 设置环境变量，包含编码相关的配置
        env={
            "PATH": f"{os.path.dirname(get_python_executable())};{os.environ.get('PATH', '')}",
            "PYTHONPATH": str(workspace_dir),
            "SYSTEMROOT": os.environ.get("SYSTEMROOT", "C:\\Windows"),
        },
    )

    agent = create_deep_agent(
        # backend=backend,
        model=llm,
        skills=[str(SKILLS_ROOT)],
        # tools=[]
        checkpointer=checkpointer,
        system_prompt=f'请尽量调用skills包来回答用户的问题，并且一定要遵循skills包中的md文件的说明描述。如果找不到对应的的skill，请自行回答，并注明：没有找到对应的Skill技能包',
    )

    print('Agent 创建成功')

    # 与Agent进行交互
    thread_id = "demo_thread_01"
    async for response in stream_agent_interaction_corrected(agent, thread_id):
        print(response, end="", flush=True)


async def stream_agent_interaction_corrected(agent, thread_id: str) -> AsyncIterator[str]:
    """
    使用官方推荐的 `agent.stream()` 方法进行流式交互。
    根据调试信息，chunk的结构是 (AIMessageChunk, metadata_dict)
    """
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        try:
            user_input = input("\n\n[用户] >>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n对话结束。")
            break

        if user_input.lower() in ('quit', 'exit', '退出', 'q'):
            print("再见！")
            break
        if not user_input:
            continue

        print("\n[助手] ", end="", flush=True)

        # 准备输入
        inputs = {"messages": [{"role": "user", "content": user_input}]}

        # 关键修正：使用 agent.stream() 并设置 stream_mode
        stream = agent.stream(inputs, config=config, stream_mode="messages", subgraphs=False)

        full_response = ""
        try:
            # 使用同步 for 循环，因为 agent.stream() 返回的是同步生成器
            for chunk in stream:  # 移除 async
                # 根据调试信息，chunk 的结构是 (AIMessageChunk, metadata_dict)
                if isinstance(chunk, tuple) and len(chunk) == 2:
                    token, metadata = chunk

                    # 1. 流式输出 AI 生成的文本内容
                    if hasattr(token, 'content') and token.content is not None:
                        content_str = str(token.content)
                        if content_str:
                            yield content_str  # 生成器
                            full_response += content_str

                    # 2. 捕获并显示工具调用开始
                    if hasattr(token, 'tool_call_chunks') and token.tool_call_chunks:
                        for tool_chunk in token.tool_call_chunks:
                            if tool_chunk and hasattr(tool_chunk, 'get'):
                                if tool_chunk.get('name'):
                                    tool_name = tool_chunk['name']
                                    yield f"\n[调用工具: {tool_name}]\n"

                    # 3. 捕获并显示工具调用结果
                    # 注意：工具调用结果通常不会出现在同一个 token 中
                    # 它们通常以独立的 token 形式出现

                else:
                    # 如果 chunk 不是预期的元组结构，打印调试信息
                    print(f"\n[调试] 意外的 chunk 结构: {type(chunk)}", file=sys.stderr)
                    continue

        except Exception as e:
            yield f"\n❌ Agent 执行出错: {e}\n"
            import traceback
            traceback.print_exc()
            continue

        # 流式结束后，将完整响应存入checkpoint，维持对话历史
        try:
            _ = await agent.ainvoke(
                {
                    "messages": [
                        {"role": "user", "content": user_input},
                        {"role": "assistant", "content": full_response}
                    ]
                },
                config=config
            )
        except Exception as e:
            print(f"\n[警告] 更新对话历史时出错: {e}", file=sys.stderr)


if __name__ == '__main__':
    asyncio.run(main())