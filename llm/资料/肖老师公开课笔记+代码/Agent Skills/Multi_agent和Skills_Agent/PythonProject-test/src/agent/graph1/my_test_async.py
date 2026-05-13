from langgraph_sdk import get_client
import asyncio

client = get_client(url="http://localhost:2024")


# # 调用智能体发布的API接口
# async def main():
#     async for chunk in client.runs.stream(
#         None,  # Threadless run
#         "agent",  # Name of assistant. Defined in langgraph.json.
#         input={
#         "messages": [{
#             "role": "human",
#             "content": "给当前用户一个祝福语?",
#             }],
#         },
#         # stream_mode="messages-tuple",
#         config={"configurable": {"user_name": "老杨"}}
#     ):
#         print(f"Receiving new event of type: {chunk.event}...")
#         print(chunk.data)
#         print("\n\n")
#         # if isinstance(chunk.data, list) and 'type' in chunk.data[0] and chunk.data[0]['type'] == 'AIMessageChunk':
#         #     print(chunk.data[0]['content'], end="|")
#
# if __name__ == '__main__':
#     asyncio.run(main())


async def main():
    async for chunk in client.runs.stream(
        None,  # Threadless run
        "agent",  # Name of assistant. Defined in langgraph.json.
        input={
        "messages": [{
            "role": "human",
            "content": "告诉我当前用户的年龄？",
            }],
        },
        # stream_mode="messages-tuple",
        config={"configurable": {"user_name": "张三"}}
    ):
        print(f"Receiving new event of type: {chunk.event}...")
        print(chunk.data)
        print("\n\n")


if __name__ == '__main__':
    asyncio.run(main())