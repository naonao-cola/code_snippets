from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters, types
import asyncio
import ollama
import json

client = ollama.Client(host='http://192.168.21.14:11434')  # 替换为你的服务器 IP 和端口

#Client会使用这里的配置来启动本地 MCP Server
server_params = StdioServerParameters(    
    command="python",     
    args=["./03mcp_server.py"],    
    env=None
    )
  

async def process_query(question, response, session):
    messages = [{"role": "user", "content": question}]
    available_tools = [{
        "type": "function",
        "function": {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.inputSchema
        }
        } for tool in response.tools]
    # print(available_tools)

    response = client.chat(
        model='qwen2.5:7b',
        messages=[{'role': 'user', 'content': str(question)}],
        tools=available_tools,
    )
    # print(response)
    content = response["message"]
    if dict(content)["tool_calls"]:
        tool_call = dict(dict(content["tool_calls"][0])["function"])
        tool_name = tool_call["name"]
        tool_args = tool_call["arguments"]

        result = await session.call_tool(tool_name, tool_args)

        # 将模型返回的调用哪个工具数据和工具执行完成后的数据都存入messages中
        messages.append({
            "role": "tool",
            "content": result.content[0].text,
            "tool_call_id": tool_call["name"],
            })
    
    # 将上面的结果再返回给大模型用于生产最终的结果
    response = client.chat(
        model='qwen2.5:7b',
        messages=[{'role': 'user', 'content': str(messages)}],
    )
    
    return response["message"]["content"]


async def main():           
    async with stdio_client(server_params) as (read, write):        
        async with ClientSession(read, write, sampling_callback=None) as session:                        
            await session.initialize()    
            response = await session.list_tools()
            # tools = response.tools
            # print('可用工具列表:', tools)  

            # res = await process_query("今天成都天气怎么样", response, session)
            # res = await process_query("今天天气怎么样", response, session)
            res = await process_query("今天是周几", response, session)
            print(res)

      
asyncio.run(main())