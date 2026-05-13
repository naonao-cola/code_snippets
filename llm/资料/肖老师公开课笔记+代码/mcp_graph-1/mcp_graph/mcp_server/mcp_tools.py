from fastmcp import FastMCP

from mcp_agent.my_llm import zhipuai_client

mcp_server = FastMCP(name='lx-mcp', instructions='我自己的MCP服务')


@mcp_server.tool('my_search_tool', description='专门搜索互联网中的内容')
def my_search(query: str) -> str:
    """搜索互联网上的内容"""
    try:
        response = zhipuai_client.web_search.web_search(
            search_engine="search-std",
            search_query=query
        )
        # print(response)
        if response.search_result:
            return "\n\n".join([d.content for d in response.search_result])
        return '没有搜索到任何内容！'
    except Exception as e:
        print(e)
        return '没有搜索到任何内容！'


@mcp_server.tool()
def add(a: int, b: int) -> int:
    """加法运算: 计算两个数字相加"""
    return a + b


@mcp_server.tool()
def multiply(a: int, b: int) -> int:
    """乘法运算：计算两个数字相乘"""
    return a * b
