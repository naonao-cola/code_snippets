from mcp.server.fastmcp import FastMCP

# 创建一个MCP服务器
mcp = FastMCP("test")

# 添加一个工具
@mcp.tool()
def get_weather(city: str) -> float:
    """
    获取天气信息。
    :param city: 城市名称（需使用英文，如 Beijing）
    :return: 天气数据字典；若出错返回包含 error 信息的字典
    """
    weather_data = {
        "city": city,
        "temperature": "20°C",
        "condition": "Clear Sky"
    }
    return weather_data
    
if __name__ == "__main__":    
  print("正在启动MCP服务器...")
  mcp.run(transport='stdio')
