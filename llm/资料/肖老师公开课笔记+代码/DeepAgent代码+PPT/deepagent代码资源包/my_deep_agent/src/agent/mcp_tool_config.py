


gaode_mcp_server_config = {  # 高德地图MCP服务端 里面有各种高德给你提供公交、地铁、公交、驾车、步行、骑行、POI搜索、IP定位、逆地理编码、云图服务、云图审图、云图审
    "url": "https://mcp.amap.com/mcp?key=01467c30e6e6d2cca314e3b657de3fbd",
    "transport": "streamable_http",
}


# 12306的MCP服务端（工具的配置）
my12306_mcp_server_config = {
    "url": "https://mcp.api-inference.modelscope.net/e1d8f8ede32546/mcp",
    "transport": "streamable_http",
}


# 数据分析报表的MCP服务端（工具的配置）
fenxi_mcp_server_config = {
    "url": "https://mcp.api-inference.modelscope.net/66c1d8f84b814a/sse",
    "transport": "sse",
}