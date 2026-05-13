from mcp_server.mcp_tools import mcp_server

if __name__ == '__main__':
    mcp_server.run(
        transport="sse",
        host="127.0.0.1",
        port=8000,
        log_level="debug",
        path="/sse",
    )
