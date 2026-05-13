from mcp_server.mcp_tools import mcp_server

if __name__ == '__main__':
    mcp_server.run(
        transport="streamable-http",
        host="127.0.0.1",
        port=8000,
        path="/streamable",
        log_level="debug",
    )
