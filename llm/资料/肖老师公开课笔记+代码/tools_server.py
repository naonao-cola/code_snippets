from fastmcp import FastMCP
from fastmcp.server.auth.providers.bearer import RSAKeyPair, BearerAuthProvider
from fastmcp.server.dependencies import get_access_token, AccessToken
from zhipuai import ZhipuAI
from fastmcp.prompts.prompt import PromptMessage, TextContent


zhipuai_client = ZhipuAI(api_key='0e49686a31e84a64b8b753e4e4c64a31.BS3xePvy3uV69Ndm')


# 1、 生成 RSA 密钥对
key_pair = RSAKeyPair.generate()

print(key_pair.public_key)

# 2. 配置认证提供方
auth = BearerAuthProvider(
    public_key=key_pair.public_key,  # 公钥用于校验签名
    issuer='https://www.laoxiao.com',  #  # 令牌签发方标识
    audience='my-dev-server'  # 服务商的一个标识
)

# 3、服务器，模拟生成一个token
token = key_pair.create_token(
    subject='dev_user',
    issuer='https://www.laoxiao.com',
    audience='my-dev-server',
    scopes=['laoxiao', 'admin'],
    expires_in_seconds=3600
)

print(f"Test token: {token}")

# server = FastMCP(name='laoxiao_mcp_server', auth=auth)  # 创建了一个MCP的服务器
server = FastMCP(name='laoxiao_mcp_server')  # 创建了一个MCP的服务器

#  tools， 提示词模板， 数据
@server.tool()
def say_hello(username: str) -> str:
    """给指定的用户打个招呼"""
    return f"Hello, {username}!, 你好，今天天气不错！"



@server.tool(name='zhipuai_search', description='')
def my_search(query: str) -> str:
    """搜索互联网上的内容,包括实时天气, 金融，技术文档等各种类型的公开数据"""
    try:
        print("执行我的Python中的工具，输入的参数为:", query)

        # 得到了验证通过之后的
        # access_token: AccessToken = get_access_token()
        # print(access_token)
        # if access_token:
        #     print("<整个token:>", access_token)
        #     print(access_token.scopes)
        # else:
        #     return '由于没有权限，所以不能搜索到任何内容！，请客户端传入有效token'

        response = zhipuai_client.web_search.web_search(
            search_engine="search_pro",
            search_query=query
        )
        # print(response)
        if response.search_result:
            return "\n\n".join([d.content for d in response.search_result])
        return '没有搜索到任何内容！'
    except Exception as e:
        print(e)
        return '没有搜索到任何内容！'


@server.prompt
def generate_code_request(language: str, task_description: str) -> PromptMessage:
    """生成代码编写请求的用户消息模板"""
    content = f"请用{language}编写一个实现以下功能的函数：{task_description}"
    return PromptMessage(
        role="user",
        content=TextContent(type="text", text=content)
    )


# 结构化资源：自动序列化字典为JSON
@server.resource("resource://config")
def get_config() -> dict:
    """以JSON格式返回应用配置"""
    return {
        "theme": "dark",          # 界面主题配置
        "version": "1.2.0",       # 当前版本号
        "features": ["tools", "resources"],  # 已启用的功能模块
    }



if __name__ == '__main__':
    server.run(
        transport='streamable-http',
        host='127.0.0.1',
        port=8080,
        path='/streamable'
    )