from typing import Annotated, Literal

from langchain_core.tools import tool, InjectedToolArg
from zai import ZhipuAiClient

from agent.env_utils import ZHIPU_API_KEY



client = ZhipuAiClient(api_key=ZHIPU_API_KEY)



@tool(parse_docstring=True)
def zhipu_web_search(
        query: str,
        max_results: Annotated[int, InjectedToolArg] = 1,
        topic: Annotated[
            Literal["general", "news", "finance"], InjectedToolArg
        ] = "general",
) -> str:
    """基于给定的查询在互联网上搜索信息。

    使用智谱AI的联网搜索功能，直接返回网页的完整文本内容。

    Args:
        query: 要执行的搜索查询
        max_results: 最多返回的结果数量 (默认: 1)
        topic: 主题筛选 - 'general'（综合）, 'news'（新闻）, 或 'finance'（财经） (默认: 'general')

    Returns:
        包含完整网页内容的格式化搜索结果
    """
    # 根据topic参数设置搜索域名过滤
    search_domain_filter = ""
    if topic == "news":
        # 示例：可以指定搜索新闻类网站
        search_domain_filter = "news.sina.com.cn,news.163.com,news.sohu.com"
    elif topic == "finance":
        # 示例：可以指定搜索财经类网站
        search_domain_filter = "finance.sina.com.cn,eastmoney.com,wallstreetcn.com"

    try:
        # 使用智谱AI进行联网搜索
        search_response = client.web_search.web_search(
            search_engine="search_pro",
            search_query=query,
            count=max_results,  # 返回结果的条数
            search_domain_filter=search_domain_filter,  # 只访问指定域名的内容
            search_recency_filter="noLimit",  # 搜索指定日期范围内的内容
            content_size="high"  # 控制网页摘要的字数，使用high获取完整内容
        )

        # 提取搜索结果
        result_texts = []

        # 检查响应结构
        if hasattr(search_response, 'search_result') and search_response.search_result:
            results_list = search_response.search_result
        elif isinstance(search_response, dict) and 'search_result' in search_response:
            results_list = search_response.get("search_result", [])
        else:
            # 如果响应结构不同，尝试其他常见字段
            results_list = search_response.get("results", []) if isinstance(search_response, dict) else []

        # 处理每个搜索结果
        for idx, result in enumerate(results_list[:max_results]):
            # 提取标题、URL和内容
            if hasattr(result, 'title'):
                title = result.title
            elif isinstance(result, dict):
                title = result.get('title', f'结果 {idx + 1}')
            else:
                title = f'结果 {idx + 1}'

            if hasattr(result, 'url'):
                url = result.url
            elif isinstance(result, dict):
                url = result.get('url', '')
            else:
                url = ''

            if hasattr(result, 'content'):
                content = result.content
            elif isinstance(result, dict):
                content = result.get('content', '')
            else:
                content = '没有获取到内容'

            # 格式化结果
            result_text = f"""## {title}
**链接:** {url if url else '无可用链接'}

**内容摘要:**
{content}

---
"""
            result_texts.append(result_text)

        # 格式化最终响应
        if result_texts:
            response = f"""🔍 为查询 "{query}" 找到 {len(result_texts)} 个结果:

{chr(10).join(result_texts)}"""
        else:
            response = f"🔍 未找到与查询 '{query}' 相关的结果。请尝试调整搜索词。"

        return response

    except Exception as e:
        return f"搜索过程中发生错误: {str(e)}"


@tool(parse_docstring=True)
def think_tool(reflection: str) -> str:
    """用于对研究进展和决策进行战略性思考的工具。

    每次搜索后使用此工具，以系统性地分析结果并计划后续步骤。
    这在研究工作流程中创建了一个用于高质量决策的刻意停顿。

    使用时机：
    - 收到搜索结果后：我找到了哪些关键信息？
    - 决定下一步之前：我掌握的信息足够进行全面回答了吗？
    - 评估研究缺口时：我还缺少哪些具体信息？
    - 结束研究之前：我现在能提供一个完整的答案了吗？

    思考内容应包含：
    1. 对当前发现的分析——我收集到了哪些具体信息？
    2. 缺口评估——仍然缺失哪些关键信息？
    3. 质量评估——我是否有足够的证据/例子来给出一个好答案？
    4. 战略决策——我应该继续搜索还是现在提供答案？

    Args:
        reflection: 您对研究进展、发现、缺口和后续步骤的详细思考

    Returns:
        确认思考已记录，用于决策
    """
    return f"思考已记录: {reflection}"
