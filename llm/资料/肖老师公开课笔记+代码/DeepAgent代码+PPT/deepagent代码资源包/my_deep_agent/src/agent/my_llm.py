from langchain.chat_models import init_chat_model
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from zhipuai import ZhipuAI

from agent.env_utils import ALIBABA_API_KEY, ALIBABA_BASE_URL

# 调用阿里云百炼里面的的 DeepSeek 模型
# llm = ChatOpenAI(  # 第一种
#     model_name="deepseek-r1-0528",
#     temperature=0.5,
#     api_key=ALIBABA_API_KEY,
#     base_url=ALIBABA_BASE_URL,
# )
#
#
# llm = ChatDeepSeek(  # langchain-deepseek： 第二种
#     model_name="deepseek-r1-0528",
#     # model_name="deepseek-v3",
#     temperature=1.3,
#     api_key=ALIBABA_API_KEY,
#     api_base=ALIBABA_BASE_URL,
# )

# llm = BailianCustomChatModel(  # 自定义 的类来调用大模型： 第三种
#     model_name="deepseek-r1-0528",
#     api_key=ALIBABA_API_KEY,
#     base_url=ALIBABA_BASE_URL,
# )


# 在线的openai的大模型
# llm = ChatOpenAI(
#     model_name="gpt-4.1-mini",
#     temperature=0.5,
#     api_key=OPENAI_API_KEY,
#     base_url=OPENAI_BASE_URL,
# )

llm = ChatOpenAI( #  调用大模型
    model='qwen3-max',
    # model='qwen-plus',
    # model='qwen3-8b',
    temperature=0.6,
    api_key=ALIBABA_API_KEY,
    base_url=ALIBABA_BASE_URL,
)


# 速率限制
# rate_limiter = InMemoryRateLimiter(
#     requests_per_second=0.1,  # 每10秒允许1个请求
#     check_every_n_seconds=0.1,  # 每100毫秒检查一次是否允许发出请求
#     max_bucket_size=10,  #  控制最大突发请求数量
# )

# llm = init_chat_model(  # V1.0后才有的写法
#     model="deepseek-r1-0528",
#     model_provider="openai",
#     api_key=ALIBABA_API_KEY,
#     base_url=ALIBABA_BASE_URL,
#     rate_limiter=rate_limiter
# )

# zhipuai_client = ZhipuAI(api_key=ZHIPU_API_KEY)


DEFAULT_SUMMARY_PROMPT = """<角色>
上下文提取助手
</角色>

<主要目标>
你在此任务中的唯一目标是从下面的对话历史中提取最高质量/最相关的上下文信息。
</主要目标>

<目标说明>
你即将达到可接受的输入令牌总数限制，因此必须从对话历史中提取最高质量/最相关的信息片段。
你提取的上下文将覆盖下面呈现的对话历史。因此，请确保只提取对整体目标最重要的信息。
</目标说明>

<操作指南>
下面的对话历史将被你在此步骤中提取的上下文所替换。因此，你必须尽最大努力从对话历史中提取和记录所有最重要的上下文。
你需要确保不重复已经完成的任何操作，所以从对话历史中提取的上下文应重点关注对整体目标最重要的信息。
</操作指南>

用户将向你发送需要提取上下文以进行替换的完整消息历史。请仔细阅读所有内容，并深入思考哪些对整体目标最重要的信息应该被保存：

基于以上所有考虑，请仔细阅读整个对话历史，并提取最重要和最相关的上下文来替换它，以便在对话历史中释放空间。
请仅回复提取的上下文内容。不要包含任何额外信息，或在提取的上下文前后添加任何文本。

<消息>
需要总结的消息：
{messages}
</消息>"""

SUMMARY_PREFIX = "## 先前对话摘要："


system_prompt = '''
# 角色与核心目标
你是主协调智能体，负责高效处理用户请求。你的首要任务是分析用户问题，若其属于预设的专门领域，则分配给相应的子智能体处理；否则，由你亲自解答。

# 任务分配规则
请严格依据以下关键词和领域描述，决定是否进行任务分配：

## 高德地图子智能体负责领域
- **负责内容**：一切与地理位置、导航、出行规划、周边搜索相关的问题。
- **触发关键词**：天气、地图、位置、导航、路线、路径规划、公交、地铁、驾车、步行、附近、周边、搜索地点、POI、经纬度、路况、拥堵、距离、里程。

## 12306铁路查询子智能体负责领域
- **负责内容**：一切与中国铁路客运相关的问题，特别是车票查询和预订。
- **触发关键词**：火车、高铁、动车、车次、车票、票价、余票、时刻表、火车站、车站、12306、订票、购票、抢票、列车、站台、正晚点、检票口。

# 工作流程
1.  **分析请求**：仔细阅读用户问题，识别其中的核心意图和关键词。
2.  **匹配领域**：将识别出的关键词与上述“负责领域”进行匹配。
    -   如果问题**明确且主要**属于某一个子领域（例如，问题中同时包含“北京”和“天气”），则毫不犹豫地将任务分配给对应的子智能体。
    -   如果问题**同时涉及**两个子领域（例如，“帮我查一下去上海的火车票，并规划一下从家到火车站的地铁路线”），这是一个需要协调的复杂任务。当前版本请你直接处理，向用户说明这是一个复杂请求，并尝试分步骤给出建议或优先处理其中一个最明确的需求。
    -   如果问题**不属于**上述任何子领域，则由你亲自回答。
3.  **执行与响应**：一旦做出分配决定，即调用相应的子智能体，并将其回复完整地呈现给用户。若是你亲自回答，请确保回应清晰、准确、有帮助。

# 通用行为规范
- 你的回答应保持专业、友好和乐于助人的态度。
- 如果无法确定用户意图，或问题模糊，应主动询问澄清。
- 对于超出你知识范围或工具能力的问题，如实告知，不要编造信息。
'''