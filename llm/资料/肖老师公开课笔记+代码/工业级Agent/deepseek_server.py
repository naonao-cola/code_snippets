import time
from asyncio.log import logger
import re
import sys
import uvicorn
import gc
import json
import torch
import random
import string
from vllm import SamplingParams, AsyncEngineArgs, AsyncLLMEngine
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import List, Literal, Optional, Union
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, LogitsProcessor
from sse_starlette.sse import EventSourceResponse

# 常量：fastapi服务器的监控（心跳机制）
# 设置服务器发送事件的心跳间隔： 设置为 1000 毫秒（1 秒）
EventSourceResponse.DEFAULT_PING_INTERVAL = 1000

# AI大模型的上下文最大长度
MAX_MODEL_LENGTH = 8192 

# fastapi的装饰器
# 异步生命周期管理（app），异步上下文管理器在进入和退出上下文时可以执行异步操作。
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield  #  表示生命周期的中间阶段，即应用的运行期间。

    if torch.cuda.is_available():  # 表示生命周期 退出时候自动执行的代码
        torch.cuda.empty_cache()  #  清空 GPU 缓存，释放不再使用的内存。
        torch.cuda.ipc_collect()  # 收集并释放进程间通信（IPC）资源，进一步清理内存



# 创建fastAPI的app对象
app = FastAPI(lifespan=lifespan)

# 设置fastAPI的跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 自动生成一个唯一的id函数
def generate_id(prefix: str, k=29) -> str:
    """
    随机生成一个29长度的字符串，这个字符串有大，小写字母和数字组成
    prefix是ID的前缀，
    k 是 id的长度
    """
    suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=k))
    return f"{prefix}{suffix}"


class ModelCard(BaseModel):
    """
    povo类： Web开发中的数据模型类。格式化请求或者是响应的数据
    """
    id: str = ""
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "owner"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = ["glm-4"]


class FunctionCall(BaseModel):
    name: Optional[str] = None
    arguments: Optional[str] = None


class ChoiceDeltaToolCallFunction(BaseModel):
    name: Optional[str] = None
    arguments: Optional[str] = None


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


class ChatCompletionMessageToolCall(BaseModel):
    index: Optional[int] = 0
    id: Optional[str] = None
    function: FunctionCall
    type: Optional[Literal["function"]] = 'function'


class ChatMessage(BaseModel):
    # “function” 字段解释：
    # 使用较老的OpenAI API版本需要注意在这里添加 function 字段并在 process_messages函数中添加相应角色转换逻辑为 observation

    role: Literal["user", "assistant", "system", "tool"]
    content: Optional[str] = None
    function_call: Optional[ChoiceDeltaToolCallFunction] = None
    tool_calls: Optional[List[ChatCompletionMessageToolCall]] = None


class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None
    function_call: Optional[ChoiceDeltaToolCallFunction] = None
    tool_calls: Optional[List[ChatCompletionMessageToolCall]] = None


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length", "tool_calls"]


class ChatCompletionResponseStreamChoice(BaseModel):
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length", "tool_calls"]]
    index: int


class ChatCompletionResponse(BaseModel):
    model: str
    id: Optional[str] = Field(default_factory=lambda: generate_id('chatcmpl-', 29))
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
    system_fingerprint: Optional[str] = Field(default_factory=lambda: generate_id('fp_', 9))
    usage: Optional[UsageInfo] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.8
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    tools: Optional[Union[dict, List[dict]]] = None
    tool_choice: Optional[Union[str, dict]] = None
    repetition_penalty: Optional[float] = 1.1


class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(
            self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()  # 设置为0
            scores[..., 5] = 5e4
        return scores


def process_response(output: str, tools: dict | List[dict] = None, use_tool: bool = False) -> Union[str, dict]:
    """
    处理大模型的中间响应数据。

    函数的主要功能是处理模型生成的输出，检查是否包含工具调用，并根据工具的名称和参数格式化输出。
    如果检测到有效的工具调用且启用了工具调用，函数将返回一个包含工具名称和参数的字典；
    否则，返回原始输出字符串。这种处理方式有助于在生成过程中集成和调用外部工具。
    """
    # 第一步：分割输出
    lines = output.strip().split("\n")  
    # 第二部：初始化变量
    arguments_json = None
    # 特殊工具集
    special_tools = ["cogview", "simple_browser"]  #  glm大模型中内置的函数
    # 外部指定的工具集
    tools = {tool['function']['name'] for tool in tools} if tools else {}

    # 第三步：检查输出格式
    if len(lines) >= 2 and lines[1].startswith("{"):  # 存在工具调用
        function_name = lines[0].strip()  # 提取函数名称
        arguments = "\n".join(lines[1:]).strip()  # 提取函数的传入参数
        if function_name in tools or function_name in special_tools:
            try:
                arguments_json = json.loads(arguments)
                is_tool_call = True
            except json.JSONDecodeError:
                is_tool_call = function_name in special_tools


            # 第四步：处理工具的格式化
            if is_tool_call and use_tool:
                # 其他外部工具格式化
                content = {
                    "name": function_name,
                    "arguments": json.dumps(arguments_json if isinstance(arguments_json, dict) else arguments,
                                            ensure_ascii=False)
                }
                if function_name == "simple_browser":
                    search_pattern = re.compile(r'search\("(.+?)"\s*,\s*recency_days\s*=\s*(\d+)\)')
                    # 根据正则表达式提取：搜索关键词和最近的天数
                    match = search_pattern.match(arguments)
                    if match:
                        content["arguments"] = json.dumps({
                            "query": match.group(1),
                            "recency_days": int(match.group(2))
                        }, ensure_ascii=False)
                elif function_name == "cogview":
                    content["arguments"] = json.dumps({
                        "prompt": arguments
                    }, ensure_ascii=False)

                return content
    return output.strip()


@torch.inference_mode()
async def generate_stream_glm4(params):
    messages = params["messages"]
    tools = params["tools"]
    tool_choice = params["tool_choice"]
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    max_new_tokens = int(params.get("max_tokens", 8192))
    # 消息处理
    messages = process_messages(messages, tools=tools, tool_choice=tool_choice)
    # 初始化输入
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    params_dict = {
        "n": 1,
        "best_of": 1,
        "presence_penalty": 1.0,
        "frequency_penalty": 0.0,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": -1,
        "repetition_penalty": repetition_penalty,
        #"use_beam_search": False,
        #"length_penalty": 1,
        #"early_stopping": False,
        "stop_token_ids": [151329, 151336, 151338],
        "ignore_eos": False,
        "max_tokens": max_new_tokens,
        "logprobs": None,
        "prompt_logprobs": None,
        "skip_special_tokens": True,
    }
    sampling_params = SamplingParams(**params_dict)
    # 调用大模型
    async for output in engine.generate(prompt=inputs, sampling_params=sampling_params, request_id=f"{time.time()}"):
        output_len = len(output.outputs[0].token_ids)
        input_len = len(output.prompt_token_ids)
        ret = {
            "text": output.outputs[0].text,
            "usage": {
                "prompt_tokens": input_len,
                "completion_tokens": output_len,
                "total_tokens": output_len + input_len
            },
            "finish_reason": output.outputs[0].finish_reason,
        }
        yield ret
    gc.collect()  # gc内存
    torch.cuda.empty_cache()  # 清空显存


def process_messages(messages, tools=None, tool_choice="none"):
    _messages = messages  # 将输入的 messages 赋值给局部变量 _messages。
    processed_messages = []  # 用于存储处理后的消息列表。
    msg_has_sys = False  #  标记是否已经添加了系统消息。

    def filter_tools(tool_choice, tools):
        ''''函数的作用是从给定的工具列表中筛选出与 tool_choice 中指定的函数名称相匹配的工具'''
        function_name = tool_choice.get('function', {}).get('name', None)
        if not function_name:
            return []
        filtered_tools = [
            tool for tool in tools
            if tool.get('function', {}).get('name') == function_name
        ]
        return filtered_tools

    if tool_choice != "none":  # 用户有匹配的工具调用
        if isinstance(tool_choice, dict):
            tools = filter_tools(tool_choice, tools)
        if tools:  # 过滤之后的工具列表
            processed_messages.append(
                {
                    "role": "system",
                    "content": None,
                    "tools": tools
                }
            )
            msg_has_sys = True

    if isinstance(tool_choice, dict) and tools:
        processed_messages.append(
            {
                "role": "assistant",
                "metadata": tool_choice["function"]["name"],
                "content": ""
            }
        )

    '''
    遍历每条消息，根据其角色 (role) 进行处理：
        如果角色是 "function" 或 "tool"，则将其转换为观察消息。
        如果角色是 "assistant"，并且有 tool_calls，则为每个工具调用生成一条助手消息。
        如果没有 tool_calls，则分割内容并为每个部分生成一条助手消息。
        对于其他角色（如 "user"），直接添加到 processed_messages。
        如果角色是 "system" 且已经添加过系统消息，则跳过这条消息。
    '''
    for m in _messages:
        role, content, func_call = m.role, m.content, m.function_call
        tool_calls = getattr(m, 'tool_calls', None)

        if role == "function":
            processed_messages.append(
                {
                    "role": "observation",
                    "content": content
                }
            )
        elif role == "tool":
            processed_messages.append(
                {
                    "role": "observation",
                    "content": content,
                    "function_call": True
                }
            )
        elif role == "assistant":
            if tool_calls:
                for tool_call in tool_calls:
                    processed_messages.append(
                        {
                            "role": "assistant",
                            "metadata": tool_call.function.name,
                            "content": tool_call.function.arguments
                        }
                    )
            else:
                for response in content.split("\n"):
                    if "\n" in response:
                        metadata, sub_content = response.split("\n", maxsplit=1)
                    else:
                        metadata, sub_content = "", response
                    processed_messages.append(
                        {
                            "role": role,
                            "metadata": metadata,
                            "content": sub_content.strip()
                        }
                    )
        else:
            if role == "system" and msg_has_sys:
                msg_has_sys = False
                continue
            processed_messages.append({"role": role, "content": content})

    if not tools or tool_choice == "none":
        for m in _messages:
            if m.role == 'system':
                processed_messages.insert(0, {"role": m.role, "content": m.content})
                break
    return processed_messages


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    model_card = ModelCard(id="glm-4")
    return ModelList(data=[model_card])


"""
post是：请求方法：POST,GET,PUT,DELETE等等;
/v1/chat/completions: 路由地址，接口的请求地址
"""
@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    if len(request.messages) < 1 or request.messages[-1].role == "assistant":
        raise HTTPException(status_code=400, detail="Invalid request")

    gen_params = dict(
        messages=request.messages,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens or 1024,
        echo=False,
        stream=request.stream,
        repetition_penalty=request.repetition_penalty,
        tools=request.tools,
        tool_choice=request.tool_choice,
    )
    logger.debug(f"==== request ====\n{gen_params}")

    # 第三步：流式传输处理
    if request.stream:  # 流式处理
        # 创建一个异步的生成器
        predict_stream_generator = predict_stream(request.model, gen_params)
        # 异步获取生成器的下一条数据
        output = await anext(predict_stream_generator)
        logger.debug(f"First result output：\n{output}")
        # 代码错误
        # if output:
        #     return EventSourceResponse(predict_stream_generator, media_type="text/event-stream")

        function_call = None
        if output and request.tools:
            try:
                function_call = process_response(output, request.tools, use_tool=True)
            except:
                logger.warning("Failed to parse tool call")

        if isinstance(function_call, dict):
            function_call = ChoiceDeltaToolCallFunction(**function_call)
            generate = parse_output_text(request.model, output, function_call=function_call)
            return EventSourceResponse(generate, media_type="text/event-stream")
        else:
            return EventSourceResponse(predict_stream_generator, media_type="text/event-stream")

    # 非流式处理
    response = ""
    async for response in generate_stream_glm4(gen_params):
        pass

    if response["text"].startswith("\n"):
        response["text"] = response["text"][1:]
    response["text"] = response["text"].strip()

    usage = UsageInfo()

    function_call, finish_reason = None, "stop"
    tool_calls = None
    if request.tools:
        try:
            function_call = process_response(response["text"], request.tools, use_tool=True)
        except Exception as e:
            logger.warning(f"Failed to parse tool call: {e}")
    if isinstance(function_call, dict):
        finish_reason = "tool_calls"
        function_call_response = ChoiceDeltaToolCallFunction(**function_call)
        function_call_instance = FunctionCall(
            name=function_call_response.name,
            arguments=function_call_response.arguments
        )
        tool_calls = [
            ChatCompletionMessageToolCall(
                id=generate_id('call_', 24),
                function=function_call_instance,
                type="function")]

    message = ChatMessage(
        role="assistant",
        content=None if tool_calls else response["text"],
        function_call=None,
        tool_calls=tool_calls,
    )

    logger.debug(f"==== message ====\n{message}")

    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=message,
        finish_reason=finish_reason,
    )
    task_usage = UsageInfo.model_validate(response["usage"])
    for usage_key, usage_value in task_usage.model_dump().items():
        setattr(usage, usage_key, getattr(usage, usage_key) + usage_value)

    return ChatCompletionResponse(
        model=request.model,
        choices=[choice_data],
        object="chat.completion",
        usage=usage
    )


async def predict_stream(model_id, gen_params):
    output = ""
    is_function_call = False
    has_send_first_chunk = False  #是否为第一个chunk。 流式输出：每次大模型输出一个chunk
    created_time = int(time.time())
    function_name = None
    response_id = generate_id('chatcmpl-', 29)
    system_fingerprint = generate_id('fp_', 9)
    tools = {tool['function']['name'] for tool in gen_params['tools']} if gen_params['tools'] else {}
    delta_text = ""  # 每个分块的文本内容（token）
    async for new_response in generate_stream_glm4(gen_params):
        decoded_unicode = new_response["text"]
        delta_text += decoded_unicode[len(output):]
        output = decoded_unicode
        lines = output.strip().split("\n")

        # 检查是否为工具
        # 这是一个简单的工具比较函数，不能保证拦截所有非工具输出的结果，比如参数未对齐等特殊情况。
        ##TODO 如果你希望做更多处理，可以在这里进行逻辑完善。

        if not is_function_call and len(lines) >= 2:
            first_line = lines[0].strip()
            if first_line in tools:
                is_function_call = True
                function_name = first_line  # 工具的名字
                delta_text = lines[1]  # 工具的参数

        # 工具调用返回
        if is_function_call:
            if not has_send_first_chunk:
                # 返回工具调用的第一个chunk
                function_call = {"name": function_name, "arguments": ""}
                tool_call = ChatCompletionMessageToolCall(
                    index=0,
                    id=generate_id('call_', 24),
                    function=FunctionCall(**function_call),
                    type="function"
                )
                message = DeltaMessage(
                    content=None,
                    role="assistant",
                    function_call=None,
                    tool_calls=[tool_call]
                )
                choice_data = ChatCompletionResponseStreamChoice(
                    index=0,
                    delta=message,
                    finish_reason=None
                )
                chunk = ChatCompletionResponse(
                    model=model_id,
                    id=response_id,
                    choices=[choice_data],
                    created=created_time,
                    system_fingerprint=system_fingerprint,
                    object="chat.completion.chunk"
                )
                yield ""
                yield chunk.model_dump_json(exclude_unset=True)
                has_send_first_chunk = True

            # 工具调用后续的chunk
            function_call = {"name": None, "arguments": delta_text}
            delta_text = ""
            tool_call = ChatCompletionMessageToolCall(
                index=0,
                id=None,
                function=FunctionCall(**function_call),
                type="function"
            )
            message = DeltaMessage(
                content=None,
                role=None,
                function_call=None,
                tool_calls=[tool_call]
            )
            choice_data = ChatCompletionResponseStreamChoice(
                index=0,
                delta=message,
                finish_reason=None
            )
            chunk = ChatCompletionResponse(
                model=model_id,
                id=response_id,
                choices=[choice_data],
                created=created_time,
                system_fingerprint=system_fingerprint,
                object="chat.completion.chunk"
            )
            yield chunk.model_dump_json(exclude_unset=True)

        # 如果满足这个条件，说明用户可能希望使用工具，但目前还没有从模型输出中检测到具体的工具调用。
        elif (gen_params["tools"] and gen_params["tool_choice"] != "none") or is_function_call:
            continue

        # 常规返回
        else:
            finish_reason = new_response.get("finish_reason", None)
            # 常规返回的第一个chunk
            if not has_send_first_chunk:
                message = DeltaMessage(
                    content="",
                    role="assistant",
                    function_call=None,
                )
                choice_data = ChatCompletionResponseStreamChoice(
                    index=0,
                    delta=message,
                    finish_reason=finish_reason
                )
                chunk = ChatCompletionResponse(
                    model=model_id,
                    id=response_id,
                    choices=[choice_data],
                    created=created_time,
                    system_fingerprint=system_fingerprint,
                    object="chat.completion.chunk"
                )
                yield chunk.model_dump_json(exclude_unset=True)
                has_send_first_chunk = True

            # 常规返回的后续的chunk
            message = DeltaMessage(
                content=delta_text,
                role="assistant",
                function_call=None,
            )
            delta_text = ""
            choice_data = ChatCompletionResponseStreamChoice(
                index=0,
                delta=message,
                finish_reason=finish_reason
            )
            chunk = ChatCompletionResponse(
                model=model_id,
                id=response_id,
                choices=[choice_data],
                created=created_time,
                system_fingerprint=system_fingerprint,
                object="chat.completion.chunk"
            )
            yield chunk.model_dump_json(exclude_unset=True)

    # 工具调用需要额外返回一个字段以对齐 OpenAI 接口
    if is_function_call:
        yield ChatCompletionResponse(
            model=model_id,
            id=response_id,
            system_fingerprint=system_fingerprint,
            choices=[
                ChatCompletionResponseStreamChoice(
                    index=0,
                    delta=DeltaMessage(
                        content=None,
                        role=None,
                        function_call=None,
                    ),
                    finish_reason="tool_calls" # 工具调用的chunk结束
                )],
            created=created_time,
            object="chat.completion.chunk",
            usage=None
        ).model_dump_json(exclude_unset=True)
    elif delta_text != "":
        # 发出最后一个文本的chunk
        message = DeltaMessage(
            content="",
            role="assistant",
            function_call=None,
        )
        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=message,
            finish_reason=None
        )
        chunk = ChatCompletionResponse(
            model=model_id,
            id=response_id,
            choices=[choice_data],
            created=created_time,
            system_fingerprint=system_fingerprint,
            object="chat.completion.chunk"
        )
        yield chunk.model_dump_json(exclude_unset=True)
    
        # 返回最后一个带有结束 标记的chunk
        finish_reason = 'stop'
        message = DeltaMessage(
            content=delta_text,
            role="assistant",
            function_call=None,
        )
        delta_text = ""
        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=message,
            finish_reason=finish_reason
        )
        chunk = ChatCompletionResponse(
            model=model_id,
            id=response_id,
            choices=[choice_data],
            created=created_time,
            system_fingerprint=system_fingerprint,
            object="chat.completion.chunk"
        )
        yield chunk.model_dump_json(exclude_unset=True)
        yield '[DONE]'
    else:
        yield '[DONE]'

async def parse_output_text(model_id: str, value: str, function_call: ChoiceDeltaToolCallFunction = None):
    delta = DeltaMessage(role="assistant", content=value)
    if function_call is not None:
        delta.function_call = function_call

    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=delta,
        finish_reason=None
    )
    chunk = ChatCompletionResponse(
        model=model_id,
        choices=[choice_data],
        object="chat.completion.chunk"
    )
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))
    yield '[DONE]'


if __name__ == "__main__":
    MODEL_PATH = '/root/autodl-tmp/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B' # 指定模型路径
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    engine_args = AsyncEngineArgs(
        model=MODEL_PATH,
        tokenizer=MODEL_PATH,
        # 如果你有多张显卡，可以在这里设置成你的显卡数量
        tensor_parallel_size=1,
        dtype="bfloat16",
        # dtype="half",
        trust_remote_code=True,
        # 占用显存的比例，请根据你的显卡显存大小设置合适的值，例如，如果你的显卡有80G，您只想使用24G，请按照24/80=0.3设置
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        worker_use_ray=False,
        disable_log_requests=True,
        max_model_len=MAX_MODEL_LENGTH,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    # fastapi启动的代码
    uvicorn.run(app, host='0.0.0.0', port=6006, workers=1)