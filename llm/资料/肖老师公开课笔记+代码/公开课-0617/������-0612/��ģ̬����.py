import base64
import io
import uuid

import gradio as gr
from PIL import Image
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

multiModal_llm = ChatOpenAI(
    model='qwen-omni-3b',
    api_key='xxxxx',
    base_url='http://127.0.0.1:6006/v1',

)


prompt = ChatPromptTemplate.from_messages(
    [
        ('system', "你是一个多模态AI助手，可以处理文本、音频和图像输入"),
        MessagesPlaceholder(variable_name="msg"),  # 消息占位符 代表：历史消息。 让大模型可以理解上下文语义
    ]
)

chain = prompt | multiModal_llm


def get_session_history(session_id: str):
    """从关系型数据库的历史消息列表中 返回当前会话 的所有历史消息"""
    return SQLChatMessageHistory(
        session_id=session_id,
        table_name="t_session_history",
        connection_string='sqlite:///chat_history.db',  # mysql+pymysql://root@密码@127.0.0.1:3306/db_test
    )

chain_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
)

config = {"configurable": {"session_id": str(uuid.uuid4())}}


# 定义处理提交事件的函数
def add_message(history, user_input):
    """将用户输入的消息添加到聊天记录中"""
    if user_input['text'] is not None:   # 输入的文本消息
        history.append({"role": "user", "content": user_input["text"]})

    for m in user_input['files']:
        print(m)
        history.append({'role': 'user', "content": {'path': m}})

    return history, ''


# 语音处理函数 =====
def transcribe_audio(audio_path):
    """使用Base64处理语音转为"""

    # 目前多模态大模型： 支持两个传参方式，1、base64（字符串）（本地）。2、网络访问的url地址（外网的服务器上） http://sxxxx.com/11.mp3
    try:
        with open(audio_path, 'rb') as audio_file:
            audio_data = base64.b64encode(audio_file.read()).decode('utf-8')
        audio_message = {  # 把音频文件，封装成一条消息
            "type": "audio_url",
            "audio_url": {
                "url": f"data:audio/wav;base64,{audio_data}",
                "duration": 30  # 单位：秒（帮助模型优化处理）
            }
        }

        return audio_message
    except Exception as e:
        print(e)
        return {}


def transcribe_image(image_path):
    """
    将任意格式的图片转换为base64编码的data URL
    :param image_path: 图片路径
    :return: 包含base64编码的字典
    """
    with Image.open(image_path) as img:
        # 获取原始图片格式（如JPEG/PNG）
        img_format = img.format if img.format else 'JPEG'

        buffered = io.BytesIO()
        # 保留原始格式（避免JPEG强制转换导致透明通道丢失）
        img.save(buffered, format=img_format)

        image_data = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/{img_format.lower()};base64,{image_data}",
                "detail": 'low'
            }
        }

def get_last_user_after_assistant(history):
    """反向遍历找到最后一个assistant的位置,并返回后面的所有user消息"""
    if not history:
        return None
    if history[-1]["role"] == "assistant":
        return None

    last_assistant_idx = -1
    for i in range(len(history) - 1, -1, -1):
        if history[i]["role"] == "assistant":
            last_assistant_idx = i
            break

    # 如果没有找到assistant
    if last_assistant_idx == -1:
        return history
    else:
        # 从assistant位置向后查找第一个user
        return history[last_assistant_idx + 1:]


def submit_messages(history):
    """提交用户输入的消息，生成机器人回复"""
    user_messages = get_last_user_after_assistant(history)
    print(user_messages)
    content = []  # HumanMessage的内容
    if user_messages:
        for x in user_messages:
            if isinstance(x['content'], str):  # 文字输入消息
                content.append({'type': 'text', 'text': x['content']})
            elif isinstance(x['content'], tuple):  # 多媒体输入消息
                file_path = x['content'][0]  # 得到多媒体的文件路径
                if file_path.endswith('.wav'):  #  输入的是音频文件
                    file_message = transcribe_audio(file_path)
                elif file_path.endswith(".jpg") or file_path.endswith(".png") or file_path.endswith(".jpeg"):
                    file_message = transcribe_image(file_path)
                content.append(file_message)
            else:
                pass
    input_message = HumanMessage(content=content)

    resp = chain_history.invoke({'messages': input_message}, config)
    history.append({'role': 'assistant', 'content': resp.content})
    return history

# 可视化界面
with gr.Blocks(title='我的多模态', theme=gr.themes.Soft()) as instance:
    # 聊天历史记录的组件
    chatbot = gr.Chatbot(type='messages', height=500, label='我的多模态数字人')

    # 创建多模态输入框
    chat_input = gr.MultimodalTextbox(
        file_types=['image', '.wav', '.mp3', '.mp4', '.doc'],
        file_count='multiple',
        placeholder='请输入信息或者语音或者其他多媒体...',
        show_label=False,
        sources=['microphone', 'upload']
    )


    chat_input.submit(
        add_message,  # 处理提交事件的函数
        [chatbot, chat_input],
        [chatbot, chat_input]
    ).then(
        submit_messages,  # 提交到大模型
        [chatbot],
        [chatbot],
    ).then(  # 回复完成后重新激活输入框
        lambda: gr.MultimodalTextbox(interactive=True),  # 匿名函数重置输入框
        None,  # 无输入
        [chat_input]  # 输出到输入框
    )





if __name__ == '__main__':
    instance.launch()


