from modelscope import AutoModelForCausalLM, AutoTokenizer
from threading import Thread
from transformers import TextIteratorStreamer
import gradio as gr



tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/models/Qwen/Qwen3-8B')
model = AutoModelForCausalLM.from_pretrained(
    '/root/autodl-tmp/models/Qwen/Qwen3-8B',
    torch_dtype="auto",
    device_map="auto"
)

with gr.Blocks() as demo:
    # 添加HTML标题
    gr.HTML("""<h1 align="center">马士兵教育私有AI大模型应用</h1>""")
    chatbot = gr.Chatbot(type="messages")
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    def user(user_message, history: list):
        return "", history + [{"role": "user", "content": user_message}]

    def bot(history: list):
        # bot_message = history[-1]['content']
        # 应用对话模板（禁用思考模式）
        text = tokenizer.apply_chat_template(
            history,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  # 关闭思考模式简化输出
        )
        
        # 创建流式迭代器（跳过特殊token）
        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        # 准备模型输入
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        # 在独立线程中启动生成过程
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=1024  # 限制生成长度
        )
        Thread(target=model.generate, kwargs=generation_kwargs).start()
        
        # 实时返回生成的文本
        # full_response = ""
        # for new_text in streamer:
        #     full_response += new_text
        #     yield new_text  # 逐片段返回
        history.append({"role": "assistant", "content": ""})
        for character in streamer:
            history[-1]['content'] += character
            yield history

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)

demo.launch(server_name="127.0.0.1", server_port=6006)
        
        
# # 示例用法
# if __name__ == "__main__":
#     chatbot = QwenChatbot()
    
#     # 第一轮对话（流式输出）
#     print("User: 如何学习Python?")
#     print("Bot: ", end="", flush=True)
#     for chunk in chatbot.stream_response("如何学习Python?"):
#         print(chunk, end="", flush=True)  # 实时打印
#     print("\n----------------------")
    
#     # 第二轮对话（延续上下文）
#     print("User: 需要哪些具体步骤？")
#     print("Bot: ", end="", flush=True)
#     for chunk in chatbot.stream_response("需要哪些具体步骤？"):
#         print(chunk, end="", flush=True)