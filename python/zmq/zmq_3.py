import zmq
import time
import multiprocessing

## 共享队列

"""
    zmq.ROUTER 是一种 Socket 类型，常用于服务器端，它可以管理多个客户端连接。
    它会自动跟踪每个连接的客户端，并在转发消息时携带客户端的标识信息。
    当 ROUTER Socket 收到消息时，它会将消息转发给指定的客户端，并确保消息能够正确地返回给发送者。

示例代码中的作用：

    frontend = context.socket(zmq.ROUTER)：创建一个 ROUTER Socket，用于接收来自请求者（REQ Socket）的消息。
    frontend.bind("inproc://frontend")：绑定到 inproc://frontend 地址，请求者通过这个地址连接到代理。

    zmq.DEALER 是一种 Socket 类型，常用于客户端，它可以将消息均匀地分发给多个服务端。
    它以轮询（round-robin）的方式将消息分发给连接的 Socket。
    DEALER Socket 通常用于负载均衡，确保多个服务端能够公平地处理请求。

示例代码中的作用：

    backend = context.socket(zmq.DEALER)：创建一个 DEALER Socket，用于将请求均匀地分发给多个响应者（REP Socket）。
    backend.bind("inproc://backend")：绑定到 inproc://backend 地址，响应者通过这个地址连接到代理。


    挨个轮训发送
    
"""
# 请求者（Requester）函数
def requester():
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://127.0.0.1:5555")

    for i in range(10):  # 发送 10 条请求
        message = f"Request {i}"
        socket.send_string(message)
        print(f"requester sent: {message}")
        response = socket.recv_string()
        print(f"requester received: {response}")
        time.sleep(1)

# 响应者（Responser）函数


def responser():
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.connect("tcp://127.0.0.1:5556")

    while True:
        request = socket.recv_string()
        print(f"responser received: {request}")
        response = f"responser to {request}"
        socket.send_string(response)
        # 模拟处理时间
        time.sleep(1)

# 代理函数（使用 zmq.QUEUE）


def proxy():
    context = zmq.Context.instance()
    frontend = context.socket(zmq.ROUTER)
    backend = context.socket(zmq.DEALER)
    frontend.bind("tcp://127.0.0.1:5555")
    backend.bind("tcp://127.0.0.1:5556")


    # 使用 zmq.proxy_steerable 创建一个可控制的代理
    # 参数解释：
    # - frontend：请求者连接的套接字
    # - backend：响应者连接的套接字
    # - capture：捕获的日志套接字（可选）
    # - signal：用于控制代理的套接字（可选）
    zmq.proxy_steerable(frontend, backend)


if __name__ == "__main__":
    # 创建代理进程
    proxy_process = multiprocessing.Process(target=proxy)
    proxy_process.start()

    # 创建请求者进程
    requester_process = multiprocessing.Process(target=requester)
    requester_process.start()

    # 创建响应者进程
    responser_process = multiprocessing.Process(target=responser)
    responser_process.start()

    # 等待请求者完成
    requester_process.join()

    # 关闭其他进程
    proxy_process.terminate()
    responser_process.terminate()

