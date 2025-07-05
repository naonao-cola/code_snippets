import zmq
import json
import time
import multiprocessing
import random

# 生产者函数


def producer(port):
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://127.0.0.1:{port}")

    while True:
        topic = random.choice(["A", "B", "C"])
        message = f"Message from publisher on port {port}, topic: {topic}"
        socket.send_string(f"{topic} {message}")
        time.sleep(1)

# 消费者函数，使用 zmq.Poller


def consumer(name):
    context = zmq.Context()
    socket = context.socket(zmq.SUB)

    # 连接到多个发布者，假设发布者分别在端口 5555 和 5556
    publishers_ports = ["5555", "5556"]
    for port in publishers_ports:
        socket.connect(f"tcp://127.0.0.1:{port}")

    socket.setsockopt_string(zmq.SUBSCRIBE, "")  # 订阅所有主题

    # 创建一个 Poller，用于监听 socket 上的事件
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)  # 注册 socket，监听 POLLIN 事件（有消息可接收）

    while True:
        # 使用 poll() 方法监听事件，设置超时时间（毫秒）
        # 这里设置为 1000 毫秒（1 秒），如果没有事件发生，poll() 将返回空列表
        events = poller.poll(100)

        if events:
            # 如果有事件发生，接收消息
            message = socket.recv_string()
            try:
                topic, content = message.split(" ", 1)
            except ValueError:
                continue

            # 根据消息类型进行不同处理
            if topic == "A":
                print(f"{name} processed type A: {content}")
            elif topic == "B":
                print(f"{name} processed type B: {content.upper()}")
            elif topic == "C":
                print(f"{name} processed type C: {content}")
                time.sleep(3)  # 模拟耗时操作
        else:
            # 如果在超时时间内没有事件发生
            print(f"{name} - No message received in the last second")


if __name__ == "__main__":
    # 创建多个生产者进程，每个生产者使用不同的端口
    producers = []
    for port in ["5555", "5556"]:  # 假设有两个发布者，分别在 5555 和 5556 端口
        p = multiprocessing.Process(target=producer, args=(port,))
        p.start()
        producers.append(p)

    # 创建多个消费者进程
    consumers = []
    for i in range(2):  # 创建 2 个消费者
        c = multiprocessing.Process(target=consumer, args=(f"Subscriber {i}",))
        c.start()
        consumers.append(c)

    # 等待生产者进程完成
    for p in producers:
        p.join()

    # 等待消费者进程完成
    for c in consumers:
        c.join()
