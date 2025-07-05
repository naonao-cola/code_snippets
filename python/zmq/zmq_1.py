import zmq
import json
import time
import multiprocessing
import uuid
import time
import random

# 生产者函数


def producer(port):
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://127.0.0.1:{port}")

   # 生成一些JSON数据
    while True:
        topic = random.choice(["A", "B", "C"])
        message = f"Message from publisher on port {port}, topic: {topic}"
        socket.send_string(f"{topic} {message}")
        time.sleep(1)


# 消费者函数
def consumer(name):
    context = zmq.Context()
    socket = context.socket(zmq.SUB)

    # 假设有多个发布者，我们可以在这里连接多个端口
    publishers_ports = ["5555", "5556"]
    for port in publishers_ports:
        socket.connect(f"tcp://127.0.0.1:{port}")

    socket.setsockopt_string(zmq.SUBSCRIBE, 'A')

    while True:
        try:
            message = socket.recv_string(zmq.DONTWAIT)
            #msg = socket.recv(zmq.DONTWAIT)
        except zmq.Again:
            # 如果没有消息，继续等待
            print(f"未收到消息")
            time.sleep(0.1)
            continue
        # 分割主题和消息内容
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
            time.sleep(3)  # 模拟耗时的操作


if __name__ == "__main__":
   # 创建多个生产者进程，每个生产者使用不同的端口
    producers = []
    for port in ["5555", "5556"]:  # 假设有两个发布者，分别在5555和5556端口
        p = multiprocessing.Process(target=producer, args=(port,))
        p.start()
        producers.append(p)

    # 创建多个消费者进程
    consumers = []
    for i in range(2):  # 创建2个消费者
        c = multiprocessing.Process(target=consumer, args=(f"Subscriber {i}",))
        c.start()
        consumers.append(c)

    # 等待生产者进程完成
    for p in producers:
        p.join()

    # 等待消费者进程完成
    for c in consumers:
        c.join()
