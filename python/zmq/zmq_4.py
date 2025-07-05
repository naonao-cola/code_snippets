import zmq
import json
import time
import multiprocessing
import uuid
import os


### 生产者数据 均衡的发给生产者
"""
    消息队列：zmq.QUEUE 提供了一个内置的消息队列，可以将生产者发送的消息存储起来。
    多生产者多消费者支持：支持多个生产者进程向队列发送消息，多个消费者进程从队列中接收消息。
    负载均衡：将消息均匀地分发给消费者，确保每个消费者都能处理大致相同数量的消息。

"""
# 生产者函数
def producer():
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.connect("inproc://internal")

    # 生成一些JSON数据
    for i in range(20):
        data = {
            "id": str(uuid.uuid4()),
            "message": f"Message {i} - {time.time()}",
            "producer_id": f"Producer_{os.getpid()}"
        }
        # 将Python字典转换为JSON字符串
        json_msg = json.dumps(data).encode('utf-8')
        socket.send(json_msg)
        print(f"Produced: {data}")
        time.sleep(1)

    socket.close()
    context.destroy()

# 消费者函数
def consumer():
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.connect("inproc://internal")

    # 从队列中接收消息
    while True:
        message = socket.recv()
        data = json.loads(message.decode('utf-8'))
        print(f"Consumer {os.getpid()}: Consumed: {data}")

        # 模拟处理时间
        time.sleep(0.5)

    socket.close()
    context.destroy()

if __name__ == "__main__":
    # 创建ZeroMQ上下文和队列
    context = zmq.Context.instance()
    queue = context.socket(zmq.QUEUE)
    queue.bind("inproc://internal")

    # 创建多个生产者进程
    producers = []
    for _ in range(3):  # 3个生产者
        p = multiprocessing.Process(target=producer)
        p.start()
        producers.append(p)

    # 创建多个消费者进程
    consumers = []
    for _ in range(2):  # 2个消费者
        c = multiprocessing.Process(target=consumer)
        c.start()
        consumers.append(c)

    # 等待生产者进程完成
    for p in producers:
        p.join()

    # 发送退出信号
    for _ in range(len(consumers)):
        queue.send_json({"command": "exit"})

    # 等待消费者进程完成
    for c in consumers:
        c.join()

    # 清理
    queue.close()
    context.destroy()