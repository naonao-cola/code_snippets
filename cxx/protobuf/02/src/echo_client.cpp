#include <brpc/channel.h>
#include <thread>
#include "echo.pb.h"

void callback(brpc::Controller* cntl, ::example::EchoResponse* response) {
    std::unique_ptr<brpc::Controller> cntl_guard(cntl);
    std::unique_ptr<example::EchoResponse> resp_guard(response);
    if (cntl->Failed() == true) {
        std::cout << "Rpc调用失败: " << cntl->ErrorText() << "\n";
        return;
    }
    std::cout << "收到响应: " << response->message() << "\n";
}

int main(int argc, char *argv[]){
    // 1. 构造 Channel 信道 连接服务器
    brpc::ChannelOptions options;
    options.connect_timeout_ms = -1;    // 连接等待超时时间，-1表示一直等待
    options.timeout_ms = -1;            // rpc请求等待超时时间，-1表示一直等待
    options.max_retry = 3;              // 请求重试次数
    options.protocol = "baidu_std";     // 序列化协议，默认使用baidu_std

    brpc::Channel channel;
    int ret = channel.Init("127.0.0.1:8080", &options);
    if (ret == -1) {
        std::cout << "初始化信道失败！\n";
        return -1;
    }

    // 2. 构造 EchoService_stub 对象 进行 rpc 调用
    example::EchoService_Stub stub(&channel);
    // 3. rpc 调用
    example::EchoRequest req;
    req.set_message("你好 Island");

    brpc::Controller *cntl = new brpc::Controller();
    example::EchoResponse *rsp = new example::EchoResponse();

    // 同步处理流程
    stub.Echo(cntl, &req, rsp, nullptr);
    if (cntl->Failed() == true) {
        std::cout << "Rpc调用失败：" << cntl->ErrorText() << std::endl;
        return -1;
    }
    std::cout << "收到响应: " << rsp->message() << std::endl;
    delete cntl;
    delete rsp;

    // 异步调用处理流程
    // auto clusure = google::protobuf::NewCallback(callback, cntl, rsp);
    // stub.Echo(cntl, &req, rsp, clusure);
    // std::cout << "异步调用结束！\n";
    // std::this_thread::sleep_for(std::chrono::seconds(3));
    return 0;
}
