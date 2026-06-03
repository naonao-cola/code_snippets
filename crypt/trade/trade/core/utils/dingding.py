"""
邢不行™️选币实盘框架
Python数字货币量化投资课程

版权所有 ©️ 邢不行
微信: xbx8662

未经授权，不得复制、修改、或使用本代码的全部或部分内容。仅限个人学习用途，禁止商业用途。

Author: 邢不行
"""

import base64
import hashlib
import os.path
import requests
import json
import traceback
from datetime import datetime
from config import proxy


# 企业微信通知
def send_wechat_work_msg(content, url):
    if not url:
        print('未配置wechat_webhook_url，不发送信息')
        return
    try:
        data = {
            "msgtype": "text",
            "text": {
                "content": content + '\n' + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        r = requests.post(url, data=json.dumps(data), timeout=10, proxies=proxy)
        print(f'调用企业微信接口返回： {r.text}')
        print('成功发送企业微信')
    except Exception as e:
        print(f"发送企业微信失败:{e}")
        print(traceback.format_exc())


# 上传图片，解析bytes
class MyEncoder(json.JSONEncoder):

    def default(self, obj):
        """
        只要检查到了是bytes类型的数据就把它转为str类型
        :param obj:
        :return:
        """
        if isinstance(obj, bytes):
            return str(obj, encoding='utf-8')
        return json.JSONEncoder.default(self, obj)


# 企业微信发送图片
def send_wechat_work_img(file_path, url):
    if not os.path.exists(file_path):
        print('找不到图片')
        return
    if not url:
        print('未配置wechat_webhook_url，不发送信息')
        return
    try:
        with open(file_path, 'rb') as f:
            image_content = f.read()
        image_base64 = base64.b64encode(image_content).decode('utf-8')
        md5 = hashlib.md5()
        md5.update(image_content)
        image_md5 = md5.hexdigest()
        data = {
            'msgtype': 'image',
            'image': {
                'base64': image_base64,
                'md5': image_md5
            }
        }
        # 服务器上传bytes图片的时候，json.dumps解析会出错，需要自己手动去转一下
        r = requests.post(url, data=json.dumps(data, cls=MyEncoder, indent=4), timeout=10, proxies=proxy)
        print(f'调用企业微信接口返回： {r.text}')
        print('成功发送企业微信')
    except Exception as e:
        print(f"发送企业微信失败:{e}")
        print(traceback.format_exc())
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


def send_msg_for_order(order_param, order_res, url):
    """
    发送下单信息，只有出问题才会推送，正常下单不在推送信息
    """
    if not url:
        print('未配置wechat_webhook_url，不发送信息')
        return
    msg = ''
    try:
        for _ in range(len(order_param)):
            if 'msg' in order_res[_].keys():
                msg += f'币种:{order_param[_]["symbol"]}\n'
                msg += f'方向:{"做多" if order_param[_]["side"] == "BUY" else "做空"}\n'
                msg += f'价格:{order_param[_]["price"]}\n'
                msg += f'数量:{order_param[_]["quantity"]}\n'
                msg += f'下单结果:{order_res[_]["msg"]}'
                msg += '\n' * 2
    except BaseException as e:
        print('send_msg_for_order ERROR', e)
        print(traceback.format_exc())

    if msg:
        send_wechat_work_msg(msg, url)
