"""
邢不行™️选币实盘框架
Python数字货币量化投资课程

版权所有 ©️ 邢不行
微信: xbx8662

未经授权，不得复制、修改、或使用本代码的全部或部分内容。仅限个人学习用途，禁止商业用途。

Author: 邢不行
"""
import time

from core.utils.path_kit import get_folder_path

"""🚨 邢不行Warning
🚫当前策略提供的默认策略，供框架研究学习使用，不能直接用于实盘。
📚请仔细研究策略、确定参数、更新config文件后，再进行实盘操作。
🛟如果有不明白的地方可以联系助教，或者私信邢不行。
🔓确定你的代码后，修改 `startup.py` 中的 `safe_mode` 即可
"""

# ====================================================================================================
# ** 账户及策略配置 **
# 【核心设置区域】设置账户API，策略详细信息，交易的一些特定参数等等
# * 注意，以下功能都是在config.py中实现
# ====================================================================================================
account_config = {
    # 交易所API配置
    'name': '策略研究实盘账户',
    'apiKey': '',
    'secret': '',

    # ++++ 策略配置 ++++
    "strategy": {
        "hold_period": "24H",  # 持仓周期，可以是H小时，或者D天。例如：1H，8H，24H，1D，3D，7D...
        "long_select_coin_num": 3,  # 多头选币数量，可为整数或百分比。2 表示 2 个，10 / 100 表示前 10%
        "short_select_coin_num": 3,  # 空头选币数量。除和多头相同外，还支持 'long_nums' 表示与多头数量一致。
        # 注意：在is_pure_long = True时，short_select_coin_num参数无效

        "factor_list": [  # 选币因子列表
            # 因子名称（与 factors 文件中的名称一致），排序方式（True 为升序，从小到大排，False 为降序，从大到小排），因子参数，因子权重
            ('VolumeMeanRatio', True, 18 * 24, 1),
            # 可添加多个选币因子
        ],
        "filter_list": [  # 过滤因子列表
            # 因子名称（与 factors 文件中的名称一致），因子参数，因子过滤规则，排序方式
            # ('QuoteVolumeMean', 7, 'pct:<0.5', True),

            # ** 因子过滤规则说明 **
            # 支持三种过滤规则：`rank` 排名过滤、`pct` 百分比过滤、`val` 数值过滤
            # - `rank:<10` 仅保留前 10 名的数据；`rank:>=10` 排除前 10 名。支持 >、>=、<、<=、==、!=
            # - `pct:<0.8` 仅保留前 80% 的数据；`pct:>=0.8` 仅保留后 20%。支持 >、>=、<、<=、==、!=
            # - `val:<0.1` 仅保留小于 0.1 的数据；`val:>=0.1` 仅保留大于等于 0.1 的数据。支持 >、>=、<、<=、==、!=
            # 可添加多个过滤因子和规则，多个条件将取交集
            # ('PctChange', 7, 'pct:>0.9', True),
        ],
    },

    'is_pure_long': True,  # True为纯多模式，False为多空模式。纯多模式下，仅使用现货数据；多空模式下，仅使用合约数据。
    'use_offset': False,  # 是否开启offset功能，False为不开启，True为开启。

    "wechat_webhook_url": 'https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=',
    # 创建企业微信机器人 参考帖子: https://bbs.quantclass.cn/thread/10975
    # 配置案例  https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxxxxxxxxxxxxxxxxxx

    # ++++ 其他配置 ++++
    # 其他实盘功能配置，可以使用默认配置，也可以自己配置
    "black_list": [],  # 黑名单列表，不参与交易的币种
    "leverage": 1,  # 交易杠杆
    "get_kline_num": 999,  # 用于计算行情k线数量，和你的因子计算需要的k线数量有关。这里跟策略日频和小时频影响。日线策略，代表999根日线k。小时策略，代表999根小时k
    "min_kline_num": 168,  # 最低要求b中有多少小时的K线， 需要过滤掉少于这个k线数量的比重，用于排除新币。168=7x24h
}

is_debug = True  # debug模式。模拟运行程序，不会去下单，正式部署实盘之前记得切换一下运行模式哦
# ====================================================================================================
# ** 交易所配置 **
# ====================================================================================================
# 如果使用代理 注意替换IP和Port
proxy = {}
# proxy = {'http': 'http://127.0.0.1:7890', 'https': 'http://127.0.0.1:7890'}  # 如果你用clash的话
exchange_basic_config = {
    'timeout': 30000,
    'rateLimit': 30,
    'enableRateLimit': False,
    'options': {
        'adjustForTimeDifference': True,
        'recvWindow': 10000,
    },
    'proxies': proxy,
}

# ====================================================================================================
# ** 运行模式及交易细节设置 **
# 设置系统的时差、并行数量，稳定币，特殊币种等等
# ====================================================================================================
# 获取当前服务器时区，距离UTC 0点的偏差
utc_offset = int(time.localtime().tm_gmtoff / 60 / 60)  # 如果服务器在上海，那么utc_offset=8

# 现货稳定币名单，不参与交易的币种
stable_symbol = ['BKRW', 'USDC', 'USDP', 'TUSD', 'BUSD', 'FDUSD', 'DAI', 'EUR', 'GBP', 'USBP', 'SUSD', 'PAXG', 'AEUR',
                 'EURI']

# kline下载数据类型。支持:spot , swap, funding
# 如果只需要现货和合约，可以只下载spot和swap，需要资金费数据，配置funding
download_kline_list = ['swap', 'spot', ]

# 特殊现货对应列表。有些币种的现货和合约的交易对不一致，需要手工做映射
special_symbol_dict = {
    'DODO': 'DODOX',  # DODO现货对应DODOX合约
    'LUNA': 'LUNA2',  # LUNA现货对应LUNA2合约
    '1000SATS': '1000SATS',  # 1000SATS现货对应1000SATS合约
}

# 全局报错机器人通知
# - 创建企业微信机器人 参考帖子: https://bbs.quantclass.cn/thread/10975
# - 配置案例  https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxxxxxxxxxxxxxxxxxx
error_webhook_url = 'https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key='

# ====================================================================================================
# ** 文件系统相关配置 **
# - 获取一些全局路径
# - 自动创建缺失的文件夹们
# ====================================================================================================
# 获取目录位置，不存在就创建目录
data_path = get_folder_path('data')

# 获取目录位置，不存在就创建目录
data_center_path = get_folder_path('data', 'data_center', as_path_type=True)

# 获取目录位置，不存在就创建目录
flag_path = get_folder_path('data', 'flag', as_path_type=True)

# 获取目录位置，不存在就创建目录
order_path = get_folder_path('data', 'order', as_path_type=True)

# 获取目录位置，不存在就创建目录
runtime_folder = get_folder_path('data', 'runtime', as_path_type=True)
