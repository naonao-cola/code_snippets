"""
邢不行™️选币框架
Python数字货币量化投资课程

版权所有 ©️ 邢不行
微信: xbx8662

未经授权，不得复制、修改、或使用本代码的全部或部分内容。仅限个人学习用途，禁止商业用途。

Author: 邢不行
"""
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Set

import pandas as pd

from config import backtest_path, backtest_name
from core.model.strategy_config import StrategyConfig
from core.utils.path_kit import get_folder_path  # 西蒙斯提供的自动获取绝对路径的函数，若存储目录不存在则自动创建


class BacktestConfig:
    data_file_fingerprint: str = ''  # 记录数据文件的指纹

    def __init__(self, name: str, **config):
        self.name: str = name  # 账户名称，建议用英文，不要带有特殊符号
        self.start_date: str = config.get("start_date", '2021-01-01')  # 回测开始时间
        self.end_date: str = config.get("end_date", '2024-03-30')  # 回测结束时间

        # 账户回测交易模拟配置
        self.initial_usdt: int | float = config.get("initial_usdt", 10000)  # 初始现金
        self.leverage: int | float = config.get("leverage", 1)  # 杠杆数。我看哪个赌狗要把这里改成大于1的。高杠杆如梦幻泡影。不要想着一夜暴富，脚踏实地赚自己该赚的钱。
        self.margin_rate = 5 / 100  # 维持保证金率，净值低于这个比例会爆仓

        self.swap_c_rate: float = config.get("swap_c_rate", 6e-4)  # 合约买卖手续费
        self.spot_c_rate: float = config.get("spot_c_rate", 2e-3)  # 现货买卖手续费

        self.swap_min_order_limit: int = 5  # 合约最小下单量
        self.spot_min_order_limit: int = 10  # 现货最小下单量

        # 策略配置
        # 拉黑名单，永远不会交易。不喜欢的币、异常的币。例：LUNA-USDT, 这里与实盘不太一样，需要有'-'
        self.black_list: List[str] = config.get('black_list', [])
        # 最少上市多久，不满该K线根数的币剔除，即剔除刚刚上市的新币。168：标识168个小时，即：7*24
        self.min_kline_num: int = config.get('min_kline_num', 168)

        self.select_scope_set: Set[str] = set()
        self.order_first_set: Set[str] = set()
        self.is_use_spot: bool = False  # 是否包含现货策略
        self.is_day_period: bool = False  # 是否是日盘，否则是小时盘
        self.is_hour_period: bool = False  # 是否是小时盘，否则是日盘
        self.factor_params_dict: Dict[str, set] = {}
        self.factor_col_name_list: List[str] = []
        self.hold_period: str = '1h'  # 最大的持仓周期，默认值设置为最小

        # 策略列表，包含每个策略的详细配置
        self.strategy: Optional[StrategyConfig] = None
        self.strategy_raw: Optional[dict] = None
        # 空头策略列表，包含每个策略的详细配置
        self.strategy_short: Optional[StrategyConfig] = None
        self.strategy_short_raw: Optional[dict] = None
        # 策略评价
        self.report: Optional[pd.DataFrame] = None

        # 遍历标记
        self.iter_round: int | str = 0  # 遍历的INDEX，0表示非遍历场景，从1、2、3、4、...开始表示是第几个循环，当然也可以赋值为具体名称

    def __repr__(self):
        return f"""{'+' * 56}
# {self.name} 配置信息如下：
+ 回测时间: {self.start_date} ~ {self.end_date}
+ 手续费: 合约{self.swap_c_rate * 100:.2f}%，现货{self.spot_c_rate * 100:.2f}%
+ 杠杆: {self.leverage:.2f}
+ 最小K线数量: {self.min_kline_num}
+ 拉黑名单: {self.black_list}
+ 策略配置如下:
{self.strategy}
{self.strategy_short if self.strategy_short is not None else ''}
{'+' * 56}
"""

    @property
    def hold_period_type(self):
        return 'D' if self.is_day_period else 'H'

    def info(self):
        # 输出一下配置信息
        print(self)

    def get_fullname(self, as_folder_name=False):
        fullname_list = [self.name, f"{self.strategy.get_fullname(as_folder_name)}"]

        fullname = ' '.join(fullname_list)
        md5_hash = hashlib.md5(fullname.encode('utf-8')).hexdigest()
        # print(fullname, md5_hash)
        return f'{self.name}-{md5_hash[:8]}' if as_folder_name else fullname

    def load_strategy_config(self, strategy_dict: dict, is_short=False):
        if is_short:
            self.strategy_short_raw = strategy_dict
        else:
            self.strategy_raw = strategy_dict

        strategy_cfg = StrategyConfig.init(**strategy_dict)

        if strategy_cfg.is_day_period:
            self.is_day_period = True
        else:
            self.is_hour_period = True

        # 缓存持仓周期的事情
        self.hold_period = strategy_cfg.hold_period.lower()

        self.is_use_spot = strategy_cfg.is_use_spot

        self.select_scope_set.add(strategy_cfg.select_scope)
        self.order_first_set.add(strategy_cfg.order_first)
        if not {'spot', 'mix'}.isdisjoint(self.select_scope_set) and self.leverage >= 2:
            print(f'现货策略不支持杠杆大于等于2的情况，请重新配置')
            exit(1)

        if strategy_cfg.long_select_coin_num == 0 and (strategy_cfg.short_select_coin_num == 0 or
                                                       strategy_cfg.short_select_coin_num == 'long_nums'):
            print('❌ 策略中的选股数量都为0，忽略此策略配置')
            exit(1)
        if is_short:
            self.strategy_short = strategy_cfg
        else:
            self.strategy = strategy_cfg
        self.factor_col_name_list += strategy_cfg.factor_columns

        # 针对当前策略的因子信息，整理之后的列名信息，并且缓存到全局
        for factor_config in strategy_cfg.all_factors:
            # 添加到并行计算的缓存中
            if factor_config.name not in self.factor_params_dict:
                self.factor_params_dict[factor_config.name] = set()
            self.factor_params_dict[factor_config.name].add(factor_config.param)

        self.factor_col_name_list = list(set(self.factor_col_name_list))

    @classmethod
    def init_from_config(cls, load_strategy_list: bool = True) -> "BacktestConfig":
        import config

        backtest_config = cls(
            config.backtest_name,
            start_date=config.start_date,  # 回测开始时间
            end_date=config.end_date,  # 回测结束时间
            # ** 交易配置 **
            initial_usdt=config.initial_usdt,  # 初始usdt
            leverage=config.leverage,  # 杠杆
            swap_c_rate=config.swap_c_rate,  # 合约买入手续费
            spot_c_rate=config.spot_c_rate,  # 现货买卖手续费
            # ** 数据参数 **
            black_list=config.black_list,  # 拉黑名单
            min_kline_num=config.min_kline_num,  # 最小K线数量，k线数量少于这个数字的部分不会计入计算
        )

        # ** 策略配置 **
        # 初始化策略，默认都是需要初始化的
        if load_strategy_list:
            backtest_config.load_strategy_config(config.strategy)
            if strategy_short := getattr(config, "strategy_short", None):
                backtest_config.load_strategy_config(strategy_short, is_short=True)

        return backtest_config

    def set_report(self, report: pd.DataFrame):
        report['param'] = self.get_fullname()
        self.report = report

    def get_result_folder(self) -> Path:
        if self.iter_round == 0:
            return get_folder_path(backtest_path, self.name, path_type=True)
        else:
            return get_folder_path(
                get_folder_path('data', '遍历结果'),
                self.name,
                f'参数组合_{self.iter_round}' if isinstance(self.iter_round, int) else self.iter_round,
                path_type=True
            )

    def get_strategy_config_sheet(self, with_factors=True) -> dict:
        factor_dict = {'hold_period': self.strategy.hold_period}
        ret = {
            '策略': self.name,
            'fullname': self.get_fullname(),
        }
        if with_factors:
            for factor_config in self.strategy.all_factors:
                _name = f'#FACTOR-{factor_config.name}'
                _val = factor_config.param
                factor_dict[_name] = _val
            ret.update(**factor_dict)

        return ret


class BacktestConfigFactory:
    """
    遍历参数的时候，动态生成配置
    """

    def __init__(self):
        # ====================================================================================================
        # ** 参数遍历配置 **
        # 可以指定因子遍历的参数范围
        # ====================================================================================================
        # 存储生成好的config list和strategy list
        self.config_list: List[BacktestConfig] = []

    @property
    def result_folder(self) -> Path:
        return get_folder_path('data', '遍历结果', backtest_name, path_type=True)

    def generate_all_factor_config(self):
        """
        产生一个conf，拥有所有策略的因子，用于因子加速并行计算
        """
        import config
        backtest_config = BacktestConfig.init_from_config(load_strategy_list=False)
        factor_list = set()
        filter_list = set()
        for conf in self.config_list:
            factor_list |= set(conf.strategy.factor_list)
            filter_list |= set(conf.strategy.filter_list)

        strategy_all = {k: v for k, v in config.strategy.items() if not k.endswith(('factor_list', 'filter_list'))}
        strategy_all['factor_list'] = list(factor_list)
        strategy_all['filter_list'] = list(filter_list)

        backtest_config.load_strategy_config(strategy_all)
        return backtest_config

    def get_name_params_sheet(self) -> pd.DataFrame:
        rows = []
        for config in self.config_list:
            rows.append(config.get_strategy_config_sheet())

        sheet = pd.DataFrame(rows)
        sheet.to_excel(self.config_list[-1].get_result_folder().parent / '策略回测参数总表.xlsx', index=False)
        return sheet

    def generate_by_strategies(self, strategies) -> List[BacktestConfig]:
        config_list = []
        iter_round = 0

        for strategy in strategies:
            iter_round += 1
            backtest_config = BacktestConfig.init_from_config(load_strategy_list=False)
            backtest_config.load_strategy_config(strategy)
            backtest_config.iter_round = iter_round

            config_list.append(backtest_config)

        self.config_list = config_list

        return config_list


def load_config() -> BacktestConfig:
    """
    config.py中的配置信息加载到回测系统中
    :return: 初始化之后的配置信息
    """
    # 从配置文件中读取并初始化回测配置
    conf = BacktestConfig.init_from_config()

    # 配置信息打印
    conf.info()

    return conf


def create_factory(strategies):
    factory = BacktestConfigFactory()
    factory.generate_by_strategies(strategies)

    return factory
