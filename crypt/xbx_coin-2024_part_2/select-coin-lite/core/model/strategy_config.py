"""
邢不行™️选币框架
Python数字货币量化投资课程

版权所有 ©️ 邢不行
微信: xbx8662

未经授权，不得复制、修改、或使用本代码的全部或部分内容。仅限个人学习用途，禁止商业用途。

Author: 邢不行
"""
import hashlib
import re
from dataclasses import dataclass
from functools import cached_property
from typing import List, Tuple

import numpy as np
import pandas as pd

from config import is_pure_long


def filter_series_by_range(series, range_str):
    # 提取运算符和数值
    operator = range_str[:2] if range_str[:2] in ['>=', '<=', '==', '!='] else range_str[0]
    value = float(range_str[len(operator):])

    match operator:
        case '>=':
            return series >= value
        case '<=':
            return series <= value
        case '==':
            return series == value
        case '!=':
            return series != value
        case '>':
            return series > value
        case '<':
            return series < value
        case _:
            raise ValueError(f"Unsupported operator: {operator}")


@dataclass(frozen=True)
class FactorConfig:
    name: str = 'Bias'  # 选币因子名称
    is_sort_asc: bool = True  # 是否正排序
    param: int = 3  # 选币因子参数
    weight: float = 1  # 选币因子权重

    @classmethod
    def parse_config_list(cls, config_list: List[tuple]):
        all_long_factor_weight = sum([factor[3] for factor in config_list])
        factor_list = []
        for factor_name, is_sort_asc, parameter_list, weight in config_list:
            new_weight = weight / all_long_factor_weight
            factor_list.append(cls(name=factor_name, is_sort_asc=is_sort_asc, param=parameter_list, weight=new_weight))
        return factor_list

    @cached_property
    def col_name(self):
        return f'{self.name}_{str(self.param)}'

    def __repr__(self):
        return f'{self.col_name}{"↑" if self.is_sort_asc else "↓"}权重:{self.weight}'

    def to_tuple(self):
        return self.name, self.is_sort_asc, self.param, self.weight


@dataclass(frozen=True)
class FilterMethod:
    how: str = ''  # 过滤方式
    range: str = ''  # 过滤值

    def __repr__(self):
        match self.how:
            case 'rank':
                name = '排名'
            case 'pct':
                name = '百分比'
            case 'val':
                name = '数值'
            case _:
                raise ValueError(f'不支持的过滤方式：`{self.how}`')

        return f'{name}:{self.range}'

    def to_val(self):
        return f'{self.how}:{self.range}'


@dataclass(frozen=True)
class FilterFactorConfig:
    name: str = 'Bias'  # 选币因子名称
    param: int = 3  # 选币因子参数
    method: FilterMethod = None  # 过滤方式
    is_sort_asc: bool = True  # 是否正排序

    def __repr__(self):
        _repr = self.col_name
        if self.method:
            _repr += f'{"↑" if self.is_sort_asc else "↓"}{self.method}'
        return _repr

    @cached_property
    def col_name(self):
        return f'{self.name}_{str(self.param)}'

    @classmethod
    def init(cls, filter_factor: tuple):
        # 仔细看，结合class的默认值，这个和默认策略中使用的过滤是一模一样的
        config = dict(name=filter_factor[0], param=filter_factor[1])
        if len(filter_factor) > 2:
            # 可以自定义过滤方式
            _how, _range = re.sub(r'\s+', '', filter_factor[2]).split(':')
            config['method'] = FilterMethod(how=_how, range=_range)
        if len(filter_factor) > 3:
            # 可以自定义排序
            config['is_sort_asc'] = filter_factor[3]
        return cls(**config)

    def to_tuple(self, full_mode=False):
        if full_mode:
            return self.name, self.param, self.method.to_val(), self.is_sort_asc
        else:
            return self.name, self.param


def calc_factor_common(df, factor_list: List[FactorConfig]):
    factor_val = np.zeros(df.shape[0])
    for factor_config in factor_list:
        col_name = f'{factor_config.name}_{str(factor_config.param)}'
        # 计算单个因子的排名
        _rank = df.groupby('candle_begin_time')[col_name].rank(ascending=factor_config.is_sort_asc, method='min')
        # 将因子按照权重累加
        factor_val += _rank * factor_config.weight
    return factor_val


def filter_common(df, filter_list):
    condition = pd.Series(True, index=df.index)

    for filter_config in filter_list:
        col_name = f'{filter_config.name}_{str(filter_config.param)}'
        match filter_config.method.how:
            case 'rank':
                rank = df.groupby('candle_begin_time')[col_name].rank(ascending=filter_config.is_sort_asc, pct=False)
                condition = condition & filter_series_by_range(rank, filter_config.method.range)
            case 'pct':
                rank = df.groupby('candle_begin_time')[col_name].rank(ascending=filter_config.is_sort_asc, pct=True)
                condition = condition & filter_series_by_range(rank, filter_config.method.range)
            case 'val':
                condition = condition & filter_series_by_range(df[col_name], filter_config.method.range)
            case _:
                raise ValueError(f'不支持的过滤方式：{filter_config.method.how}')

    return condition


@dataclass
class StrategyConfig:
    name: str = 'Strategy'
    strategy: str = 'Strategy'

    # 持仓周期。目前回测支持日线级别、小时级别。例：1H，6H，3D，7D......
    # 当持仓周期为D时，选币指标也是按照每天一根K线进行计算。
    # 当持仓周期为H时，选币指标也是按照每小时一根K线进行计算。
    hold_period: str = '1D'.replace('h', 'H').replace('d', 'D')

    # 配置offset
    offset_list: List[int] = (0,)

    # 是否使用现货
    is_use_spot: bool = False  # True：使用现货。False：不使用现货，只使用合约。

    # 多头选币数量。1 表示做多一个币; 0.1 表示做多10%的币
    long_select_coin_num: int | float = 0.1
    # 空头选币数量。1 表示做空一个币; 0.1 表示做空10%的币，'long_nums'表示和多头一样多的数量
    short_select_coin_num: int | float | str = 'long_nums'  # 注意：多头为0的时候，不能配置'long_nums'

    # 因子列名。
    factor_name: str = '因子'  # 因子：表示使用复合因子，默认是 factor_list 里面的因子组合。需要修改 calc_factor 函数配合使用

    # 选币因子信息列表，用于`2_选币_单offset.py`，`3_计算多offset资金曲线.py`共用计算资金曲线
    factor_list: List[FactorConfig] = ()  # 因子名（和factors文件中相同），排序方式，参数，权重。

    # 确认过滤因子及其参数，用于`2_选币_单offset.py`进行过滤
    filter_list: List[FilterFactorConfig] = ()  # 因子名（和factors文件中相同），参数

    @cached_property
    def is_day_period(self):
        return self.hold_period.endswith('D')

    @cached_property
    def is_hour_period(self):
        return self.hold_period.endswith('H')

    @cached_property
    def period_num(self) -> int:
        return int(self.hold_period.upper().replace('H', '').replace('D', ''))

    @cached_property
    def period_type(self) -> str:
        return self.hold_period[-1]

    @cached_property
    def factor_columns(self) -> List[str]:
        factor_columns = set()  # 去重

        # 针对当前策略的因子信息，整理之后的列名信息，并且缓存到全局
        for factor_config in self.factor_list:
            # 策略因子最终在df中的列名
            factor_columns.add(factor_config.col_name)  # 添加到当前策略缓存信息中

        # 针对当前策略的过滤因子信息，整理之后的列名信息，并且缓存到全局
        for filter_factor in self.filter_list:
            # 策略过滤因子最终在df中的列名
            factor_columns.add(filter_factor.col_name)  # 添加到当前策略缓存信息中

        return list(factor_columns)

    @cached_property
    def all_factors(self) -> set:
        return set(self.factor_list + self.filter_list)

    @classmethod
    def init(cls, **config):
        # 自动补充因子列表
        config['long_select_coin_num'] = config.get('long_select_coin_num', 0.1)
        config['short_select_coin_num'] = config.get('short_select_coin_num', 'long_nums')

        # 初始化策略因子
        config['factor_list'] = FactorConfig.parse_config_list(config.get('factor_list', []))

        # 初始化过滤因子
        config['filter_list'] = [FilterFactorConfig.init(filter_config) for filter_config in
                                 config.get('filter_list', [])]

        # 开始初始化策略对象
        return cls(**config)

    def get_fullname(self, as_folder_name=False):
        factor_desc_list = [f'{self.factor_list}', f'过滤{self.filter_list}']
        factor_desc = '&'.join(factor_desc_list)

        # ** 回测特有 ** 因为需要计算hash，因此包含的信息不同
        fullname = f"""{self.name}-{self.hold_period}-多头数量:{self.long_select_coin_num}-空头数量:{self.short_select_coin_num}-因子:{factor_desc}"""

        md5_hash = hashlib.md5(f'{fullname}-{self.offset_list}'.encode('utf-8')).hexdigest()
        return f'{self.name}-{md5_hash[:8]}' if as_folder_name else fullname

    def __repr__(self):
        return f"""- 策略名称: {self.name}
- 持仓周期: {self.hold_period}
- offset功能: {'开启' if len(self.offset_list) > 1 else '关闭'}
- 运行模式: {'纯多模式' if is_pure_long else '多空模式'}
- 选币数量: {self.long_select_coin_num}
- 策略因子: {self.factor_list}
- 过滤因子: {self.filter_list}"""

    def calc_factor(self, df, **kwargs) -> pd.DataFrame:
        raise NotImplementedError

    def calc_select_factor(self, df) -> pd.DataFrame:
        # 计算多头因子
        new_cols = {self.factor_name: calc_factor_common(df, self.factor_list)}

        return pd.DataFrame(new_cols, index=df.index)

    def before_filter(self, df, **kwargs) -> (pd.DataFrame, pd.DataFrame):
        raise NotImplementedError

    def filter_before_select(self, df):
        filter_condition = filter_common(df, self.filter_list)

        return df[filter_condition].copy(), df[filter_condition].copy()

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def after_merge_index(self, candle_df, symbol, factor_dict, data_dict) -> Tuple[pd.DataFrame, dict, dict]:
        return candle_df, factor_dict, data_dict
