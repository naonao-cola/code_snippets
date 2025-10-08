"""
邢不行™️选币框架
Python数字货币量化投资课程

版权所有 ©️ 邢不行
微信: xbx8662

未经授权，不得复制、修改、或使用本代码的全部或部分内容。仅限个人学习用途，禁止商业用途。

Author: 邢不行
"""
import numba as nb
import numpy as np
from numba.experimental import jitclass

"""
# 新语法小讲堂
通过操作对象的值而不是更换reference，来保证所有引用的位置都能同步更新。

`self.target_lots[:] = target_lots`
这个写法涉及 Python 中的切片（slice）操作和对象的属性赋值。

`target_lots: nb.int64[:]  # 目标持仓手数`，self.target_lots 是一个列表，`[:]` 是切片操作符，表示对整个列表进行切片。

### 详细解释：

1. **`self.target_lots[:] = target_lots`**:
   - `self.target_lots` 是对象的一个属性，通常是一个列表（或者其它支持切片操作的可变序列）。
   - `[:]` 是切片操作符，表示对整个列表进行切片。具体来说，`[:]` 是对列表的所有元素进行选择，这种写法可以用于复制列表或对整个列表内容进行替换。

2. **具体操作**：
   - `self.target_lots[:] = target_lots` 不是直接将 `target_lots` 赋值给 `self.target_lots`，而是将 `target_lots` 中的所有元素替换 `self.target_lots` 中的所有元素。
   - 这种做法的一个好处是不会改变 `self.target_lots` 对象的引用，而是修改它的内容。这在有其他对象引用 `self.target_lots` 时非常有用，确保所有引用者看到的列表内容都被更新，而不会因为重新赋值而改变列表的引用。

### 举个例子：

```python
a = [1, 2, 3]
b = a
a[:] = [4, 5, 6]  # 只改变列表内容，不改变引用

print(a)  # 输出: [4, 5, 6]
print(b)  # 输出: [4, 5, 6]，因为 a 和 b 引用的是同一个列表，修改 a 的内容也影响了 b
```

如果直接用 `a = [4, 5, 6]` 替换 `[:]` 操作，那么 `b` 就不会受到影响，因为 `a` 重新指向了一个新的列表对象。
"""


@jitclass
class Simulator:
    equity: float  # 账户权益, 单位 USDT
    fee_rate: float  # 手续费/交易成本
    min_order_limit: float  # 最小下单金额

    lot_sizes: nb.float64[:]  # 每手币数，表示一手加密货币中包含的币数
    lots: nb.int64[:]  # 当前持仓手数
    target_lots: nb.int64[:]  # 目标持仓手数

    last_prices: nb.float64[:]  # 最新价格
    has_last_prices: bool  # 是否有最新价

    def __init__(self, init_capital, lot_sizes, fee_rate, init_lots, min_order_limit):
        """
        初始化
        :param init_capital: 初始资金 
        :param lot_sizes: 每个币种的最小下单量
        :param fee_rate: 手续费率
        :param init_lots: 初始持仓
        :param min_order_limit: 最小下单金额
        """
        self.equity = init_capital  # 账户权益
        self.fee_rate = fee_rate  # 交易成本
        self.min_order_limit = min_order_limit  # 最小下单金额

        n = len(lot_sizes)

        # 合约面值
        self.lot_sizes = np.zeros(n, dtype=np.float64)
        self.lot_sizes[:] = lot_sizes

        # 前收盘价
        self.last_prices = np.zeros(n, dtype=np.float64)
        self.has_last_prices = False

        # 当前持仓手数
        self.lots = np.zeros(n, dtype=np.int64)
        self.lots[:] = init_lots

        # 目标持仓手数
        self.target_lots = np.zeros(n, dtype=np.int64)
        self.target_lots[:] = init_lots

    def set_target_lots(self, target_lots):
        self.target_lots[:] = target_lots

    def fill_last_prices(self, prices):
        mask = np.logical_not(np.isnan(prices))
        self.last_prices[mask] = prices[mask]
        self.has_last_prices = True

    def settle_equity(self, prices):
        """
        结算当前账户权益
        :param prices: 当前价格
        :return:
        """
        mask = np.logical_and(self.lots != 0, np.logical_not(np.isnan(prices)))
        # 计算公式：
        # 1. 净值涨跌 = (最新价格 - 前最新价（前收盘价）) * 持币数量。
        # 2. 其中，持币数量 = min_qty * 持仓手数。
        # 3. 所有币种对应的净值涨跌累加起来
        equity_delta = np.sum((prices[mask] - self.last_prices[mask]) * self.lot_sizes[mask] * self.lots[mask])

        # 反映到净值上
        self.equity += equity_delta

    def on_open(self, open_prices, funding_rates, mark_prices):
        """
        模拟: K 线开盘 -> K 线收盘时刻
        :param open_prices: 开盘价
        :param funding_rates: 资金费
        :param mark_prices: 计算资金费的标记价格（目前就用开盘价来）
        :return:
        """
        if not self.has_last_prices:
            self.fill_last_prices(open_prices)

        # 根据开盘价和前最新价（前收盘价），结算当前账户权益
        self.settle_equity(open_prices)

        # 根据标记价格和资金费率，结算资金费盈亏
        mask = np.logical_and(self.lots != 0, np.logical_not(np.isnan(mark_prices)))
        pos_val = notional_value = self.lot_sizes[mask] * self.lots[mask] * mark_prices[mask]
        funding_fee = np.sum(notional_value * funding_rates[mask])
        self.equity -= funding_fee

        # 最新价为开盘价
        self.fill_last_prices(open_prices)

        # 返回扣除资金费后开盘账户权益、资金费和带方向的仓位名义价值
        return self.equity, funding_fee, pos_val

    def on_execution(self, exec_prices):
        """
        模拟: K 线开盘时刻 -> 调仓时刻
        :param exec_prices:  执行价格
        :return:            调仓后的账户权益、调仓后的仓位名义价值
        """
        if not self.has_last_prices:
            self.fill_last_prices(exec_prices)

        # 根据调仓价和前最新价（开盘价），结算当前账户权益
        self.settle_equity(exec_prices)

        # 计算需要买入或卖出的合约数量
        delta = self.target_lots - self.lots
        mask = np.logical_and(delta != 0, np.logical_not(np.isnan(exec_prices)))

        # 计算成交额
        turnover = np.zeros(len(self.lot_sizes), dtype=np.float64)
        turnover[mask] = np.abs(delta[mask]) * self.lot_sizes[mask] * exec_prices[mask]

        # 成交额小于 min_order_limit 则无法调仓
        mask = np.logical_and(mask, turnover >= self.min_order_limit)

        # 本期调仓总成交额
        turnover_total = turnover[mask].sum()

        if np.isnan(turnover_total):
            raise RuntimeError('Turnover is nan')

        # 根据总成交额计算并扣除手续费
        fee = turnover_total * self.fee_rate
        self.equity -= fee

        # 更新已成功调仓的 symbol 持仓
        self.lots[mask] = self.target_lots[mask]

        # 最新价为调仓价
        self.fill_last_prices(exec_prices)

        # 返回扣除手续费的调仓后账户权益，成交额，和手续费
        return self.equity, turnover_total, fee

    def on_close(self, close_prices):
        """
        模拟: K 线收盘 -> K 线收盘时刻
        :param close_prices: 收盘价
        :return:           收盘后的账户权益
        """
        if not self.has_last_prices:
            self.fill_last_prices(close_prices)

        # 模拟: 调仓时刻 -> K 线收盘时刻

        # 根据收盘价和前最新价（调仓价），结算当前账户权益
        self.settle_equity(close_prices)

        # 最新价为收盘价
        self.fill_last_prices(close_prices)

        mask = np.logical_and(self.lots != 0, np.logical_not(np.isnan(close_prices)))
        pos_val = self.lot_sizes[mask] * self.lots[mask] * close_prices[mask]

        # 返回收盘账户权益
        return self.equity, pos_val
