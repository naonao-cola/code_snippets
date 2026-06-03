"""
邢不行™️选币框架
Python数字货币量化投资课程

版权所有 ©️ 邢不行
微信: xbx8662

未经授权，不得复制、修改、或使用本代码的全部或部分内容。仅限个人学习用途，禁止商业用途。

Author: 邢不行
"""

import numpy as np
import numba as nb
from numba.experimental import jitclass


@jitclass
class RebAlways:
    spot_lot_sizes: nb.float64[:]  # 每手币数，表示一手加密货币中包含的币数
    swap_lot_sizes: nb.float64[:]

    def __init__(self, spot_lot_sizes, swap_lot_sizes):
        n_syms_spot = len(spot_lot_sizes)
        n_syms_swap = len(swap_lot_sizes)

        self.spot_lot_sizes = np.zeros(n_syms_spot, dtype=np.float64)
        self.spot_lot_sizes[:] = spot_lot_sizes

        self.swap_lot_sizes = np.zeros(n_syms_swap, dtype=np.float64)
        self.swap_lot_sizes[:] = swap_lot_sizes

    def _calc(self, equity, prices, ratios, lot_sizes):
        # 初始化目标持仓手数
        target_lots = np.zeros(len(lot_sizes), dtype=np.int64)

        # 每个币分配的资金(带方向)
        symbol_equity = equity * ratios

        # 分配资金大于 0.01U 则认为是有效持仓
        mask = np.abs(symbol_equity) > 0.01

        # 为有效持仓分配仓位
        target_lots[mask] = (symbol_equity[mask] / prices[mask] / lot_sizes[mask]).astype(np.int64)

        return target_lots

    def calc_lots(self, equity, spot_prices, spot_lots, spot_ratios, swap_prices, swap_lots, swap_ratios):
        """
        计算每个币种的目标手数
        :param equity: 总权益
        :param spot_prices: 现货最新价格
        :param spot_lots: 现货当前持仓手数
        :param spot_ratios: 现货币种的资金比例
        :param swap_prices: 合约最新价格
        :param swap_lots: 合约当前持仓手数
        :param swap_ratios: 合约币种的资金比例
        :return: tuple[现货目标手数, 合约目标手数]
        """
        is_spot_only = False

        # 合约总权重小于极小值，认为是纯现货模式
        if np.sum(np.abs(swap_ratios)) < 1e-6:
            is_spot_only = True
            equity *= 0.99  # 纯现货留 1% 的资金作为缓冲            

        # 现货目标持仓手数
        spot_target_lots = self._calc(equity, spot_prices, spot_ratios, self.spot_lot_sizes)

        if is_spot_only:
            swap_target_lots = np.zeros(len(self.swap_lot_sizes), dtype=np.int64)
            return spot_target_lots, swap_target_lots

        # 合约目标持仓手数
        swap_target_lots = self._calc(equity, swap_prices, swap_ratios, self.swap_lot_sizes)

        return spot_target_lots, swap_target_lots
