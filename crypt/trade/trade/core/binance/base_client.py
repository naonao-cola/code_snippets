"""
邢不行™️选币实盘框架
Python数字货币量化投资课程

版权所有 ©️ 邢不行
微信: xbx8662

未经授权，不得复制、修改、或使用本代码的全部或部分内容。仅限个人学习用途，禁止商业用途。

Author: 邢不行
"""
# ==================================================================================================
# !!! 前置非常重要说明
# !!! 前置非常重要说明
# !!! 前置非常重要说明
# ---------------------------------------------------------------------------------------------------
# ** 方法名前缀规范 **
# 1. load_* 从硬盘获取数据
# 2. fetch_* 从接口获取数据
# 3. get_* 从对象获取数据，可能从硬盘，也可能从接口
# ====================================================================================================

import math
import time
import traceback

import ccxt
import numpy as np
import pandas as pd

from config import exchange_basic_config, utc_offset, stable_symbol
from core.utils.commons import apply_precision
from core.utils.commons import retry_wrapper
from core.utils.dingding import send_wechat_work_msg, send_msg_for_order


# 现货接口
# sapi

# 合约接口
# dapi：普通账户，包含币本位交易
# fapi，普通账户，包含U本位交易

# 统一账户
# papi, um的接口：U本位合约
# papi, cm的接口：币本位合约
# papi, margin：现货API，全仓杠杆现货

class BinanceClient:
    diff_timestamp = 0
    constants = dict()

    market_info = {}  # 缓存市场信息，并且自动更新，全局共享
    common_exchange = ccxt.binance(exchange_basic_config)

    def __init__(self, **config):
        self.api_key: str = config.get('apiKey', '')
        self.secret: str = config.get('secret', '')

        self.order_money_limit: dict = {
            'spot': config.get('spot_order_money_limit', 10),
            'swap': config.get('swap_order_money_limit', 5),
        }

        self.exchange = ccxt.binance(config.get('exchange_config', exchange_basic_config))
        self.wechat_webhook_url: str = config.get('wechat_webhook_url', '')

        self.swap_account = None

        self.coin_margin: dict = config.get('coin_margin', {})  # 用做保证金的币种

    # ====================================================================================================
    # ** 市场信息 **
    # ====================================================================================================
    def _fetch_swap_exchange_info_list(self) -> list:
        exchange_info = retry_wrapper(self.exchange.fapipublic_get_exchangeinfo, func_name='获取BN合约币种规则数据')
        return exchange_info['symbols']

    def _fetch_spot_exchange_info_list(self) -> list:
        exchange_info = retry_wrapper(self.exchange.public_get_exchangeinfo, func_name='获取BN现货币种规则数据')
        return exchange_info['symbols']

    # region 市场信息数据获取
    def fetch_market_info(self, symbol_type='swap', quote_symbol='USDT'):
        """
        加载市场数据
        :param symbol_type: 币种信息。swap为合约，spot为现货
        :param quote_symbol: 报价币种
        :return:
            symbol_list     交易对列表
            price_precision 币种价格精     例： 2 代表 0.01
                {'BTCUSD_PERP': 1, 'BTCUSD_231229': 1, 'BTCUSD_240329': 1, 'BTCUSD_240628': 1, ...}
            min_notional    最小下单金额    例： 5.0 代表 最小下单金额是5U
                {'BTCUSDT': 5.0, 'ETHUSDT': 5.0, 'BCHUSDT': 5.0, 'XRPUSDT': 5.0...}
        """
        print(f'🔄更新{symbol_type}市场数据...')
        # ===获取所有币种信息
        if symbol_type == 'swap':  # 合约
            exchange_info_list = self._fetch_swap_exchange_info_list()
        else:  # 现货
            exchange_info_list = self._fetch_spot_exchange_info_list()

        # ===获取币种列表
        symbol_list = []  # 如果是合约，只包含永续合约。如果是现货，包含所有数据
        full_symbol_list = []  # 包含所有币种信息

        # ===获取各个交易对的精度、下单量等信息
        min_qty = {}  # 最小下单精度，例如bnb，一次最少买入0.001个
        price_precision = {}  # 币种价格精，例如bnb，价格是158.887，不能是158.8869
        min_notional = {}  # 最小下单金额，例如bnb，一次下单至少买入金额是5usdt
        # 遍历获得想要的数据
        for info in exchange_info_list:
            symbol = info['symbol']  # 交易对信息

            # 过滤掉非报价币对 ， 非交易币对
            if info['quoteAsset'] != quote_symbol or info['status'] != 'TRADING':
                continue

            full_symbol_list.append(symbol)  # 添加到全量信息中

            if (symbol_type == 'swap' and info['contractType'] != 'PERPETUAL') or info['baseAsset'] in stable_symbol:
                pass  # 获取合约的时候，非永续的symbol会被排除
            else:
                symbol_list.append(symbol)

            for _filter in info['filters']:  # 遍历获得想要的数据
                if _filter['filterType'] == 'PRICE_FILTER':  # 获取价格精度
                    price_precision[symbol] = int(math.log(float(_filter['tickSize']), 0.1))
                elif _filter['filterType'] == 'LOT_SIZE':  # 获取最小下单量
                    min_qty[symbol] = int(math.log(float(_filter['minQty']), 0.1))
                elif _filter['filterType'] == 'MIN_NOTIONAL' and symbol_type == 'swap':  # 合约的最小下单金额
                    min_notional[symbol] = float(_filter['notional'])
                elif _filter['filterType'] == 'NOTIONAL' and symbol_type == 'spot':  # 现货的最小下单金额
                    min_notional[symbol] = float(_filter['minNotional'])

        self.market_info[symbol_type] = {
            'symbol_list': symbol_list,  # 如果是合约，只包含永续合约。如果是现货，包含所有数据
            'full_symbol_list': full_symbol_list,  # 包含所有币种信息
            'min_qty': min_qty,
            'price_precision': price_precision,
            'min_notional': min_notional,
            'last_update': int(time.time())
        }
        return self.market_info[symbol_type]

    def get_market_info(self, symbol_type, expire_seconds: int = 3600 * 12, require_update: bool = False,
                        quote_symbol='USDT') -> dict:
        if require_update:  # 如果强制刷新的话，就当我们系统没有更新过
            last_update = 0
        else:
            last_update = self.market_info.get(symbol_type, {}).get('last_update', 0)
        if last_update + expire_seconds < int(time.time()):
            self.fetch_market_info(symbol_type, quote_symbol)

        return self.market_info[symbol_type]

    # endregion

    # ====================================================================================================
    # ** 行情数据获取 **
    # ====================================================================================================
    # region 行情数据获取
    """K线数据获取"""

    def get_candle_df(self, symbol, run_time, limit=1500, interval='1h', symbol_type='swap') -> pd.DataFrame:
        # ===获取K线数据
        _limit = limit
        # 定义请求的参数：现货最大1000，合约最大499。
        if limit > 1000:  # 如果参数大于1000
            if symbol_type == 'spot':  # 如果是现货，最大设置1000
                _limit = 1000
            else:  # 如果不是现货，那就设置499
                _limit = 499
        # limit = 1000 if limit > 1000 and symbol_type == 'spot' else limit  # 现货最多获取1000根K
        # 计算获取k线的开始时间
        start_time_dt = run_time - pd.to_timedelta(interval) * limit

        df_list = []  # 定义获取的k线数据
        data_len = 0  # 记录数据长度
        params = {
            'symbol': symbol,  # 获取币种
            'interval': interval,  # 获取k线周期
            'limit': _limit,  # 获取多少根
            'startTime': int(time.mktime(start_time_dt.timetuple())) * 1000  # 获取币种开始时间
        }
        while True:
            # 获取指定币种的k线数据
            try:
                if symbol_type == 'swap':
                    kline = retry_wrapper(
                        self.exchange.fapipublic_get_klines, params=params, func_name='获取币种K线',
                        if_exit=False
                    )
                else:
                    kline = retry_wrapper(
                        self.exchange.public_get_klines, params=params, func_name='获取币种K线',
                        if_exit=False
                    )
            except Exception as e:
                print(e)
                print(traceback.format_exc())
                # 如果获取k线重试出错，直接返回，当前币种不参与交易
                return pd.DataFrame()

            # ===整理数据
            # 将数据转换为DataFrame
            df = pd.DataFrame(kline, dtype='float')
            if df.empty:
                break
            # 对字段进行重命名，字段对应数据可以查询文档（https://binance-docs.github.io/apidocs/futures/cn/#k）
            columns = {0: 'candle_begin_time', 1: 'open', 2: 'high', 3: 'low', 4: 'close', 5: 'volume', 6: 'close_time',
                       7: 'quote_volume',
                       8: 'trade_num', 9: 'taker_buy_base_asset_volume', 10: 'taker_buy_quote_asset_volume',
                       11: 'ignore'}
            df.rename(columns=columns, inplace=True)
            df['candle_begin_time'] = pd.to_datetime(df['candle_begin_time'], unit='ms')
            df.sort_values(by=['candle_begin_time'], inplace=True)  # 排序

            # 数据追加
            df_list.append(df)
            data_len = data_len + df.shape[0] - 1

            # 判断请求的数据是否足够
            if data_len >= limit:
                break

            if params['startTime'] == int(df.iloc[-1]['candle_begin_time'].timestamp()) * 1000:
                break

            # 更新一下k线数据
            params['startTime'] = int(df.iloc[-1]['candle_begin_time'].timestamp()) * 1000
            # 下载太多的k线的时候，中间sleep一下
            time.sleep(0.1)

        if not df_list:
            return pd.DataFrame()

        all_df = pd.concat(df_list, ignore_index=True)
        all_df['symbol'] = symbol  # 添加symbol列
        all_df['symbol_type'] = symbol_type  # 添加类型字段
        all_df.sort_values(by=['candle_begin_time'], inplace=True)  # 排序
        all_df.drop_duplicates(subset=['candle_begin_time'], keep='last', inplace=True)  # 去重

        # 删除runtime那根未走完的k线数据（交易所有时候会返回这条数据）
        all_df = all_df[all_df['candle_begin_time'] + pd.Timedelta(hours=utc_offset) < run_time]
        all_df.reset_index(drop=True, inplace=True)

        return all_df

    """最新报价数据获取"""

    def fetch_ticker_price(self, symbol: str = None, symbol_type: str = 'swap') -> dict:
        params = {'symbol': symbol} if symbol else {}
        match symbol_type:
            case 'spot':
                api_func = self.exchange.public_get_ticker_price
                func_name = f'获取{symbol}现货的ticker数据' if symbol else '获取所有现货币种的ticker数据'
            case 'swap':
                api_func = self.exchange.fapipublic_get_ticker_price
                func_name = f'获取{symbol}合约的ticker数据' if symbol else '获取所有合约币种的ticker数据'
            case _:
                raise ValueError(f'未知的symbol_type：{symbol_type}')

        tickers = retry_wrapper(api_func, params=params, func_name=func_name)
        return tickers

    def fetch_spot_ticker_price(self, spot_symbol: str = None) -> dict:
        return self.fetch_ticker_price(spot_symbol, symbol_type='spot')

    def fetch_swap_ticker_price(self, swap_symbol: str = None) -> dict:
        return self.fetch_ticker_price(swap_symbol, symbol_type='swap')

    def get_spot_ticker_price_series(self) -> pd.Series:
        ticker_price_df = pd.DataFrame(self.fetch_ticker_price(symbol_type='spot'))
        ticker_price_df['price'] = pd.to_numeric(ticker_price_df['price'], errors='coerce')
        return ticker_price_df.set_index(['symbol'])['price']

    def get_swap_ticker_price_series(self) -> pd.Series:
        ticker_price_df = pd.DataFrame(self.fetch_ticker_price(symbol_type='swap'))
        ticker_price_df['price'] = pd.to_numeric(ticker_price_df['price'], errors='coerce')
        return ticker_price_df.set_index(['symbol'])['price']

    """盘口数据获取"""

    def fetch_book_ticker(self, symbol, symbol_type='swap') -> dict:
        if symbol_type == 'swap':
            # 获取合约的盘口数据
            swap_book_ticker_data = retry_wrapper(
                self.exchange.fapiPublicGetTickerBookTicker, params={'symbol': symbol}, func_name='获取合约盘口数据')
            return swap_book_ticker_data
        else:
            # 获取现货的盘口数据
            spot_book_ticker_data = retry_wrapper(
                self.exchange.publicGetTickerBookTicker, params={'symbol': symbol}, func_name='获取现货盘口数据'
            )
            return spot_book_ticker_data

    def fetch_spot_book_ticker(self, spot_symbol) -> dict:
        return self.fetch_book_ticker(spot_symbol, symbol_type='spot')

    def fetch_swap_book_ticker(self, swap_symbol) -> dict:
        return self.fetch_book_ticker(swap_symbol, symbol_type='swap')

    # endregion

    # ====================================================================================================
    # ** 资金费数据 **
    # ====================================================================================================
    def get_premium_index_df(self) -> pd.DataFrame:
        """
        获取币安的最新资金费数据
        """
        last_funding_df = retry_wrapper(self.exchange.fapipublic_get_premiumindex, func_name='获取最新的资金费数据')
        last_funding_df = pd.DataFrame(last_funding_df)

        last_funding_df['nextFundingTime'] = pd.to_numeric(last_funding_df['nextFundingTime'], errors='coerce')
        last_funding_df['time'] = pd.to_numeric(last_funding_df['time'], errors='coerce')

        last_funding_df['nextFundingTime'] = pd.to_datetime(last_funding_df['nextFundingTime'], unit='ms')
        last_funding_df['time'] = pd.to_datetime(last_funding_df['time'], unit='ms')
        last_funding_df = last_funding_df[['symbol', 'nextFundingTime', 'lastFundingRate']]  # 保留部分字段
        last_funding_df.rename(columns={'nextFundingTime': 'fundingTime', 'lastFundingRate': 'fundingRate'},
                               inplace=True)

        return last_funding_df

    def get_funding_rate_df(self, symbol, limit=1000) -> pd.DataFrame:
        """
        获取币安的历史资金费数据
        :param symbol: 币种名称
        :param limit: 请求获取多少条数据，最大1000
        """
        param = {'symbol': symbol, 'limit': limit}
        # 获取历史数据
        try:
            funding_df = retry_wrapper(
                self.exchange.fapipublic_get_fundingrate, params=param,
                func_name='获取合约历史资金费数据'
            )
        except Exception as e:
            print(e)
            return pd.DataFrame()
        funding_df = pd.DataFrame(funding_df)
        if funding_df.empty:
            return funding_df

        funding_df['fundingTime'] = pd.to_datetime(funding_df['fundingTime'].astype(float) // 1000 * 1000,
                                                   unit='ms')  # 时间戳内容含有一些纳秒数据需要处理
        funding_df.sort_values('fundingTime', inplace=True)

        return funding_df

    # ====================================================================================================
    # ** 账户设置 **
    # ====================================================================================================
    def fetch_transfer_history(self):
        raise NotImplementedError

    def set_single_side_position(self):
        raise NotImplementedError

    def set_multi_assets_margin(self):
        """
        检查是否开启了联合保证金模式
        """
        # 查询保证金模式
        pass

    def reset_max_leverage(self, max_leverage=5, coin_list=()):
        """
        重置一下页面最大杠杆
        :param max_leverage:    设置页面最大杠杆
        :param coin_list:       对指定币种进行调整页面杠杆
        """
        """
        重置一下页面最大杠杆
        :param exchange:        交易所对象，用于获取数据
        :param max_leverage:    设置页面最大杠杆
        :param coin_list:       对指定币种进行调整页面杠杆
        """
        # 获取账户持仓风险（这里有杠杆数据）
        account_info = self.get_swap_account()
        if account_info is None:
            print(f'ℹ️获取账户持仓风险数据为空')
            exit(1)

        position_risk = pd.DataFrame(account_info['positions'])  # 将数据转成DataFrame
        if len(coin_list) > 0:
            position_risk = position_risk[position_risk['symbol'].isin(coin_list)]  # 只对选币池中的币种进行调整页面杠杆
        position_risk.set_index('symbol', inplace=True)  # 将symbol设为index

        # 遍历每一个可以持仓的币种，修改页面最大杠杆
        for symbol, row in position_risk.iterrows():
            if int(row['leverage']) != max_leverage:
                reset_leverage_func = getattr(self.exchange, self.constants.get('reset_page_leverage_api'))
                # 设置杠杆
                retry_wrapper(
                    reset_leverage_func,
                    params={'symbol': symbol, 'leverage': max_leverage, 'timestamp': ''},
                    func_name='设置杠杆'
                )

    # ====================================================================================================
    # ** 交易函数 **
    # ====================================================================================================
    def cancel_all_spot_orders(self):
        # 现货撤单
        get_spot_open_orders_func = getattr(self.exchange, self.constants.get('get_spot_open_orders_api'))
        orders = retry_wrapper(
            get_spot_open_orders_func,
            params={'timestamp': ''}, func_name='查询现货所有挂单'
        )
        symbols = [_['symbol'] for _ in orders]
        symbols = list(set(symbols))
        cancel_spot_open_orders_func = getattr(self.exchange, self.constants.get('cancel_spot_open_orders_api'))
        for _ in symbols:
            retry_wrapper(
                cancel_spot_open_orders_func,
                params={'symbol': _, 'timestamp': ''}, func_name='取消现货挂单'
            )

    def cancel_all_swap_orders(self):
        # 合约撤单
        get_swap_open_orders_func = getattr(self.exchange, self.constants.get('get_swap_open_orders_api'))
        orders = retry_wrapper(
            get_swap_open_orders_func,
            params={'timestamp': ''}, func_name='查询U本位合约所有挂单'
        )
        symbols = [_['symbol'] for _ in orders]
        symbols = list(set(symbols))
        cancel_swap_open_orders_func = getattr(self.exchange, self.constants.get('cancel_swap_open_orders_api'))
        for _ in symbols:
            retry_wrapper(
                cancel_swap_open_orders_func,
                params={'symbol': _, 'timestamp': ''}, func_name='取消U本位合约挂单'
            )

    def prepare_order_params_list(
            self, orders_df: pd.DataFrame, symbol_type: str, symbol_ticker_price: pd.Series,
            slip_rate: float = 0.015) -> list:
        """
        根据策略产生的订单数据，构建每个币种的下单参数
        :param orders_df: 策略产生的订单信息
        :param symbol_type: 下单类型。spot/swap
        :param symbol_ticker_price: 每个币种最新价格
        :param slip_rate: 滑点
        :return: order_params_list 每个币种的下单参数
        """
        orders_df.sort_values('实际下单资金', ascending=True, inplace=True)
        orders_df.set_index('symbol', inplace=True)  # 重新设置index

        market_info = self.get_market_info(symbol_type)
        min_qty = market_info['min_qty']
        price_precision = market_info['price_precision']
        min_notional = market_info['min_notional']

        # 遍历symbol_order，构建每个币种的下单参数
        order_params_list = []
        for symbol, row in orders_df.iterrows():
            # ===若当前币种没有最小下单精度、或最小价格精度，报错
            if (symbol not in min_qty) or (symbol not in price_precision):
                # 报错
                print(f'❌当前币种{symbol}没有最小下单精度、或最小价格精度，币种信息异常')
                continue

            # ===计算下单量、方向、价格
            quantity = row['实际下单量']
            # 按照最小下单量对合约进行四舍五入，对现货就低不就高处理
            # 注意点：合约有reduceOnly参数可以超过你持有的持仓量，现货不行，只能卖的时候留一点点残渣
            quantity = round(quantity, min_qty[symbol]) if symbol_type == 'swap' else apply_precision(quantity,
                                                                                                      min_qty[symbol])
            # 计算下单方向、价格，并增加一定的滑点
            if quantity > 0:
                side = 'BUY'
                price = symbol_ticker_price[symbol] * (1 + slip_rate)
            elif quantity < 0:
                side = 'SELL'
                price = symbol_ticker_price[symbol] * (1 - slip_rate)
            else:
                print('⚠️下单量为0，不进行下单')
                continue
            # 下单量取绝对值
            quantity = abs(quantity)
            # 通过最小价格精度对下单价格进行四舍五入
            price = round(price, price_precision[symbol])

            # ===判断是否是清仓交易
            reduce_only = True if row['交易模式'] == '清仓' and symbol_type == 'swap' else False

            # ===判断交易金额是否小于最小下单金额（一般是5元），小于的跳过
            if quantity * price < min_notional.get(symbol, self.order_money_limit[symbol_type]):
                if not reduce_only:  # 清仓状态不跳过
                    print(f'⚠️ {symbol} 交易金额过小，跳过 (金额:{quantity * price:.2f}U < 最小:{min_notional.get(symbol, self.order_money_limit[symbol_type])}U)')
                    continue

            # ===构建下单参数
            price = f'{price:.{price_precision[symbol]}f}'  # 根据精度将价格转成str
            quantity = np.format_float_positional(quantity).rstrip('.')  # 解决科学计数法的问题
            order_params = {
                'symbol': symbol,
                'side': side,
                'type': 'LIMIT',
                'price': price,
                'quantity': quantity,
                'newClientOrderId': str(int(time.time())),
                'timeInForce': 'GTC',
                'reduceOnly': str(bool(reduce_only)),
                'timestamp': ''
            }
            # 如果是合约下单，添加进行下单列表中，放便后续批量下单
            order_params_list.append(order_params)
        return order_params_list

    def place_spot_orders_bulk(self, orders_df, slip_rate=0.015):
        symbol_last_price = self.get_spot_ticker_price_series()
        order_params_list = self.prepare_order_params_list(orders_df, 'spot', symbol_last_price, slip_rate)

        for order_param in order_params_list:
            del order_param['reduceOnly']  # 现货没有这个参数，进行移除
            self.place_spot_order(**order_param)

    def place_swap_orders_bulk(self, orders_df, slip_rate=0.015):
        symbol_last_price = self.get_swap_ticker_price_series()
        order_params_list = self.prepare_order_params_list(orders_df, 'swap', symbol_last_price, slip_rate)

        for order_params in order_params_list:
            self.place_swap_order(**order_params)

    def place_spot_order(self, symbol, side, quantity, price=None, **kwargs) -> dict:
        # 确定下单参数
        params = {
            'symbol': symbol,
            'side': side,
            'type': 'MARKET',
            'quantity': str(quantity),
            **kwargs
        }

        if price is not None:
            params['price'] = str(price)
            params['timeInForce'] = 'GTC'
            params['type'] = 'LIMIT'
            order_type = 'LIMIT'
            print(f'📈 现货{side} {symbol} 数量:{quantity} 价格:{price}')
        else:
            order_type = 'MARKET'
            print(f'📈 现货{side} {symbol} 数量:{quantity} (市价单)')

        try:
            # 下单
            order_res = retry_wrapper(
                self.exchange.private_post_order,
                params=params,
                func_name='现货下单'
            )
            order_id = order_res.get('orderId', 'N/A')
            status = order_res.get('status', 'N/A')
            print(f'✅ 现货下单成功 订单ID:{order_id} 状态:{status}')
        except Exception as e:
            print(f'❌ 现货下单失败 {symbol}: {e}')
            send_wechat_work_msg(
                f'现货 {symbol} 下单 {float(quantity) * float(price if price else 0)}U 出错，请查看程序日志',
                self.wechat_webhook_url
            )
            return {}
            # 发送下单结果到钉钉
        send_msg_for_order([params], [order_res], self.wechat_webhook_url)
        return order_res

    def place_swap_order(self, symbol, side, quantity, price=None, **kwargs) -> dict:
        # 确定下单参数
        params = {
            'symbol': symbol,
            'side': side,
            'type': 'MARKET',
            'quantity': str(quantity),
            **kwargs
        }

        if price is not None:
            params['price'] = str(price)
            params['timeInForce'] = 'GTC'
            params['type'] = 'LIMIT'
            order_type = 'LIMIT'
            print(f'🔄 合约{side} {symbol} 数量:{quantity} 价格:{price}')
        else:
            order_type = 'MARKET'
            print(f'🔄 合约{side} {symbol} 数量:{quantity} (市价单)')

        try:
            # 下单
            order_res = retry_wrapper(
                self.exchange.fapiprivate_post_order,
                params=params,
                func_name='U本位合约下单'
            )
            order_id = order_res.get('orderId', 'N/A')
            status = order_res.get('status', 'N/A')
            print(f'✅ 合约下单成功 订单ID:{order_id} 状态:{status}')
        except Exception as e:
            print(f'❌ 合约下单失败 {symbol}: {e}')
            send_wechat_work_msg(
                f'U本位合约 {symbol} 下单 {float(quantity) * float(price if price else 0)}U 出错，请查看程序日志',
                self.wechat_webhook_url
            )
            return {}
        send_msg_for_order([params], [order_res], self.wechat_webhook_url)
        return order_res

    def get_spot_position_df(self) -> pd.DataFrame:
        """
        获取账户净值


        :return:
            swap_equity=1000  (表示账户里资金总价值为 1000U )

        """
        # 获取U本位合约账户净值(不包含未实现盈亏)
        position_df = retry_wrapper(self.exchange.private_get_account, params={'timestamp': ''},
                                    func_name='获取现货账户净值')  # 获取账户净值
        position_df = pd.DataFrame(position_df['balances'])

        position_df['free'] = pd.to_numeric(position_df['free'])
        position_df['locked'] = pd.to_numeric(position_df['locked'])

        position_df['free'] += position_df['locked']
        position_df = position_df[position_df['free'] != 0]

        position_df.rename(columns={'asset': 'symbol', 'free': '当前持仓量'}, inplace=True)

        # 保留指定字段
        position_df = position_df[['symbol', '当前持仓量']]
        position_df['仓位价值'] = None  # 设置默认值

        return position_df

    # =====获取持仓
    # 获取币安账户的实际持仓
    def get_swap_position_df(self) -> pd.DataFrame:
        """
        获取币安账户的实际持仓

        :return:

                  当前持仓量   均价  持仓盈亏
        symbol
        RUNEUSDT       -82.0  1.208 -0.328000
        FTMUSDT        523.0  0.189  1.208156

        """
        # 获取原始数据
        get_swap_position_func = getattr(self.exchange, self.constants.get('get_swap_position_api'))
        position_df = retry_wrapper(get_swap_position_func, params={'timestamp': ''}, func_name='获取账户持仓风险')
        if position_df is None or len(position_df) == 0:
            return pd.DataFrame(columns=['symbol', '当前持仓量', '均价', '持仓盈亏', '当前标记价格', '仓位价值'])

        position_df = pd.DataFrame(position_df)  # 将原始数据转化为dataframe

        # 整理数据
        columns = {'positionAmt': '当前持仓量', 'entryPrice': '均价', 'unRealizedProfit': '持仓盈亏',
                   'markPrice': '当前标记价格'}
        position_df.rename(columns=columns, inplace=True)
        for col in columns.values():  # 转成数字
            position_df[col] = pd.to_numeric(position_df[col])

        position_df = position_df[position_df['当前持仓量'] != 0]  # 只保留有仓位的币种
        position_df.set_index('symbol', inplace=True)  # 将symbol设置为index
        position_df['仓位价值'] = position_df['当前持仓量'] * position_df['当前标记价格']

        # 保留指定字段
        position_df = position_df[['当前持仓量', '均价', '持仓盈亏', '当前标记价格', '仓位价值']]

        return position_df

    def update_swap_account(self) -> dict:
        self.swap_account = retry_wrapper(
            self.exchange.fapiprivatev2_get_account, params={'timestamp': ''},
            func_name='获取U本位合约账户信息'
        )
        return self.swap_account

    def get_swap_account(self, require_update: bool = False) -> dict:
        if self.swap_account is None or require_update:
            self.update_swap_account()
        return self.swap_account

    def get_account_overview(self):
        raise NotImplementedError

    def fetch_spot_trades(self, symbol, end_time) -> pd.DataFrame:
        # =设置获取订单时的参数
        params = {
            'symbol': symbol,  # 设置获取订单的币种
            'endTime': int(time.mktime(end_time.timetuple())) * 1000,  # 设置获取订单的截止时间
            'limit': 1000,  # 最大获取1000条订单信息
            'timestamp': ''
        }

        # =调用API获取订单信息
        get_spot_my_trades_func = getattr(self.exchange, self.constants.get('get_spot_my_trades_api'))
        trades = retry_wrapper(get_spot_my_trades_func, params=params, func_name='获取币种历史订单信息',
                               if_exit=False)  # 获取账户净值
        # 如果获取订单数据失败，进行容错处理，返回空df
        if trades is None:
            return pd.DataFrame()

        trades = pd.DataFrame(trades)  # 转成df格式
        # =如果获取到的该币种的订单数据是空的，则跳过，继续获取另外一个币种
        if trades.empty:
            return pd.DataFrame()

        # 转换数据格式
        for col in ('isBuyer', 'price', 'qty', 'quoteQty', 'commission'):
            trades[col] = pd.to_numeric(trades[col], errors='coerce')

        # =如果isBuyer为1则为买入，否则为卖出
        trades['方向'] = np.where(trades['isBuyer'] == 1, 1, -1)
        # =整理下有用的数据
        trades = trades[['time', 'symbol', 'price', 'qty', 'quoteQty', 'commission', 'commissionAsset', '方向']]

        return trades

    @classmethod
    def get_dummy_client(cls) -> 'BinanceClient':
        return cls()
