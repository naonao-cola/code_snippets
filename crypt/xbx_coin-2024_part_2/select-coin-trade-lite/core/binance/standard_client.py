"""
邢不行™️选币实盘框架
Python数字货币量化投资课程

版权所有 ©️ 邢不行
微信: xbx8662

未经授权，不得复制、修改、或使用本代码的全部或部分内容。仅限个人学习用途，禁止商业用途。

Author: 邢不行
"""

import time
from datetime import datetime

import pandas as pd

from config import utc_offset
from core.binance.base_client import BinanceClient
from core.utils.commons import retry_wrapper


class StandardClient(BinanceClient):
    constants = dict(
        spot_account_type='SPOT',
        reset_page_leverage_api='fapiprivate_post_leverage',
        get_swap_position_api='fapiprivatev2_get_positionrisk',
        get_spot_open_orders_api='private_get_openorders',
        cancel_spot_open_orders_api='private_delete_openorders',
        get_swap_open_orders_api='fapiprivate_get_openorders',
        cancel_swap_open_orders_api='fapiprivate_delete_allopenorders',
        get_spot_my_trades_api='private_get_mytrades',
    )

    def __init__(self, **config):
        super().__init__(**config)
        self.is_pure_long: bool = config.get('is_pure_long', False)

    def _set_position_side(self, dual_side_position=False):
        """
        检查是否是单向持仓模式
        """
        params = {'dualSidePosition': 'true' if dual_side_position else 'false', 'timestamp': ''}
        retry_wrapper(self.exchange.fapiprivate_post_positionside_dual, params=params,
                      func_name='fapiprivate_post_positionside_dual', if_exit=False)
        print(f'ℹ️修改持仓模式为单向持仓')

    def set_single_side_position(self):

        # 查询持仓模式
        res = retry_wrapper(
            self.exchange.fapiprivate_get_positionside_dual, params={'timestamp': ''}, func_name='设置单向持仓',
            if_exit=False
        )

        is_duel_side_position = bool(res['dualSidePosition'])

        # 判断是否是单向持仓模式
        if is_duel_side_position:  # 若当前持仓模式不是单向持仓模式，则调用接口修改持仓模式为单向持仓模式
            self._set_position_side(dual_side_position=False)

    def set_multi_assets_margin(self):
        """
        检查是否开启了联合保证金模式
        """
        # 查询保证金模式
        res = retry_wrapper(self.exchange.fapiprivate_get_multiassetsmargin, params={'timestamp': ''},
                            func_name='fapiprivate_get_multiassetsmargin', if_exit=False)
        # 判断是否开启了联合保证金模式
        if not bool(res['multiAssetsMargin']):  # 若联合保证金模式没有开启，则调用接口开启一下联合保证金模式
            params = {'multiAssetsMargin': 'true', 'timestamp': ''}
            retry_wrapper(self.exchange.fapiprivate_post_multiassetsmargin, params=params,
                          func_name='fapiprivate_post_multiassetsmargin', if_exit=False)
            print('✅开启联合保证金模式')

    def get_account_overview(self):
        spot_ticker_data = self.fetch_spot_ticker_price()
        spot_ticker = {_['symbol']: float(_['price']) for _ in spot_ticker_data}

        swap_account = self.get_swap_account()
        equity = pd.DataFrame(swap_account['assets'])
        swap_usdt_balance = float(equity[equity['asset'] == 'USDT']['walletBalance'])  # 获取usdt资产
        # 计算联合保证金
        if self.coin_margin:
            for _symbol, _coin_balance in self.coin_margin.items():
                if _symbol.replace('USDT', '') in equity['asset'].to_list():
                    swap_usdt_balance += _coin_balance
                else:
                    print(f'⚠️合约账户未找到 {_symbol} 的资产，无法计算 {_symbol} 的保证金')

        swap_position_df = self.get_swap_position_df()
        spot_position_df = self.get_spot_position_df()
        print('✅获取账户资产数据完成\n')

        print(f'ℹ️准备处理资产数据...')
        # 获取当前账号现货U的数量
        if 'USDT' in spot_position_df['symbol'].to_list():
            spot_usdt_balance = spot_position_df.loc[spot_position_df['symbol'] == 'USDT', '当前持仓量'].iloc[0]
            # 去除掉USDT现货
            spot_position_df = spot_position_df[spot_position_df['symbol'] != 'USDT']
        else:
            spot_usdt_balance = 0
        # 追加USDT后缀，方便计算usdt价值
        spot_position_df.loc[spot_position_df['symbol'] != 'USDT', 'symbol'] = spot_position_df['symbol'] + 'USDT'
        spot_position_df['仓位价值'] = spot_position_df.apply(
            lambda row: row['当前持仓量'] * spot_ticker.get(row["symbol"], 0), axis=1)

        # 过滤掉不含报价的币
        spot_position_df = spot_position_df[spot_position_df['仓位价值'] != 0]
        # 仓位价值 小于 5U，无法下单的碎币，单独记录
        dust_spot_df = spot_position_df[spot_position_df['仓位价值'] < 5]
        # 过滤掉仓位价值 小于 5U
        spot_position_df = spot_position_df[spot_position_df['仓位价值'] > 5]
        # 过滤掉BNB，用于抵扣手续费，不参与现货交易
        spot_position_df = spot_position_df[spot_position_df['symbol'] != 'BNBUSDT']

        # 现货净值
        spot_equity = spot_position_df['仓位价值'].sum() + spot_usdt_balance

        # 持仓盈亏
        account_pnl = swap_position_df['持仓盈亏'].sum()

        # =====处理现货持仓列表信息
        # 构建币种的balance信息
        # 币种 : 价值
        spot_assets_pos_dict = spot_position_df[['symbol', '仓位价值']].to_dict(orient='records')
        spot_assets_pos_dict = {_['symbol']: _['仓位价值'] for _ in spot_assets_pos_dict}

        # 币种 : 数量
        spot_asset_amount_dict = spot_position_df[['symbol', '当前持仓量']].to_dict(orient='records')
        spot_asset_amount_dict = {_['symbol']: _['当前持仓量'] for _ in spot_asset_amount_dict}

        # =====处理合约持仓列表信息
        # 币种 : 价值
        swap_position_df.reset_index(inplace=True)
        swap_assets_pos_dict = swap_position_df[['symbol', '仓位价值']].to_dict(orient='records')
        swap_assets_pos_dict = {_['symbol']: _['仓位价值'] for _ in swap_assets_pos_dict}

        # 币种 : 数量
        swap_asset_amount_dict = swap_position_df[['symbol', '当前持仓量']].to_dict(orient='records')
        swap_asset_amount_dict = {_['symbol']: _['当前持仓量'] for _ in swap_asset_amount_dict}

        # 币种 : pnl
        swap_asset_pnl_dict = swap_position_df[['symbol', '持仓盈亏']].to_dict(orient='records')
        swap_asset_pnl_dict = {_['symbol']: _['持仓盈亏'] for _ in swap_asset_pnl_dict}

        # 处理完成之后在设置index
        swap_position_df.set_index('symbol', inplace=True)

        # 账户总净值 = 现货总价值 + 合约usdt + 持仓盈亏
        account_equity = (spot_equity + swap_usdt_balance + account_pnl)

        print('✅处理资产数据完成\n')

        return {
            'usdt_balance': spot_usdt_balance + swap_usdt_balance,
            'negative_balance': 0,
            'account_pnl': account_pnl,
            'account_equity': account_equity,
            'spot_assets': {
                'assets_pos_value': spot_assets_pos_dict,
                'assets_amount': spot_asset_amount_dict,
                'usdt': spot_usdt_balance,
                'equity': spot_equity,
                'dust_spot_df': dust_spot_df,
                'spot_position_df': spot_position_df
            },
            'swap_assets': {
                'assets_pos_value': swap_assets_pos_dict,
                'assets_amount': swap_asset_amount_dict,
                'assets_pnl': swap_asset_pnl_dict,
                'usdt': swap_usdt_balance,
                'equity': swap_usdt_balance + account_pnl,
                'swap_position_df': swap_position_df
            }
        }

    def fetch_transfer_history(self, start_time=datetime.now()):
        """
        获取划转记录

        MAIN_UMFUTURE 现货钱包转向U本位合约钱包
        MAIN_MARGIN 现货钱包转向杠杆全仓钱包

        UMFUTURE_MAIN U本位合约钱包转向现货钱包
        UMFUTURE_MARGIN U本位合约钱包转向杠杆全仓钱包

        CMFUTURE_MAIN 币本位合约钱包转向现货钱包

        MARGIN_MAIN 杠杆全仓钱包转向现货钱包
        MARGIN_UMFUTURE 杠杆全仓钱包转向U本位合约钱包

        MAIN_FUNDING 现货钱包转向资金钱包
        FUNDING_MAIN 资金钱包转向现货钱包

        FUNDING_UMFUTURE 资金钱包转向U本位合约钱包
        UMFUTURE_FUNDING U本位合约钱包转向资金钱包

        MAIN_OPTION 现货钱包转向期权钱包
        OPTION_MAIN 期权钱包转向现货钱包

        UMFUTURE_OPTION U本位合约钱包转向期权钱包
        OPTION_UMFUTURE 期权钱包转向U本位合约钱包

        MAIN_PORTFOLIO_MARGIN 现货钱包转向统一账户钱包
        PORTFOLIO_MARGIN_MAIN 统一账户钱包转向现货钱包

        MAIN_ISOLATED_MARGIN 现货钱包转向逐仓账户钱包
        ISOLATED_MARGIN_MAIN 逐仓钱包转向现货账户钱包
        """
        start_time = start_time - pd.Timedelta(days=10)
        add_type = ['CMFUTURE_MAIN', 'MARGIN_MAIN', 'MARGIN_UMFUTURE', 'FUNDING_MAIN', 'FUNDING_UMFUTURE',
                    'OPTION_MAIN', 'OPTION_UMFUTURE', 'PORTFOLIO_MARGIN_MAIN', 'ISOLATED_MARGIN_MAIN']
        reduce_type = ['MAIN_MARGIN', 'UMFUTURE_MARGIN', 'MAIN_FUNDING', 'UMFUTURE_FUNDING', 'MAIN_OPTION',
                       'UMFUTURE_OPTION', 'MAIN_PORTFOLIO_MARGIN', 'MAIN_ISOLATED_MARGIN']

        result = []
        for _ in add_type + reduce_type:
            params = {
                'fromSymbol': 'USDT',
                'startTime': int(start_time.timestamp() * 1000),
                'type': _,
                'timestamp': int(round(time.time() * 1000)),
                'size': 100,
            }
            if _ == 'MAIN_ISOLATED_MARGIN':
                params['toSymbol'] = 'USDT'
                del params['fromSymbol']
            # 获取划转信息(取上一小时到当前时间的划转记录)
            try:
                account_info = self.exchange.sapi_get_asset_transfer(params)
            except BaseException as e:
                print(e)
                print(f'当前账户查询类型【{_}】失败，不影响后续操作，请忽略')
                continue
            if account_info and int(account_info['total']) > 0:
                res = pd.DataFrame(account_info['rows'])
                res['timestamp'] = pd.to_datetime(res['timestamp'], unit='ms')
                res.loc[res['type'].isin(add_type), 'flag'] = 1
                res.loc[res['type'].isin(reduce_type), 'flag'] = -1
                res = res[res['status'] == 'CONFIRMED']
                result.append(res)

        # 获取主账号与子账号之间划转记录
        result2 = []
        for transfer_type in [1, 2]:  # 1: 划入。从主账号划转进来  2: 划出。从子账号划转出去
            params = {
                'asset': 'USDT',
                'type': transfer_type,
                'startTime': int(start_time.timestamp() * 1000),
            }
            try:
                account_info = self.exchange.sapi_get_sub_account_transfer_subuserhistory(params)
            except BaseException as e:
                print(e)
                print(f'当前账户查询类型【{transfer_type}】失败，不影响后续操作，请忽略')
                continue
            if account_info and len(account_info):
                res = pd.DataFrame(account_info)
                res['time'] = pd.to_datetime(res['time'], unit='ms')
                res.rename(columns={'qty': 'amount', 'time': 'timestamp'}, inplace=True)
                res.loc[res['toAccountType'] == 'SPOT', 'flag'] = 1 if transfer_type == 1 else -1
                res.loc[res['toAccountType'] == 'USDT_FUTURE', 'flag'] = 1 if transfer_type == 1 else -1
                res = res[res['status'] == 'SUCCESS']
                res = res[res['toAccountType'].isin(['SPOT', 'USDT_FUTURE'])]
                result2.append(res)

        # 将账号之间的划转与单账号内部换转数据合并
        result.extend(result2)
        if not len(result):
            return pd.DataFrame()

        all_df = pd.concat(result, ignore_index=True)
        all_df.drop_duplicates(subset=['timestamp', 'tranId', 'flag'], inplace=True)
        all_df = all_df[all_df['asset'] == 'USDT']
        all_df.sort_values('timestamp', inplace=True)

        all_df['amount'] = all_df['amount'].astype(float) * all_df['flag']
        all_df.rename(columns={'amount': '账户总净值'}, inplace=True)
        all_df['type'] = 'transfer'
        all_df = all_df[['timestamp', '账户总净值', 'type']]
        all_df['timestamp'] = all_df['timestamp'] + pd.Timedelta(hours=utc_offset)
        all_df.reset_index(inplace=True, drop=True)

        all_df['time'] = all_df['timestamp']
        result_df = all_df.resample(rule='1H', on='timestamp').agg(
            {'time': 'last', '账户总净值': 'sum', 'type': 'last'})
        result_df = result_df[result_df['type'].notna()]
        result_df.reset_index(inplace=True, drop=True)

        return result_df
