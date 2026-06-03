import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import re
import requests
from bs4 import BeautifulSoup

class TurtleGoldTrader:
    def __init__(self, timeframe='30m', lookback=20, exit_period=10):
        """
        基于海龟交易法则的黄金交易信号生成器

        参数:
        - timeframe: 数据周期 ('15m', '30m', '1h', '4h', '1d')
        - lookback: 突破周期 (默认20)
        - exit_period: 出场周期 (默认10)
        """
        self.timeframe = timeframe
        self.lookback = lookback
        self.exit_period = exit_period
        self.atr_period = 14

        # TradingView 数据源
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    def fetch_gold_data(self, bars=200):
        """
        从TradingView获取黄金K线数据

        参数:
        - bars: 需要获取的K线数量

        返回:
        - DataFrame包含: timestamp, open, high, low, close
        """

        # TradingView 的黄金数据API (XAUUSD)
        # 不同周期对应不同的symbol
        symbol_map = {
            '15m': 'OANDA:XAUUSD',
            '30m': 'OANDA:XAUUSD',
            '1h': 'OANDA:XAUUSD',
            '4h': 'OANDA:XAUUSD',
            '1d': 'OANDA:XAUUSD'
        }

        # 转换为TradingView的时间间隔
        interval_map = {
            '15m': '15',
            '30m': '30',
            '1h': '60',
            '4h': '240',
            '1d': 'D'
        }

        symbol = symbol_map.get(self.timeframe, 'OANDA:XAUUSD')
        interval = interval_map.get(self.timeframe, '30')

        print(f"正在从TradingView获取黄金{self.timeframe}数据...")

        # 尝试使用TradingView的数据
        try:
            url = f"https://scanner.tradingview.com/forex/scan"

            # 构造请求数据
            payload = {
                "symbols": {
                    "tickers": [symbol],
                    "query": {
                        "types": []
                    }
                },
                "columns": [
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "time"
                ]
            }

            response = self.session.post(url, json=payload, timeout=10)

            if response.status_code == 200:
                data = response.json()
                if 'data' in data and len(data['data']) > 0:
                    # 解析数据
                    kline_data = []
                    for item in data['data']:
                        if len(item) >= 6:
                            kline_data.append({
                                'timestamp': datetime.fromtimestamp(item[5]),
                                'open': item[0],
                                'high': item[1],
                                'low': item[2],
                                'close': item[3],
                                'volume': item[4]
                            })

                    if kline_data:
                        df = pd.DataFrame(kline_data)
                        df = df.sort_values('timestamp')
                        print(f"成功获取 {len(df)} 根K线数据")
                        return df
        except Exception as e:
            print(f"TradingView API错误: {e}")

        # 如果TradingView失败，使用Investing.com的模拟方法
        print("使用Investing.com数据源...")
        return self.fetch_investing_data(bars)

    def fetch_investing_data(self, bars=200):
        """
        从Investing.com获取历史数据（模拟）
        实际使用时需要更复杂的爬虫逻辑
        """
        # 这里简单模拟一些数据用于测试
        print("注意：正在使用模拟数据，实际使用时需要实现真实的爬虫逻辑")

        # 创建模拟数据
        np.random.seed(42)
        base_price = 1950  # 基准价格

        dates = []
        prices = []

        current_time = datetime.now()

        # 根据时间间隔确定时间差
        if self.timeframe == '15m':
            delta = timedelta(minutes=15)
        elif self.timeframe == '30m':
            delta = timedelta(minutes=30)
        elif self.timeframe == '1h':
            delta = timedelta(hours=1)
        elif self.timeframe == '4h':
            delta = timedelta(hours=4)
        else:  # 1d
            delta = timedelta(days=1)

        # 生成模拟K线
        price = base_price
        for i in range(bars):
            timestamp = current_time - (delta * (bars - i))
            dates.append(timestamp)

            # 模拟价格波动
            change = np.random.normal(0, 5)  # 正常波动
            price += change
            if price < 1900:
                price = 1900 + np.random.random() * 50
            if price > 2100:
                price = 2100 - np.random.random() * 50

            # 生成OHLC
            open_price = price
            high_price = open_price + abs(np.random.normal(0, 3))
            low_price = open_price - abs(np.random.normal(0, 3))
            close_price = open_price + np.random.normal(0, 2)

            prices.append({
                'timestamp': timestamp,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': np.random.randint(1000, 10000)
            })

        df = pd.DataFrame(prices)
        return df

    def calculate_technical_indicators(self, df):
        """计算技术指标"""
        df = df.copy()

        # 计算ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=self.atr_period).mean()

        # 计算突破点
        df[f'{self.lookback}_High'] = df['close'].rolling(window=self.lookback).max()
        df[f'{self.lookback}_Low'] = df['close'].rolling(window=self.lookback).min()
        df[f'{self.exit_period}_High'] = df['close'].rolling(window=self.exit_period).max()
        df[f'{self.exit_period}_Low'] = df['close'].rolling(window=self.exit_period).min()

        return df

    def generate_signals(self, df):
        """
        生成海龟交易法则信号

        返回:
        - 包含信号的DataFrame
        """
        df = df.copy()

        # 初始化信号列
        df['signal'] = ''
        df['action'] = ''
        df['reason'] = ''

        # 状态跟踪
        in_long = False
        in_short = False
        last_long_price = 0
        last_short_price = 0

        # 确保有足够的数据
        if len(df) < max(self.lookback, self.exit_period, self.atr_period):
            print("数据不足，无法计算信号")
            return df

        print(f"开始分析 {len(df)} 根K线数据...")

        for i in range(self.lookback, len(df)):
            current_price = df['close'].iloc[i]
            atr = df['ATR'].iloc[i]

            # 避免ATR为0
            if pd.isna(atr) or atr == 0:
                atr = 1

            # === 多头逻辑 ===
            if not in_long:
                # 入场条件：价格突破N周期高点
                if current_price > df[f'{self.lookback}_High'].iloc[i-1]:
                    df.at[i, 'signal'] = 'BUY'
                    df.at[i, 'action'] = '开多仓'
                    df.at[i, 'reason'] = f'价格突破{self.lookback}周期高点'
                    in_long = True
                    last_long_price = current_price
            else:
                # 加仓条件：价格 > 上次买入价 + 0.5 * ATR
                if current_price > last_long_price + 0.5 * atr:
                    df.at[i, 'signal'] = 'BUY'
                    df.at[i, 'action'] = '加仓多单'
                    df.at[i, 'reason'] = f'价格>上次买入价+0.5ATR ({last_long_price:.1f}+{0.5*atr:.1f})'
                    last_long_price = current_price

                # 止损条件1：价格 < 上次买入价 - 2 * ATR
                if current_price < last_long_price - 2 * atr:
                    df.at[i, 'signal'] = 'SELL'
                    df.at[i, 'action'] = '止损平多'
                    df.at[i, 'reason'] = f'价格<上次买入价-2ATR ({last_long_price:.1f}-{2*atr:.1f})'
                    in_long = False

                # 止损条件2：价格跌破M周期低点
                if current_price < df[f'{self.exit_period}_Low'].iloc[i-1]:
                    df.at[i, 'signal'] = 'SELL'
                    df.at[i, 'action'] = '出场平多'
                    df.at[i, 'reason'] = f'价格跌破{self.exit_period}周期低点'
                    in_long = False

            # === 空头逻辑 ===
            if not in_short:
                # 入场条件：价格跌破N周期低点
                if current_price < df[f'{self.lookback}_Low'].iloc[i-1]:
                    df.at[i, 'signal'] = 'SELL'
                    df.at[i, 'action'] = '开空仓'
                    df.at[i, 'reason'] = f'价格跌破{self.lookback}周期低点'
                    in_short = True
                    last_short_price = current_price
            else:
                # 加仓条件：价格 < 上次卖出价 - 0.5 * ATR
                if current_price < last_short_price - 0.5 * atr:
                    df.at[i, 'signal'] = 'SELL'
                    df.at[i, 'action'] = '加仓空单'
                    df.at[i, 'reason'] = f'价格<上次卖出价-0.5ATR ({last_short_price:.1f}-{0.5*atr:.1f})'
                    last_short_price = current_price

                # 止损条件1：价格 > 上次卖出价 + 2 * ATR
                if current_price > last_short_price + 2 * atr:
                    df.at[i, 'signal'] = 'BUY'
                    df.at[i, 'action'] = '止损平空'
                    df.at[i, 'reason'] = f'价格>上次卖出价+2ATR ({last_short_price:.1f}+{2*atr:.1f})'
                    in_short = False

                # 止损条件2：价格突破M周期高点
                if current_price > df[f'{self.exit_period}_High'].iloc[i-1]:
                    df.at[i, 'signal'] = 'BUY'
                    df.at[i, 'action'] = '出场平空'
                    df.at[i, 'reason'] = f'价格突破{self.exit_period}周期高点'
                    in_short = False

        return df

    def get_current_signals(self, show_history=10):
        """
        获取当前交易信号

        参数:
        - show_history: 显示最近多少个信号

        返回:
        - 最新的交易信号
        """
        # 获取数据
        df = self.fetch_gold_data(bars=200)

        if df is None or len(df) == 0:
            return "无法获取数据，请检查网络连接"

        # 计算技术指标
        df = self.calculate_technical_indicators(df)

        # 生成信号
        df = self.generate_signals(df)

        # 过滤有信号的记录
        signals_df = df[df['signal'] != ''].copy()

        if len(signals_df) == 0:
            # 如果没有历史信号，检查当前是否满足入场条件
            current_price = df['close'].iloc[-1]
            high_20 = df[f'{self.lookback}_High'].iloc[-2]
            low_20 = df[f'{self.lookback}_Low'].iloc[-2]
            atr = df['ATR'].iloc[-1]

            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            result = f"""
            🕒 更新时间: {current_time}
            📊 黄金价格: ${current_price:.2f}
            📈 {self.lookback}周期高点: ${high_20:.2f}
            📉 {self.lookback}周期低点: ${low_20:.2f}
            📏 ATR波动率: {atr:.2f}

            ⚠️  当前状态: 无持仓信号
            """

            # 检查潜在入场机会
            potential_signals = []
            if current_price > high_20:
                potential_signals.append(f"✅ 潜在多头机会: 价格 > {self.lookback}周期高点")
            if current_price < low_20:
                potential_signals.append(f"✅ 潜在空头机会: 价格 < {self.lookback}周期低点")

            if potential_signals:
                result += "\n🔍 潜在交易机会:\n"
                for signal in potential_signals:
                    result += f"   • {signal}\n"

            return result

        else:
            # 有历史信号
            signals_df = signals_df.sort_values('timestamp', ascending=False)

            # 获取当前状态
            latest_signal = signals_df.iloc[0]
            current_price = df['close'].iloc[-1]
            atr = df['ATR'].iloc[-1]

            result = f"""
            🕒 更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            📊 黄金价格: ${current_price:.2f}
            📏 ATR波动率: {atr:.2f}

            📣 最新信号:
            ⏰ 时间: {latest_signal['timestamp'].strftime('%m-%d %H:%M')}
            📈 价格: ${latest_signal['close']:.2f}
            🎯 操作: {latest_signal['action']}
            📝 理由: {latest_signal['reason']}
            """

            # 显示最近几个信号
            if len(signals_df) > 1:
                result += f"\n📜 最近{min(show_history, len(signals_df)-1)}个历史信号:\n"
                for idx, row in signals_df.iloc[1:show_history+1].iterrows():
                    result += (f"   [{row['timestamp'].strftime('%m-%d %H:%M')}] "
                             f"{row['action']} @ ${row['close']:.2f}\n")

            return result

    def run_realtime_monitor(self, interval_seconds=60):
        """
        实时监控模式

        参数:
        - interval_seconds: 检查间隔（秒）
        """
        print(f"启动黄金海龟交易法则监控系统")
        print(f"时间周期: {self.timeframe}")
        print(f"突破周期: {self.lookback}")
        print(f"出场周期: {self.exit_period}")
        print("=" * 60)

        last_signal = None

        try:
            while True:
                signals = self.get_current_signals(show_history=5)
                print(signals)
                print("=" * 60)
                print(f"下次更新: {interval_seconds}秒后")
                print("=" * 60 + "\n")

                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            print("\n监控已停止")


# 快速使用函数
def check_gold_signals(timeframe='30m'):
    """
    快速检查黄金交易信号
    """
    trader = TurtleGoldTrader(timeframe=timeframe)
    return trader.get_current_signals()


# 使用示例
if __name__ == "__main__":
    # 方法1: 快速检查当前信号
    print("=" * 60)
    print("黄金30分钟K线海龟交易信号")
    print("=" * 60)
    signals_30m = check_gold_signals('30m')
    print(signals_30m)

    print("\n" + "=" * 60)
    print("黄金15分钟K线海龟交易信号")
    print("=" * 60)
    signals_15m = check_gold_signals('15m')
    print(signals_15m)

    # 方法2: 启动实时监控
    # trader = TurtleGoldTrader(timeframe='30m')
    # trader.run_realtime_monitor(interval_seconds=300)  # 每5分钟检查一次

    # 方法3: 获取原始数据进行分析
    # trader = TurtleGoldTrader(timeframe='1h')
    # data = trader.fetch_gold_data(bars=100)
    # data_with_indicators = trader.calculate_technical_indicators(data)
    # signals = trader.generate_signals(data_with_indicators)
    #
    # # 保存到CSV
    # signals.to_csv('gold_turtle_signals.csv', index=False, encoding='utf-8-sig')
    # print("信号已保存到 gold_turtle_signals.csv")