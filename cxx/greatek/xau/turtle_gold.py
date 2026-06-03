import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np
import time
import argparse

# ===========================
# 1. 爬虫逻辑 (VSTAR)
# ===========================
class VStarScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        })
        self.URL = "https://trade.vstar.com/cn/XAUUSD"

    def fetch_current_price(self):
        """
        从 VSTAR 获取当前黄金价格。
        """
        try:
            print(f"正在从 {self.URL} 获取实时价格...", end=" ")
            response = self.session.get(self.URL, timeout=15)
            print(f"状态码: {response.status_code}")

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')

                # 方案1: 查找页面上的价格文本模式 (例如 "5063.13")
                text = soup.get_text()

                # 搜索 "黄金" 关键字附近
                idx = text.find("黄金")
                if idx != -1:
                    context = text[idx:idx+100]
                    matches = re.findall(r"(\d{4}\.\d{2})", context)
                    if matches:
                        price = float(matches[0])
                        print(f"获取成功 (文本匹配): {price}")
                        return price

                # 方案2: 全页搜索 4位数.2位数 的模式
                all_matches = re.findall(r"(\d{4}\.\d{2})", text)
                for m in all_matches:
                    val = float(m)
                    if 2000 < val < 6000:
                        print(f"获取成功 (模糊匹配): {val}")
                        return val

            print("未能解析出价格 (可能需要JS渲染)。")
            return None
        except Exception as e:
            print(f"获取价格出错: {e}")
            return None

# ===========================
# 2. 数据处理与策略
# ===========================

def get_gold_data(interval='1d', periods=120):
    """
    获取数据用于策略计算。
    支持自定义时间周期 (interval): '15m', '1h', '4h', '1d'

    策略:
    1. 尝试从 VSTAR 抓取最新价格。
    2. 历史数据: 使用模拟趋势数据生成历史K线，时间间隔由 interval 决定。
    """
    scraper = VStarScraper()
    current_price = scraper.fetch_current_price()

    if current_price is None:
        print("警告: 无法从 VSTAR 获取实时价格。")
        fallback_price = 5063.13
        print(f"使用兜底参考价格: {fallback_price} (仅供演示)")
        current_price = fallback_price

    print(f"\n当前国际金价: {current_price}")
    print(f"当前周期: {interval}")
    print("注意: 历史数据将使用模拟生成的趋势数据来演示海龟法则的计算逻辑。")

    # 映射 interval 到 pandas frequency
    freq_map = {
        '15m': '15min',
        '1h': 'h',
        '4h': '4h',
        '1d': 'D'
    }
    freq = freq_map.get(interval, 'D')

    # 调整模拟波动率 (时间越短，单根K线波动越小)
    vol_map = {
        '15m': 0.001,  # 0.1%
        '1h':  0.002,  # 0.2%
        '4h':  0.004,  # 0.4%
        '1d':  0.008   # 0.8%
    }
    vol_scale = vol_map.get(interval, 0.008)

    # 生成历史时间序列 (以当前时间为终点)
    end_date = pd.Timestamp.now()
    # 生成 periods + 1 个点，最后一个点用作"当前实时K线"
    dates = pd.date_range(end=end_date, periods=periods, freq=freq)

    # 我们保留最后一个时间点给"当前K线"，前面的用于生成历史
    history_dates = dates[:-1]

    data = []
    # 倒推生成历史数据
    sim_price = current_price
    np.random.seed(42)

    # 生成波动序列
    changes = np.random.normal(0, vol_scale, len(history_dates))

    history_prices = []
    temp_price = current_price
    for change in reversed(changes):
        temp_price = temp_price / (1 + change)
        history_prices.append(temp_price)

    history_prices.reverse()

    for i, date in enumerate(history_dates):
        p = history_prices[i]
        # K线内波动也随周期调整
        intra_vol = vol_scale * 1.2

        open_p = p * (1 + np.random.normal(0, intra_vol/5))
        close_p = p
        high_p = max(open_p, close_p) * (1 + abs(np.random.normal(0, intra_vol/2)))
        low_p = min(open_p, close_p) * (1 - abs(np.random.normal(0, intra_vol/2)))

        data.append({
            'Date': date,
            'Open': open_p,
            'High': high_p,
            'Low': low_p,
            'Close': close_p,
            'Volume': 1000 + np.random.randint(0, 500)
        })

    # 添加"当前"K线 (基于实时价格)
    current_date = dates[-1]
    data.append({
        'Date': current_date,
        'Open': current_price, # 简化: Open=Close
        'High': current_price,
        'Low': current_price,
        'Close': current_price,
        'Volume': 0
    })

    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)
    return df

def calculate_indicators(df):
    """
    计算海龟法则所需的技术指标:
    1. ATR (20)
    2. 过去20日收盘价最高价 (Donchian High 20)
    3. 过去20日收盘价最低价 (Donchian Low 20)
    4. 过去10日收盘价最高价 (Donchian High 10)
    5. 过去10日收盘价最低价 (Donchian Low 10)
    """
    # 1. 计算 TR
    # TR = Max(H-L, Abs(H-PreClose), Abs(L-PreClose))
    df['prev_close'] = df['Close'].shift(1)
    df['tr1'] = df['High'] - df['Low']
    df['tr2'] = (df['High'] - df['prev_close']).abs()
    df['tr3'] = (df['Low'] - df['prev_close']).abs()
    df['TR'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)

    # ATR (20) - 简单移动平均
    df['ATR'] = df['TR'].rolling(window=20).mean()

    # 2. 过去20日收盘价最高价 (用于多头入场)
    # 注意: 是"过去"20根，不包含当前K线，所以要shift(1)
    df['max_close_20'] = df['Close'].rolling(window=20).max().shift(1)

    # 3. 过去20日收盘价最低价 (用于空头入场)
    df['min_close_20'] = df['Close'].rolling(window=20).min().shift(1)

    # 4. 过去10日收盘价最低价 (用于多头止损/离场)
    df['min_close_10'] = df['Close'].rolling(window=10).min().shift(1)

    # 5. 过去10日收盘价最高价 (用于空头止损/离场)
    df['max_close_10'] = df['Close'].rolling(window=10).max().shift(1)

    return df

def run_turtle_strategy(df):
    """
    执行海龟策略逻辑，生成买卖点。
    """
    signals = []

    # 仓位状态
    position = 0  # 0: 空仓, >0: 多头份数, <0: 空头份数
    entry_price = 0.0 # 上一次买入价
    stop_loss = 0.0 # 当前止损价

    # 遍历数据 (从第21天开始，因为前面数据不足)
    for i in range(21, len(df)):
        date = df.index[i]
        price = df['Close'].iloc[i] # 使用收盘价作为判定依据
        atr = df['ATR'].iloc[i]

        max_20 = df['max_close_20'].iloc[i]
        min_20 = df['min_close_20'].iloc[i]
        min_10 = df['min_close_10'].iloc[i]
        max_10 = df['max_close_10'].iloc[i]

        if pd.isna(atr) or pd.isna(max_20):
            continue

        action = None
        detail = ""

        # --- 多头策略 ---
        # 1. 入场: 价格 > 过去20日收盘最高价
        if position == 0:
            if price > max_20:
                position = 1
                entry_price = price
                # 初始止损: 上一次买入价 - 2ATR (规则三)
                stop_loss = entry_price - 2 * atr
                # 加仓点: 上一次买入价 + 0.5ATR (规则二)
                next_add = entry_price + 0.5 * atr

                action = "Buy Open (Long)"
                detail = f"Price({price:.2f}) > Max20({max_20:.2f}), ATR={atr:.2f}"
                signals.append({
                    'Date': date, 'Type': action, 'Price': price,
                    'Units': position, 'StopLoss': stop_loss, 'Reason': detail
                })

        # 持有多头时的逻辑
        elif position > 0:
            # 3. 止损: 价格 < 上一次买入价 - 2ATR
            # (注意: 这里使用了更新后的 stop_loss，因为每次加仓都会更新 stop_loss)
            if price < stop_loss:
                position = 0
                action = "Sell Close (Stop Loss)"
                detail = f"Price({price:.2f}) < StopLoss({stop_loss:.2f})"
                signals.append({
                    'Date': date, 'Type': action, 'Price': price,
                    'Units': 0, 'StopLoss': 0, 'Reason': detail
                })

            # 4. 离场: 价格 < 过去10日收盘最低价
            elif price < min_10:
                position = 0
                action = "Sell Close (Exit)"
                detail = f"Price({price:.2f}) < Min10({min_10:.2f})"
                signals.append({
                    'Date': date, 'Type': action, 'Price': price,
                    'Units': 0, 'StopLoss': 0, 'Reason': detail
                })

            # 2. 加仓: 价格 > 上一次买入价 + 0.5 ATR
            elif price > entry_price + 0.5 * atr:
                position += 1
                prev_entry = entry_price
                entry_price = price # 更新"上一次买入价"

                # 更新止损: 新的买入价 - 2ATR (海龟法则通常会提升整体止损)
                stop_loss = entry_price - 2 * atr

                action = "Buy Add"
                detail = f"Price({price:.2f}) > PrevEntry({prev_entry:.2f}) + 0.5*ATR, New Stop={stop_loss:.2f}"
                signals.append({
                    'Date': date, 'Type': action, 'Price': price,
                    'Units': position, 'StopLoss': stop_loss, 'Reason': detail
                })

        # --- 空头策略 ---
        # (如果当前持有多头，不会开空头，必须先平仓。这里简化为独立判断，但实际应互斥)
        # 如果是空仓，检查空头信号
        if position == 0:
            # 1. 入场: 价格 < 过去20日收盘最低价
            if price < min_20:
                position = -1
                entry_price = price
                # 初始止损: 上一次买入价 + 2ATR
                stop_loss = entry_price + 2 * atr

                action = "Sell Short Open"
                detail = f"Price({price:.2f}) < Min20({min_20:.2f})"
                signals.append({
                    'Date': date, 'Type': action, 'Price': price,
                    'Units': position, 'StopLoss': stop_loss, 'Reason': detail
                })

        # 持有空头时的逻辑
        elif position < 0:
            # 3. 止损: 价格 > 上一次买入价 + 2ATR
            if price > stop_loss:
                position = 0
                action = "Cover Close (Stop Loss)"
                detail = f"Price({price:.2f}) > StopLoss({stop_loss:.2f})"
                signals.append({
                    'Date': date, 'Type': action, 'Price': price,
                    'Units': 0, 'StopLoss': 0, 'Reason': detail
                })

            # 4. 离场: 价格 > 过去10日收盘最高价
            elif price > max_10:
                position = 0
                action = "Cover Close (Exit)"
                detail = f"Price({price:.2f}) > Max10({max_10:.2f})"
                signals.append({
                    'Date': date, 'Type': action, 'Price': price,
                    'Units': 0, 'StopLoss': 0, 'Reason': detail
                })

            # 2. 加仓: 价格 < 上一次买入价 - 0.5 ATR
            elif price < entry_price - 0.5 * atr:
                position -= 1 # 负数表示空头份数增加
                prev_entry = entry_price
                entry_price = price

                # 更新止损
                stop_loss = entry_price + 2 * atr

                action = "Sell Short Add"
                detail = f"Price({price:.2f}) < PrevEntry({prev_entry:.2f}) - 0.5*ATR, New Stop={stop_loss:.2f}"
                signals.append({
                    'Date': date, 'Type': action, 'Price': price,
                    'Units': position, 'StopLoss': stop_loss, 'Reason': detail
                })

    return pd.DataFrame(signals)

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Turtle Trading Strategy for Gold')
    parser.add_argument('--interval', type=str, default='1d', choices=['15m', '1h', '4h', '1d'],
                        help='K线周期 (15m, 1h, 4h, 1d)')
    args = parser.parse_args()

    # 1. 获取数据
    df = get_gold_data(interval=args.interval, periods=120)

    if df is not None:
        # 2. 计算指标
        df = calculate_indicators(df)

        # 3. 运行策略
        results = run_turtle_strategy(df)

        if not results.empty:
            print("\n=== 海龟交易法则买卖点 (最近10条) ===")
            # 格式化输出
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 1000)

            # 显示最后10条记录
            print(results.tail(10).to_string(index=False))

            # 保存结果到 CSV
            results.to_csv("turtle_signals.csv")
            print("\n已保存完整信号记录到 turtle_signals.csv")

        # 显示当前具体的交易状态判定
        print("\n=== 实时交易信号判定 ===")
        latest = df.iloc[-1]
        prev = df.iloc[-2] # 前一天的指标用于判定今天

        # 获取判定阈值 (注意：海龟法则是用"前一天"的指标来判定"今天"的价格)
        # 在 calculate_indicators 中，我们已经 shift(1) 了，
        # 所以 df.iloc[-1]['max_close_20'] 实际上是基于 yesterday 及之前的数据计算出来的，可以直接用于判定 today

        # 修正: calculate_indicators 中:
        # df['max_close_20'] = df['Close'].rolling(window=20).max().shift(1)
        # 这意味着 row[i] 的 max_close_20 已经是 i-1, i-2... 的最大值
        # 所以直接取 latest 的指标即可

        buy_point = latest['max_close_20']
        sell_point = latest['min_close_20']
        atr = latest['ATR']
        current_p = latest['Close']

        print(f"当前时间: {latest.name}")
        print(f"当前价格: {current_p:.2f}")
        print(f"当前 ATR(20): {atr:.2f}")
        print("-" * 30)
        print(f"【多头入场点】: 价格突破 {buy_point:.2f} (过去20日最高)")
        print(f"【空头入场点】: 价格跌破 {sell_point:.2f} (过去20日最低)")

        if current_p > buy_point:
            print(f"★ 信号: 满足多头入场条件！(当前 {current_p:.2f} > {buy_point:.2f})")
        elif current_p < sell_point:
            print(f"★ 信号: 满足空头入场条件！(当前 {current_p:.2f} < {sell_point:.2f})")
        else:
            print(f"☆ 信号: 无新入场信号 (价格在通道内)")

        # 如果有持仓，还需要判断加仓/止损 (需结合策略运行结果的状态)
        if not results.empty:
            last_sig = results.iloc[-1]
            # 简单判断最后一次信号如果是 Open 或 Add 且不是 Close
            if "Close" not in last_sig['Type']:
                print(f"\n【持仓监控】")
                print(f"当前持仓状态: {last_sig['Type']} ({last_sig['Units']} units)")
                print(f"当前止损价: {last_sig['StopLoss']:.2f}")
                if "Long" in last_sig['Type'] or last_sig['Units'] > 0:
                     add_price = last_sig['Price'] + 0.5 * atr
                     print(f"下一次加仓价: > {add_price:.2f}")
                elif "Short" in last_sig['Type'] or last_sig['Units'] < 0:
                     add_price = last_sig['Price'] - 0.5 * atr
                     print(f"下一次加仓价: < {add_price:.2f}")

    else:
        print("无法运行策略。")
