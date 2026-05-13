#!/usr/bin/env python3
"""
API数据查询工具
通过参数调用不同的公开API获取数据
"""
import argparse
import json
import requests
import socket
import sys
from datetime import datetime
from typing import Dict, Any, Optional


class APIFetcher:
    """API数据获取器"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def get_weather(self, city: str) -> Dict[str, Any]:
        """获取天气信息（使用OpenWeatherMap API，免费版）"""
        try:
            # 注意：实际使用时需要注册OpenWeatherMap并获取API密钥
            # 这里使用模拟数据作为演示
            api_key = "demo_key"  # 实际使用时替换为真实API密钥

            # 模拟响应数据
            if "北京" in city or "beijing" in city.lower():
                return {
                    "city": "Beijing",
                    "temperature": 22.5,
                    "humidity": 65,
                    "description": "clear sky",
                    "wind_speed": 3.2,
                    "pressure": 1013,
                    "timestamp": datetime.now().isoformat()
                }
            elif "上海" in city or "shanghai" in city.lower():
                return {
                    "city": "Shanghai",
                    "temperature": 24.0,
                    "humidity": 70,
                    "description": "few clouds",
                    "wind_speed": 4.1,
                    "pressure": 1012,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "city": city,
                    "temperature": 20.0,
                    "humidity": 60,
                    "description": "moderate rain",
                    "wind_speed": 2.5,
                    "pressure": 1015,
                    "timestamp": datetime.now().isoformat(),
                    "note": "模拟数据 - 实际使用时需要真实API密钥"
                }
        except Exception as e:
            return {"error": f"获取天气信息失败: {str(e)}"}

    def get_exchange_rate(self, base_currency: str, target_currency: str = "CNY") -> Dict[str, Any]:
        """获取汇率信息（使用Frankfurter API）"""
        try:
            # 使用Frankfurter免费API
            url = f"https://api.frankfurter.app/latest?from={base_currency}&to={target_currency}"
            response = self.session.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                return {
                    "base": data["base"],
                    "date": data["date"],
                    "rates": data["rates"],
                    "timestamp": datetime.now().isoformat()
                }
            else:
                # 模拟数据（当API不可用时）
                rates = {
                    "USD": {"CNY": 7.25, "EUR": 0.92, "JPY": 147.50},
                    "EUR": {"USD": 1.09, "CNY": 7.90, "JPY": 160.20},
                    "CNY": {"USD": 0.14, "EUR": 0.13, "JPY": 20.35}
                }

                if base_currency in rates and target_currency in rates.get(base_currency, {}):
                    rate = rates[base_currency][target_currency]
                    return {
                        "base": base_currency,
                        "date": datetime.now().strftime("%Y-%m-%d"),
                        "rates": {target_currency: rate},
                        "timestamp": datetime.now().isoformat(),
                        "note": "模拟数据 - 实际汇率可能不同"
                    }
                else:
                    return {"error": f"不支持{base_currency}到{target_currency}的汇率查询"}

        except Exception as e:
            return {"error": f"获取汇率信息失败: {str(e)}"}

    def get_news_headlines(self, category: str = "general") -> Dict[str, Any]:
        """获取新闻头条（使用NewsAPI模拟）"""
        try:
            # 模拟新闻数据
            news_by_category = {
                "technology": [
                    {"title": "AI Breakthrough in Natural Language Processing", "source": "TechNews"},
                    {"title": "New Quantum Computing Milestone Achieved", "source": "ScienceDaily"},
                    {"title": "Major Tech Company Announces New AI Chip", "source": "BusinessTech"}
                ],
                "business": [
                    {"title": "Global Markets Reach New Highs", "source": "FinancialTimes"},
                    {"title": "Central Banks Announce Interest Rate Decisions", "source": "Bloomberg"},
                    {"title": "New Startup Valuation Tops $1 Billion", "source": "Forbes"}
                ],
                "sports": [
                    {"title": "National Team Wins Championship", "source": "SportsNetwork"},
                    {"title": "Record Broken in International Competition", "source": "AthleticsWorld"},
                    {"title": "Major League Announces New Season Format", "source": "SportsToday"}
                ],
                "general": [
                    {"title": "International Summit Concludes with New Agreements", "source": "WorldNews"},
                    {"title": "Scientific Discovery Could Change Energy Industry", "source": "ScienceJournal"},
                    {"title": "Cultural Festival Attracts Millions of Visitors", "source": "CultureDaily"}
                ]
            }

            news_list = news_by_category.get(category, news_by_category["general"])

            return {
                "category": category,
                "total_results": len(news_list),
                "articles": news_list,
                "timestamp": datetime.now().isoformat(),
                "note": "模拟数据 - 实际使用时需要NewsAPI密钥"
            }

        except Exception as e:
            return {"error": f"获取新闻信息失败: {str(e)}"}

    def get_random_quote(self) -> Dict[str, Any]:
        """获取随机名言（使用ZenQuotes API）"""
        try:
            # 尝试获取真实数据
            url = "https://zenquotes.io/api/random"
            response = self.session.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                if data:
                    return {
                        "quote": data[0]["q"],
                        "author": data[0]["a"],
                        "timestamp": datetime.now().isoformat()
                    }

            # 备用模拟数据
            quotes = [
                {"q": "The only way to do great work is to love what you do.", "a": "Steve Jobs"},
                {"q": "Life is what happens when you're busy making other plans.", "a": "John Lennon"},
                {"q": "The future belongs to those who believe in the beauty of their dreams.",
                 "a": "Eleanor Roosevelt"},
                {"q": "It is during our darkest moments that we must focus to see the light.", "a": "Aristotle"},
                {"q": "Whoever is happy will make others happy too.", "a": "Anne Frank"}
            ]

            import random
            quote = random.choice(quotes)

            return {
                "quote": quote["q"],
                "author": quote["a"],
                "timestamp": datetime.now().isoformat(),
                "note": "模拟数据 - 实际名言可能不同"
            }

        except Exception as e:
            return {"error": f"获取名言失败: {str(e)}"}

    def get_ip_info(self, ip_address: Optional[str] = None) -> Dict[str, Any]:
        """获取IP地址信息（使用ip-api.com）"""
        try:
            if not ip_address:
                # 获取公网IP
                response = self.session.get("https://api.ipify.org?format=json", timeout=10)
                if response.status_code == 200:
                    ip_address = response.json()["ip"]
                else:
                    # 如果无法获取公网IP，使用本地IP
                    hostname = socket.gethostname()
                    ip_address = socket.gethostbyname(hostname)

            # 查询IP信息
            url = f"http://ip-api.com/json/{ip_address}?fields=status,message,country,countryCode,region,regionName,city,zip,lat,lon,timezone,isp,org,as,query"
            response = self.session.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "success":
                    return {
                        "ip": data["query"],
                        "country": data["country"],
                        "region": data["regionName"],
                        "city": data["city"],
                        "latitude": data["lat"],
                        "longitude": data["lon"],
                        "timezone": data["timezone"],
                        "isp": data["isp"],
                        "timestamp": datetime.now().isoformat()
                    }

            # 模拟数据（当API不可用时）
            return {
                "ip": ip_address or "Unknown",
                "country": "China",
                "region": "Beijing",
                "city": "Beijing",
                "latitude": 39.9042,
                "longitude": 116.4074,
                "timezone": "Asia/Shanghai",
                "isp": "China Unicom",
                "timestamp": datetime.now().isoformat(),
                "note": "模拟数据 - 实际IP信息可能不同"
            }

        except Exception as e:
            return {"error": f"获取IP信息失败: {str(e)}"}


def format_output(data: Dict[str, Any], api_type: str) -> str:
    """格式化输出数据"""
    if "error" in data:
        return f"错误: {data['error']}"

    output_lines = []

    if api_type == "weather":
        output_lines.append("天气信息")
        output_lines.append("=" * 40)
        output_lines.append(f"城市: {data.get('city', 'N/A')}")
        output_lines.append(f"温度: {data.get('temperature', 'N/A')}°C")
        output_lines.append(f"湿度: {data.get('humidity', 'N/A')}%")
        output_lines.append(f"天气: {data.get('description', 'N/A')}")
        output_lines.append(f"风速: {data.get('wind_speed', 'N/A')} m/s")
        output_lines.append(f"气压: {data.get('pressure', 'N/A')} hPa")

    elif api_type == "exchange":
        output_lines.append("汇率信息")
        output_lines.append("=" * 40)
        output_lines.append(f"基准货币: {data.get('base', 'N/A')}")
        output_lines.append(f"汇率日期: {data.get('date', 'N/A')}")
        output_lines.append("汇率:")
        for currency, rate in data.get('rates', {}).items():
            output_lines.append(f"  {data['base']} 1 = {currency} {rate:.4f}")

    elif api_type == "news":
        output_lines.append("新闻头条")
        output_lines.append("=" * 40)
        output_lines.append(f"分类: {data.get('category', 'N/A')}")
        output_lines.append(f"新闻数量: {data.get('total_results', 0)}")
        output_lines.append("")
        for i, article in enumerate(data.get('articles', []), 1):
            output_lines.append(f"{i}. {article.get('title', 'N/A')}")
            output_lines.append(f"   来源: {article.get('source', 'N/A')}")

    elif api_type == "quote":
        output_lines.append("随机名言")
        output_lines.append("=" * 40)
        output_lines.append(f"\"{data.get('quote', 'N/A')}\"")
        output_lines.append(f"—— {data.get('author', 'N/A')}")

    elif api_type == "ipinfo":
        output_lines.append("IP地址信息")
        output_lines.append("=" * 40)
        output_lines.append(f"IP地址: {data.get('ip', 'N/A')}")
        output_lines.append(f"国家: {data.get('country', 'N/A')}")
        output_lines.append(f"地区: {data.get('region', 'N/A')}")
        output_lines.append(f"城市: {data.get('city', 'N/A')}")
        output_lines.append(f"坐标: {data.get('latitude', 'N/A')}, {data.get('longitude', 'N/A')}")
        output_lines.append(f"时区: {data.get('timezone', 'N/A')}")
        output_lines.append(f"ISP: {data.get('isp', 'N/A')}")

    if "note" in data:
        output_lines.append("")
        output_lines.append(f"备注: {data['note']}")

    output_lines.append("")
    output_lines.append(f"查询时间: {data.get('timestamp', datetime.now().isoformat())}")

    return "\n".join(output_lines)


def main():
    parser = argparse.ArgumentParser(description='API数据查询工具')
    parser.add_argument('--api-type', '-t', required=True,
                        choices=['weather', 'exchange', 'news', 'quote', 'ipinfo'],
                        help='API类型: weather(天气), exchange(汇率), news(新闻), quote(名言), ipinfo(IP信息)')
    parser.add_argument('--city', '-c', help='城市名称（仅weather类型需要）')
    parser.add_argument('--base-currency', '-b', help='基础货币代码（仅exchange类型需要）')
    parser.add_argument('--target-currency', '-r', default='CNY', help='目标货币代码（exchange类型，默认CNY）')
    parser.add_argument('--category', '-g', default='general',
                        choices=['general', 'technology', 'business', 'sports'],
                        help='新闻分类（仅news类型需要，默认general）')
    parser.add_argument('--ip-address', '-i', help='IP地址（仅ipinfo类型需要，默认当前IP）')

    args = parser.parse_args()

    print("API数据查询工具")
    print("=" * 60)

    fetcher = APIFetcher()
    result = {}

    try:
        if args.api_type == 'weather':
            if not args.city:
                print("错误: weather类型需要--city参数")
                sys.exit(1)
            result = fetcher.get_weather(args.city)

        elif args.api_type == 'exchange':
            if not args.base_currency:
                print("错误: exchange类型需要--base-currency参数")
                sys.exit(1)
            result = fetcher.get_exchange_rate(args.base_currency, args.target_currency)

        elif args.api_type == 'news':
            result = fetcher.get_news_headlines(args.category)

        elif args.api_type == 'quote':
            result = fetcher.get_random_quote()

        elif args.api_type == 'ipinfo':
            result = fetcher.get_ip_info(args.ip_address)

        # 格式化并输出结果
        formatted_output = format_output(result, args.api_type)
        print(formatted_output)

        # 如果有错误，返回非零退出码
        if "error" in result:
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n查询被用户中断。")
        sys.exit(1)
    except Exception as e:
        print(f"\n查询过程中发生错误: {e}")
        sys.exit(1)

    print("=" * 60)
    print("查询完成")


if __name__ == "__main__":
    main()