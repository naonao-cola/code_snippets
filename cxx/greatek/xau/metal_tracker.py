"""
贵金属价格监控小程序 (MetalTracker)
功能：实时抓取黄金、白银价格和汇率，计算人民币价格，并在任务栏图标显示。
支持三种模式：仅黄金、仅白银、黄金+白银（显示双图标）。
"""
import sys
import time
import threading
import re
import requests
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw, ImageFont
import pystray

# ===========================
# 1. 配置区域
# ===========================
GOLD_URL = "https://zh.tradingeconomics.com/commodity/gold"
SILVER_URL = "https://zh.tradingeconomics.com/commodity/silver"
CURRENCY_URL = "https://tradingeconomics.com/usdcny:cur"
UPDATE_INTERVAL = 5  # 刷新间隔（秒）

# 模式常量
MODE_GOLD = "gold"
MODE_SILVER = "silver"
MODE_BOTH = "both"

class MetalTracker:
    def __init__(self):
        # 状态变量
        self.price_gold_usd = None
        self.price_silver_usd = None
        self.exchange_rate = None
        
        self.price_gold_cny = "..."
        self.price_silver_cny = "..."
        
        self.current_mode = MODE_GOLD # 默认模式
        
        # 图标实例
        self.icon_gold = None
        self.icon_silver = None
        
        self.running = True
        self.refresh_event = threading.Event()
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    def request_refresh(self, icon, item):
        self.refresh_event.set()

    def set_mode(self, mode):
        def inner(icon, item):
            self.current_mode = mode
            self.update_icons_visibility()
            # 立即刷新以更新图标
            self.refresh_event.set()
        return inner

    def update_icons_visibility(self):
        # 根据当前模式更新图标的可见性
        if self.current_mode == MODE_GOLD:
            if self.icon_gold: self.icon_gold.visible = True
            if self.icon_silver: self.icon_silver.visible = False
        elif self.current_mode == MODE_SILVER:
            if self.icon_gold: self.icon_gold.visible = False
            if self.icon_silver: self.icon_silver.visible = True
        elif self.current_mode == MODE_BOTH:
            if self.icon_gold: self.icon_gold.visible = True
            if self.icon_silver: self.icon_silver.visible = True

    # ===========================
    # 3. 爬虫函数
    # ===========================
    def fetch_value(self, url, specific_id=None, target_symbol=None):
        try:
            print(f"Fetching {url}...", end=" ")
            response = self.session.get(url, timeout=5)
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')

                # Priority 0: Target Symbol Row (Most precise for tables)
                if target_symbol:
                    row = soup.find('tr', attrs={'data-symbol': target_symbol})
                    if row:
                        print(f"  -> Found Row with symbol '{target_symbol}'")
                        search_id = specific_id if specific_id else 'p'
                        element = row.find(id=search_id)
                        if element:
                            text = element.get_text(strip=True)
                            print(f"  -> Found ID '{search_id}' in row: {text}")
                            match = re.search(r"([\d,\.]+)", text)
                            if match:
                                val = float(match.group(1).replace(',', ''))
                                return val
                
                # Priority 1: Specific ID (Global search)
                if specific_id:
                    element = soup.find(id=specific_id)
                    if element:
                        text = element.get_text(strip=True)
                        print(f"  -> Found Specific ID '{specific_id}': {text}")
                        match = re.search(r"([\d,\.]+)", text)
                        if match:
                            val = float(match.group(1).replace(',', ''))
                            return val

                # Priority 2: ID 'market_last' (Currency)
                market_last = soup.find(id='market_last')
                if market_last:
                    text = market_last.text.strip()
                    match = re.search(r"([\d,\.]+)", text)
                    if match:
                        return float(match.group(1).replace(',', ''))

                # Priority 3: ID 'p' (Generic)
                element_p = soup.find(id='p')
                if element_p:
                    text = element_p.get_text(strip=True)
                    match = re.search(r"([\d,\.]+)", text)
                    if match:
                         return float(match.group(1).replace(',', ''))

        except Exception as e:
            print(f"Error fetching {url}: {e}")
        return None

    # ===========================
    # 4. 数据更新
    # ===========================
    def update_data(self):
        # 获取汇率 (总是需要)
        rate = self.fetch_value(CURRENCY_URL, specific_id='market_last')
        if rate:
            self.exchange_rate = rate
        
        # 总是获取所有数据
        gold = self.fetch_value(GOLD_URL, target_symbol='XAUUSD:CUR', specific_id='p')
        if gold:
            self.price_gold_usd = gold
            
        silver = self.fetch_value(SILVER_URL, target_symbol='XAGUSD:CUR', specific_id='p')
        if silver:
            self.price_silver_usd = silver

        # 计算人民币价格
        grams_per_oz = 31.1034768
        if self.exchange_rate:
            if self.price_gold_usd:
                p_gold = (self.price_gold_usd * self.exchange_rate) / grams_per_oz
                self.price_gold_cny = f"{p_gold:.2f}"
            
            if self.price_silver_usd:
                p_silver = (self.price_silver_usd * self.exchange_rate) / grams_per_oz
                self.price_silver_cny = f"{p_silver:.2f}"
            
            return True
        return False

    # ===========================
    # 5. 绘图逻辑
    # ===========================
    def create_image(self, type_):
        width = 64
        height = 64
        image = Image.new('RGB', (width, height), color=(0, 0, 0))
        draw = ImageDraw.Draw(image)
        
        # 字体加载辅助函数
        def get_font(size):
            try:
                return ImageFont.truetype("arialbd.ttf", size)
            except:
                try:
                    return ImageFont.truetype("arial.ttf", size)
                except:
                    return ImageFont.load_default()

        # 辅助绘制文本函数
        def draw_centered_text(text, y_center, color, max_size=40):
            # Parse int part
            try:
                val = float(text)
                display_text = str(int(val))
            except:
                display_text = text

            font_size = max_size
            font = get_font(font_size)
            
            # 自动缩小字体
            if isinstance(font, ImageFont.FreeTypeFont):
                while font_size > 8:
                    font = get_font(font_size)
                    bbox = draw.textbbox((0, 0), display_text, font=font)
                    w = bbox[2] - bbox[0]
                    h = bbox[3] - bbox[1]
                    if w < width - 2 and h < height - 2:
                        break
                    font_size -= 2
            
            bbox = draw.textbbox((0, 0), display_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            x = (width - text_width) / 2
            y = y_center - (text_height / 2)
            
            draw.text((x, y), display_text, font=font, fill=color)

        gold_color = (255, 215, 0)
        silver_color = (224, 224, 224)

        if type_ == 'gold':
            draw_centered_text(self.price_gold_cny, height/2, gold_color, 40)
        elif type_ == 'silver':
            draw_centered_text(self.price_silver_cny, height/2, silver_color, 40)

        return image

    # ===========================
    # 6. 主循环
    # ===========================
    def data_update_loop(self):
        # 初始时确保图标可见性正确
        self.update_icons_visibility()
        
        while self.running:
            success = self.update_data()
            timestamp = time.strftime("%H:%M:%S")
            
            tooltip = (f"更新: {timestamp}\n"
                       f"汇率: {self.exchange_rate}\n"
                       f"黄金: {self.price_gold_cny} 元/克 (${self.price_gold_usd})\n"
                       f"白银: {self.price_silver_cny} 元/克 (${self.price_silver_usd})")
            
            # 更新黄金图标
            if self.icon_gold:
                self.icon_gold.title = tooltip
                self.icon_gold.icon = self.create_image('gold')
                
            # 更新白银图标
            if self.icon_silver:
                self.icon_silver.title = tooltip
                self.icon_silver.icon = self.create_image('silver')
            
            self.refresh_event.wait(UPDATE_INTERVAL)
            self.refresh_event.clear()

    def on_exit(self, icon, item):
        self.running = False
        self.refresh_event.set()
        
        # 停止所有图标
        if self.icon_gold: self.icon_gold.stop()
        if self.icon_silver: self.icon_silver.stop()

    def run(self):
        # 创建菜单
        menu = pystray.Menu(
            pystray.MenuItem("Mode: Gold Only", self.set_mode(MODE_GOLD), checked=lambda item: self.current_mode == MODE_GOLD),
            pystray.MenuItem("Mode: Silver Only", self.set_mode(MODE_SILVER), checked=lambda item: self.current_mode == MODE_SILVER),
            pystray.MenuItem("Mode: Gold & Silver", self.set_mode(MODE_BOTH), checked=lambda item: self.current_mode == MODE_BOTH),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Refresh", self.request_refresh),
            pystray.MenuItem("Exit", self.on_exit)
        )

        # 创建两个图标
        # 注意：这里我们使用相同的图片和菜单，因为它们共享控制逻辑
        img_gold = self.create_image('gold')
        img_silver = self.create_image('silver')
        
        self.icon_gold = pystray.Icon("GoldTracker", img_gold, "Loading...", menu)
        self.icon_silver = pystray.Icon("SilverTracker", img_silver, "Loading...", menu)

        # 启动数据更新线程
        data_thread = threading.Thread(target=self.data_update_loop)
        data_thread.daemon = True
        data_thread.start()

        # 启动白银图标线程 (非阻塞)
        silver_thread = threading.Thread(target=self.icon_silver.run)
        silver_thread.daemon = True
        silver_thread.start()

        # 启动黄金图标 (在主线程阻塞运行)
        # 注意：如果MODE_SILVER被选中，icon_gold会被隐藏，但必须保持运行以接收菜单事件（如果用户切换回Gold）
        # 实际上，如果隐藏了，用户无法点击它。但只要至少有一个图标显示，用户就可以切换模式。
        self.icon_gold.run()

if __name__ == "__main__":
    app = MetalTracker()
    app.run()
