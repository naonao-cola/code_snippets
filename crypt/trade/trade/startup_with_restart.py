#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
邢不行™️选币实盘框架 - 带自动重启功能的启动脚本
Python数字货币量化投资课程

版权所有 ©️ 邢不行
微信: xbx8662

在原有startup.py基础上增加自动重启功能
包含内存监控、定时重启、异常重启等功能

Author: 邢不行
"""

import gc
import os
import psutil
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

# 导入原有模块
from startup import main as original_main
from config import error_webhook_url
from core.utils.dingding import send_wechat_work_msg

# ====================================================================================================
# ** USER CONFIG - 自动重启配置区域 **
# ====================================================================================================

# 1. 内存监控配置
ENABLE_MEMORY_MONITOR = True     # 是否启用内存监控
MEMORY_WARNING_MB = 1936         # 内存警告阈值（MB）
MEMORY_CRITICAL_MB = 2048        # 内存临界阈值（MB），超过将重启
MEMORY_CHECK_INTERVAL = 300      # 内存检查间隔（秒）

# 2. 定时重启配置
ENABLE_SCHEDULED_RESTART = True  # 是否启用定时重启
RESTART_INTERVAL_HOURS = 24      # 重启间隔（小时）
RESTART_TIME_HOUR = 3            # 每日重启时间（小时）
RESTART_TIME_MINUTE = 30         # 每日重启时间（分钟）

# 3. 异常重启配置
MAX_CONSECUTIVE_ERRORS = 5       # 最大连续错误次数
ERROR_RESET_INTERVAL = 3600      # 错误计数重置间隔（秒）
RESTART_COOLDOWN = 300           # 重启冷却时间（秒）

# 4. 运行时间限制
MAX_RUNTIME_HOURS = 24           # 最大运行时间（小时），超过将重启

# 5. 垃圾回收配置
FORCE_GC_INTERVAL = 600          # 强制垃圾回收间隔（秒）
GC_THRESHOLD_RATIO = 0.8         # 垃圾回收阈值比例

# ====================================================================================================

class RestartManager:
    """
    重启管理器 - 轻量级版本
    专门为startup.py设计的重启功能
    """
    
    def __init__(self):
        self.start_time = datetime.now()
        self.last_restart_time = datetime.now()
        self.last_memory_check = 0
        self.last_gc_time = 0
        self.consecutive_errors = 0
        self.last_error_time = None
        self.restart_count = 0
        
        # 创建日志目录
        self.log_dir = Path('logs/restart')
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.log("🚀 重启管理器初始化完成")
    
    def log(self, message: str):
        """日志输出"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        
        # 写入日志文件
        try:
            log_file = self.log_dir / f"restart_{datetime.now().strftime('%Y%m%d')}.log"
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(log_message + '\n')
        except Exception:
            pass  # 忽略日志写入错误
    
    def get_memory_usage(self) -> float:
        """获取当前进程内存使用量（MB）"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / 1024 / 1024
        except Exception:
            return 0
    
    def get_system_memory_info(self) -> dict:
        """获取系统内存信息"""
        try:
            memory = psutil.virtual_memory()
            return {
                'total_mb': memory.total / 1024 / 1024,
                'available_mb': memory.available / 1024 / 1024,
                'percent': memory.percent
            }
        except Exception:
            return {}
    
    def force_garbage_collection(self):
        """强制垃圾回收"""
        try:
            # 获取回收前的内存
            memory_before = self.get_memory_usage()
            
            # 执行垃圾回收
            collected = gc.collect()
            
            # 获取回收后的内存
            memory_after = self.get_memory_usage()
            memory_freed = memory_before - memory_after
            
            if memory_freed > 10:  # 只有释放超过10MB才记录
                self.log(f"🧹 垃圾回收: 释放 {memory_freed:.1f}MB 内存，回收 {collected} 个对象")
            
            self.last_gc_time = time.time()
            
        except Exception as e:
            self.log(f"垃圾回收失败: {e}")
    
    def check_memory_status(self) -> str:
        """检查内存状态"""
        if not ENABLE_MEMORY_MONITOR:
            return "normal"
        
        current_time = time.time()
        if current_time - self.last_memory_check < MEMORY_CHECK_INTERVAL:
            return "normal"
        
        self.last_memory_check = current_time
        
        # 检查进程内存
        process_memory = self.get_memory_usage()
        system_memory = self.get_system_memory_info()
        
        # 记录内存状态
        if system_memory:
            self.log(f"📊 内存状态: 进程 {process_memory:.1f}MB, 系统可用 {system_memory.get('available_mb', 0):.1f}MB ({system_memory.get('percent', 0):.1f}%)")
        
        # 检查是否需要垃圾回收
        if (current_time - self.last_gc_time > FORCE_GC_INTERVAL or 
            process_memory > MEMORY_WARNING_MB * GC_THRESHOLD_RATIO):
            self.force_garbage_collection()
        
        # 检查是否需要重启
        if process_memory > MEMORY_CRITICAL_MB:
            return "critical"
        elif process_memory > MEMORY_WARNING_MB:
            return "warning"
        
        # 检查系统内存
        if system_memory.get('available_mb', float('inf')) < 256:
            return "critical"
        
        return "normal"
    
    def should_restart_by_schedule(self) -> bool:
        """检查是否需要定时重启"""
        if not ENABLE_SCHEDULED_RESTART:
            return False
        
        now = datetime.now()
        
        # 检查运行时间
        runtime_hours = (now - self.start_time).total_seconds() / 3600
        if runtime_hours >= MAX_RUNTIME_HOURS:
            self.log(f"⏰ 达到最大运行时间: {runtime_hours:.1f}h >= {MAX_RUNTIME_HOURS}h")
            return True
        
        # 检查间隔重启
        if RESTART_INTERVAL_HOURS > 0:
            time_since_restart = (now - self.last_restart_time).total_seconds() / 3600
            if time_since_restart >= RESTART_INTERVAL_HOURS:
                self.log(f"⏰ 达到重启间隔: {time_since_restart:.1f}h >= {RESTART_INTERVAL_HOURS}h")
                return True
        
        # 检查每日重启时间
        if RESTART_TIME_HOUR is not None:
            current_hour = now.hour
            current_minute = now.minute
            
            # 检查是否在重启时间窗口内（±2分钟）
            target_minutes = RESTART_TIME_HOUR * 60 + RESTART_TIME_MINUTE
            current_minutes = current_hour * 60 + current_minute
            
            if abs(current_minutes - target_minutes) <= 2:
                # 检查今天是否已经重启过
                if self.last_restart_time.date() < now.date():
                    self.log(f"⏰ 达到每日重启时间: {RESTART_TIME_HOUR:02d}:{RESTART_TIME_MINUTE:02d}")
                    return True
        
        return False
    
    def record_error(self):
        """记录错误"""
        now = datetime.now()
        
        # 如果距离上次错误超过重置间隔，重置错误计数
        if (self.last_error_time and 
            (now - self.last_error_time).total_seconds() > ERROR_RESET_INTERVAL):
            self.consecutive_errors = 0
        
        self.consecutive_errors += 1
        self.last_error_time = now
        
        self.log(f"❌ 记录错误，连续错误次数: {self.consecutive_errors}")
    
    def should_restart_by_errors(self) -> bool:
        """检查是否需要因错误而重启"""
        return self.consecutive_errors >= MAX_CONSECUTIVE_ERRORS
    
    def can_restart(self) -> bool:
        """检查是否可以重启"""
        now = datetime.now()
        time_since_restart = (now - self.last_restart_time).total_seconds()
        
        if time_since_restart < RESTART_COOLDOWN:
            remaining = RESTART_COOLDOWN - time_since_restart
            self.log(f"⏳ 重启冷却中，剩余 {remaining:.0f} 秒")
            return False
        
        return True
    
    def prepare_restart(self, reason: str):
        """准备重启"""
        self.log(f"🔄 准备重启，原因: {reason}")
        
        # 发送通知
        if error_webhook_url:
            try:
                msg = f"🔄 系统自动重启\n原因: {reason}\n时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n重启次数: {self.restart_count + 1}"
                send_wechat_work_msg(msg, error_webhook_url)
            except Exception as e:
                self.log(f"发送重启通知失败: {e}")
        
        # 强制垃圾回收
        self.force_garbage_collection()
        
        # 更新重启信息
        self.last_restart_time = datetime.now()
        self.restart_count += 1
        self.consecutive_errors = 0  # 重置错误计数
        
        self.log(f"✅ 重启准备完成，即将重启 (第 {self.restart_count} 次)")
        
        # 等待一段时间
        time.sleep(3)
    
    def check_restart_conditions(self) -> tuple[bool, str]:
        """检查所有重启条件"""
        # 检查内存状态
        memory_status = self.check_memory_status()
        if memory_status == "critical":
            return True, "内存使用过高"
        
        # 检查定时重启
        if self.should_restart_by_schedule():
            return True, "定时重启"
        
        # 检查错误重启
        if self.should_restart_by_errors():
            return True, f"连续错误次数过多 ({self.consecutive_errors})"
        
        return False, ""


def enhanced_main():
    """
    增强版main函数，包含重启管理功能
    """
    restart_manager = RestartManager()
    
    restart_manager.log("🎯 启动增强版选币框架")
    restart_manager.log(f"⚙️ 配置: 内存监控={ENABLE_MEMORY_MONITOR}, 定时重启={ENABLE_SCHEDULED_RESTART}")
    restart_manager.log(f"📊 阈值: 内存警告={MEMORY_WARNING_MB}MB, 内存临界={MEMORY_CRITICAL_MB}MB")
    
    while True:
        try:
            # 检查重启条件
            should_restart, restart_reason = restart_manager.check_restart_conditions()
            if should_restart and restart_manager.can_restart():
                restart_manager.prepare_restart(restart_reason)
                # 重启程序
                os.execv(sys.executable, [sys.executable] + sys.argv)
            
            # 执行原始main函数
            result = original_main()
            
            # 如果main函数正常返回，说明一个周期完成
            if result:
                restart_manager.log(f"✅ 周期完成，下次运行时间: {result.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # 重置错误计数（成功执行一次）
            if restart_manager.consecutive_errors > 0:
                restart_manager.log("🔄 重置错误计数")
                restart_manager.consecutive_errors = 0
            
        except KeyboardInterrupt:
            restart_manager.log("⌨️ 收到中断信号，正在退出...")
            break
            
        except Exception as err:
            # 记录错误
            restart_manager.record_error()
            
            error_msg = f'❌系统出错 (第{restart_manager.consecutive_errors}次)，原因: {str(err)}'
            restart_manager.log(error_msg)
            restart_manager.log(traceback.format_exc())
            
            # 发送错误通知
            if error_webhook_url:
                try:
                    full_msg = f"{error_msg}\n连续错误: {restart_manager.consecutive_errors}/{MAX_CONSECUTIVE_ERRORS}"
                    send_wechat_work_msg(full_msg, error_webhook_url)
                except Exception as e:
                    restart_manager.log(f"发送错误通知失败: {e}")
            
            # 检查是否需要因错误重启
            if restart_manager.should_restart_by_errors():
                if restart_manager.can_restart():
                    restart_manager.prepare_restart(f"连续错误次数过多 ({restart_manager.consecutive_errors})")
                    os.execv(sys.executable, [sys.executable] + sys.argv)
                else:
                    restart_manager.log("🚫 无法重启，可能达到重启限制")
                    break
            
            # 等待后重试
            wait_time = min(11 + restart_manager.consecutive_errors * 5, 60)  # 递增等待时间
            restart_manager.log(f"⏳ 等待 {wait_time} 秒后重试...")
            time.sleep(wait_time)
        
        finally:
            # 定期强制垃圾回收
            current_time = time.time()
            if current_time - restart_manager.last_gc_time > FORCE_GC_INTERVAL:
                restart_manager.force_garbage_collection()
    
    restart_manager.log(f"📊 运行统计: 总重启次数 {restart_manager.restart_count}")
    restart_manager.log("👋 程序退出")


if __name__ == '__main__':
    print("🚀 邢不行选币框架 - 增强重启版")
    print("=====================================")
    print(f"📊 配置信息:")
    print(f"   内存监控: {ENABLE_MEMORY_MONITOR} (警告: {MEMORY_WARNING_MB}MB, 临界: {MEMORY_CRITICAL_MB}MB)")
    print(f"   定时重启: {ENABLE_SCHEDULED_RESTART} (间隔: {RESTART_INTERVAL_HOURS}h, 时间: {RESTART_TIME_HOUR:02d}:{RESTART_TIME_MINUTE:02d})")
    print(f"   最大错误: {MAX_CONSECUTIVE_ERRORS} 次")
    print(f"   最大运行: {MAX_RUNTIME_HOURS} 小时")
    print("=====================================")
    
    enhanced_main()
