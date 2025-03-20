import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
import sys
from typing import Dict, Optional, Any

# 导入配置
from config import config

# 添加彩色日志支持
class ColoredFormatter(logging.Formatter):
    """
    彩色日志格式化器
    """
    # 定义ANSI颜色代码
    COLORS = {
        'DEBUG': '\033[36m',     # 青色
        'INFO': '\033[32m',      # 绿色
        'WARNING': '\033[33m',   # 黄色
        'ERROR': '\033[31m',     # 红色
        'CRITICAL': '\033[35m',  # 紫色
        'RESET': '\033[0m'       # 重置
    }
    
    def format(self, record):
        """格式化日志记录"""
        # 获取原始格式化的消息
        log_message = super().format(record)
        
        # 如果是控制台输出且启用了彩色输出，添加颜色
        if config.LOGGING_CONFIG.get("use_colors", True):
            levelname = record.levelname
            if levelname in self.COLORS:
                # 为日志级别添加颜色
                colored_levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
                # 替换日志消息中的级别名称
                log_message = log_message.replace(levelname, colored_levelname)
                
                # 为整个消息添加颜色前缀
                # log_message = f"{self.COLORS[levelname]}{log_message}{self.COLORS['RESET']}"
        
        return log_message

class LoggerManager:
    """
    日志管理器，用于统一管理项目中的日志配置和记录器
    """
    _instance = None
    _loggers = {}
    
    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super(LoggerManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """初始化日志管理器"""
        # 确保日志目录存在
        self.log_dir = config.LOGGING_CONFIG.get("log_dir", "./logs")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 创建会话日志目录
        self.session_logs_dir = os.path.join(self.log_dir, "session_logs")
        os.makedirs(self.session_logs_dir, exist_ok=True)
        
        # 获取日志配置
        self.log_level = self._get_log_level(config.LOGGING_CONFIG.get("level", "INFO"))
        
        # 根据日志级别设置不同的格式
        if self.log_level <= logging.DEBUG:
            # DEBUG级别时显示更详细的信息，包括文件名、行号和函数名
            self.log_format = config.LOGGING_CONFIG.get(
                "debug_format", 
                "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d:%(funcName)s] - %(message)s"
            )
        else:
            # 非DEBUG级别使用标准格式
            self.log_format = config.LOGGING_CONFIG.get(
                "format", 
                "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
            )
            
        self.max_bytes = config.LOGGING_CONFIG.get("max_bytes", 10*1024*1024)  # 默认10MB
        self.backup_count = config.LOGGING_CONFIG.get("backup_count", 5)
        self.encoding = config.LOGGING_CONFIG.get("encoding", "utf-8")
        self.console_output = config.LOGGING_CONFIG.get("console_output", True)
        self.file_output = config.LOGGING_CONFIG.get("file_output", True)
        self.session_logs = config.LOGGING_CONFIG.get("session_logs", True)
        self.use_colors = config.LOGGING_CONFIG.get("use_colors", True)
        
        # 创建根日志记录器
        self._setup_root_logger()
    
    def _get_log_level(self, level_str: str) -> int:
        """将日志级别字符串转换为对应的常量值"""
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        return level_map.get(level_str.upper(), logging.INFO)
    
    def _setup_root_logger(self):
        """设置根日志记录器"""
        # 获取根日志记录器
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)
        
        # 清除现有的处理器
        if root_logger.handlers:
            root_logger.handlers.clear()
        
        # 创建标准格式化器（用于文件输出）
        standard_formatter = logging.Formatter(self.log_format)
        
        # 创建彩色格式化器（用于控制台输出）
        colored_formatter = ColoredFormatter(self.log_format)
        
        # 添加控制台处理器
        if self.console_output:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self.log_level)
            # 使用彩色格式化器
            console_handler.setFormatter(colored_formatter if self.use_colors else standard_formatter)
            root_logger.addHandler(console_handler)
        
        # 添加文件处理器
        if self.file_output:
            file_handler = RotatingFileHandler(
                os.path.join(self.log_dir, "app.log"),
                maxBytes=self.max_bytes,
                backupCount=self.backup_count,
                encoding=self.encoding
            )
            file_handler.setLevel(self.log_level)
            file_handler.setFormatter(standard_formatter)
            root_logger.addHandler(file_handler)
    
    def get_logger(self, name: str, module_specific_file: bool = True) -> logging.Logger:
        """
        获取指定名称的日志记录器
        
        Args:
            name: 日志记录器名称
            module_specific_file: 是否为该模块创建特定的日志文件
            
        Returns:
            配置好的日志记录器
        """
        # 检查是否已经创建过该记录器
        if name in self._loggers:
            return self._loggers[name]
        
        # 创建新的日志记录器
        logger = logging.getLogger(name)
        logger.setLevel(self.log_level)
        
        # 如果需要模块特定的日志文件
        if module_specific_file and self.file_output:
            # 创建模块特定的文件处理器
            module_file_handler = RotatingFileHandler(
                os.path.join(self.log_dir, f"{name.replace('.', '_')}.log"),
                maxBytes=self.max_bytes,
                backupCount=self.backup_count,
                encoding=self.encoding
            )
            module_file_handler.setLevel(self.log_level)
            module_file_handler.setFormatter(logging.Formatter(self.log_format))
            logger.addHandler(module_file_handler)
        
        # 缓存并返回日志记录器
        self._loggers[name] = logger
        return logger
    
    def create_session_logger(self, session_id: str) -> logging.Logger:
        """
        创建会话特定的日志记录器
        
        Args:
            session_id: 会话ID
            
        Returns:
            会话特定的日志记录器
        """
        if not self.session_logs:
            # 如果未启用会话日志，返回普通日志记录器
            return self.get_logger("session")
        
        # 检查是否已经创建过该会话记录器
        logger_name = f"session.{session_id}"
        if logger_name in self._loggers:
            return self._loggers[logger_name]
        
        # 创建会话日志文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_log_file = os.path.join(
            self.session_logs_dir, 
            f"session_{session_id}_{timestamp}.log"
        )
        
        # 创建会话日志记录器
        logger = logging.getLogger(logger_name)
        logger.setLevel(self.log_level)
        
        # 创建会话文件处理器
        session_handler = logging.FileHandler(
            session_log_file,
            encoding=self.encoding
        )
        session_handler.setLevel(self.log_level)
        session_handler.setFormatter(logging.Formatter(self.log_format))
        logger.addHandler(session_handler)
        
        # 缓存并返回会话日志记录器
        self._loggers[logger_name] = logger
        return logger

# 创建日志管理器实例
logger_manager = LoggerManager()

def get_logger(name: str, module_specific_file: bool = True) -> logging.Logger:
    """
    获取指定名称的日志记录器
    
    Args:
        name: 日志记录器名称
        module_specific_file: 是否为该模块创建特定的日志文件
        
    Returns:
        配置好的日志记录器
    """
    return logger_manager.get_logger(name, module_specific_file)

def create_session_logger(session_id: str) -> logging.Logger:
    """
    创建会话特定的日志记录器
    
    Args:
        session_id: 会话ID
        
    Returns:
        会话特定的日志记录器
    """
    return logger_manager.create_session_logger(session_id)

# 定义会话日志记录器类，用于记录会话中的问题和回答
class SessionLogger:
    """会话日志记录器，用于记录会话中的问题和回答"""
    
    @staticmethod
    def get_logger(session_id: str) -> logging.Logger:
        """获取会话日志记录器"""
        return create_session_logger(session_id)
    
    @staticmethod
    def log_question(session_id: str, question: str):
        """记录用户问题"""
        logger = SessionLogger.get_logger(session_id)
        logger.info(f"用户问题: {question}")
    
    @staticmethod
    def log_answer_chunk(session_id: str, answer: str, is_final: bool = False):
        """记录AI回答（片段或完整）"""
        logger = SessionLogger.get_logger(session_id)
        if is_final:
            logger.info(f"AI完整回答: {answer}")
        else:
            # 只记录片段长度，避免日志过大
            logger.debug(f"AI回答片段: 长度 {len(answer)} 字符")

# 添加测试函数
def test_logger():
    """
    测试日志记录功能
    
    此函数用于演示不同级别的日志记录，以及如何在不同函数中记录日志
    """
    logger = get_logger("test_logger")
    
    logger.debug("这是一条调试信息，包含详细的代码位置")
    logger.info("这是一条普通信息")
    logger.warning("这是一条警告信息")
    logger.error("这是一条错误信息")
    logger.critical("这是一条严重错误信息")
    
    # 调用内部函数测试嵌套函数的日志记录
    def inner_function():
        logger.debug("这是内部函数中的调试信息")
        logger.info("这是内部函数中的普通信息")
    
    inner_function()
    
    return "日志测试完成，请查看日志文件"

# 如果直接运行此模块，则执行测试
if __name__ == "__main__":
    test_logger()
    print("日志测试完成，请查看日志文件和控制台输出") 