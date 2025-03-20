"""
配置模块，包含所有配置项
"""
import os
import tomli
from pathlib import Path

class Config:
    """配置类"""
    
    _instance = None
    _observers = []
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.config_path = Path("config.toml")
        self.config = self._load_config()
        self._initialize_config()
        self._initialized = True
    
    def _load_config(self):
        """加载配置文件"""
        try:
            with open(self.config_path, "rb") as f:
                return tomli.load(f)
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            return {}
    
    def _initialize_config(self):
        """初始化配置"""
        # 服务器配置
        self.SERVER_CONFIG = self.config.get("server", {
            "host": "0.0.0.0",
            "port": 8000,
            "log_dir": "./logs",
            "timeout_keep_alive": 120,
            "timeout_graceful_shutdown": 10,
            "limit_concurrency": 20
        })
        
        # API密钥配置
        self.API_KEYS_CONFIG = self.config.get("api_keys", {
            "admin": "admin-key",
            "user": "user-key"
        })
        
        # LLM配置
        self.LLM_CONFIG = self.config.get("llm", {})
        
        # RAG配置
        self.RAG_CONFIG = self.config.get("rag", {})

        # LLM_AGENT配置
        self.LLM_AGENT_CONFIG = self.config.get("llm_agent", {})
        
        # 日志配置
        self.LOGGING_CONFIG = self.config.get("logging", {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
            "debug_format": "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d:%(funcName)s] - %(message)s",
            "log_dir": "./logs",
            "max_bytes": 10*1024*1024,  # 10MB
            "backup_count": 5,
            "encoding": "utf-8",
            "console_output": True,
            "file_output": True,
            "session_logs": True,
            "use_colors": True  # 是否在控制台输出中使用彩色
        })
        
        # 设置RAG相关配置项
        self.DATASET_PATH = self.RAG_CONFIG.get("dataset_path", "./data/tmp/dataset")
        self.SPLIT_PATH = self.RAG_CONFIG.get("split_path", "./data/tmp/split")
        self.EMBEDDING_CACHE_PATH = self.RAG_CONFIG.get("embedding_cache_path", "./data/tmp/embedding_cache")
        self.VECTOR_DB_PATH = self.RAG_CONFIG.get("vector_db_path", "./data/vector_db")
        self.COLLECTION_NAME = self.RAG_CONFIG.get("collection_name", "default")
        self.DEFAULT_EMBEDDING_MODEL = self.RAG_CONFIG.get("default_embedding_model", "BAAI/bge-m3")
        self.DEFAULT_RERANK_MODEL = self.RAG_CONFIG.get("default_rerank_model", "BAAI/bge-reranker-v2-m3")
        self.USE_RERANKER = self.RAG_CONFIG.get("use_reranker", True)
        self.TOP_K = self.RAG_CONFIG.get("top_k", 5)
        self.CHUNK_SIZE = self.RAG_CONFIG.get("chunk_size", 300)
        self.CHUNK_OVERLAP = self.RAG_CONFIG.get("chunk_overlap", 48)
        self.SPLIT_METHOD = self.RAG_CONFIG.get("split_method", "separator")
        self.DEFAULT_SEPARATORS = self.RAG_CONFIG.get("default_separators", ["\n\n", "\n", "。", "，", ".", ",", " "])
        
        # 确保目录存在
        os.makedirs(self.DATASET_PATH, exist_ok=True)
        os.makedirs(self.SPLIT_PATH, exist_ok=True)
        os.makedirs(self.EMBEDDING_CACHE_PATH, exist_ok=True)
        os.makedirs(self.VECTOR_DB_PATH, exist_ok=True)
        
        # 设备配置
        import torch
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.USE_GPU = self.DEVICE == "cuda"
        
        # 模型列表
        self.EMBEDDING_MODELS = self.config.get("model_list", {}).get("embedding_models", [])
        self.RERANK_MODELS = self.config.get("model_list", {}).get("rerank_models", [])
        self.LLM_MODELS = self.config.get("model_list", {}).get("llm_models", [])
    
    def add_observer(self, observer):
        """添加观察者"""
        if observer not in self._observers:
            self._observers.append(observer)
    
    def remove_observer(self, observer):
        """移除观察者"""
        if observer in self._observers:
            self._observers.remove(observer)
    
    def notify_observers(self):
        """通知所有观察者"""
        for observer in self._observers:
            observer(self)
    
    def reload_config(self) -> bool:
        """重新加载配置"""
        try:
            self.config = self._load_config()
            self._initialize_config()
            self.notify_observers()
            return True
        except Exception as e:
            print(f"重新加载配置失败: {e}")
            return False

# 创建全局配置实例
config = Config() 