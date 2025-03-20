# 智能问答系统

基于向量检索增强生成（RAG）的智能问答系统，支持流式响应和会话日志。本系统适用于构建企业级知识库问答系统，可以快速接入自定义数据源。

## 主要特性

- 🚀 基于 RAG 技术，提供高质量的问答响应
- 💬 支持流式输出，提供更好的用户体验
- 📝 完整的会话日志记录和追踪
- 🔑 API 密钥认证和访问控制
- 🔄 支持多种 LLM 模型接入（已支持 Qwen）
- 📊 可配置的 RAG 检索参数
- 🎯 支持重排序，提高检索精度
- 🛠️ 提供完整的文档处理工具集
- 🔌 兼容 Dify API 格式

## 项目结构

```
.
├── api_server.py          # FastAPI服务器（主入口）
├── chat_llm/             # 聊天LLM模块
├── chat_llm_prompt/      # 提示词模板
├── chat_llm_api_server/  # LLM API服务
├── tools/                # 工具集
│   └── rag_tools/       # RAG相关工具
├── config.toml           # 配置文件
├── data/                 # 数据目录
│   └── vector_db/       # 向量数据库
└── logs/                 # 日志目录
    └── session_logs/    # 会话日志
```

## 配置文件

项目使用 `config.toml` 作为统一配置文件，包含以下部分：

### 服务器配置

```toml
[server]
host = "0.0.0.0"
port = 8000
log_dir = "./logs"
timeout_keep_alive = 120
timeout_graceful_shutdown = 10
limit_concurrency = 20
```

### API密钥配置

```toml
[api_keys]
admin = "admin-key"
user = "user-key"
```

### LLM模型配置

```toml
[llm]
model = "qwen-plus"
api_key = "your-api-key"
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
system_prompt = "你是一个知识库助手。"
```

### RAG检索配置

```toml
[rag]
vector_db_path = "./data/vector_db"
collection_name = "default"
embedding_model = "shibing624/text2vec-base-chinese"
rerank_model = "maidalun1020/bce-reranker-base_v1"
use_reranker = true
top_k = 5
```

### 流式响应配置

```toml
[streaming]
continue_on_disconnect = false  # 断开连接时是否继续处理
```

## 快速开始

1. 克隆项目：
   ```bash
   git clone [项目地址]
   cd [项目目录]
   ```

2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

3. 配置系统：
   - 复制 `config.toml.example` 为 `config.toml`
   - 修改配置文件中的相关参数，特别是 API 密钥

4. 准备知识库：
   - 使用 `tools/rag_tools` 中的工具处理文档
   - 将处理后的文档存入向量数据库

5. 运行API服务器：
   ```bash
   python api_server.py
   ```

## API使用

### 标准聊天API

```
POST /chat/conversation
```

请求体：
```json
{
  "questions": ["你好，请问如何使用这个系统？"],
  "stream": true,
  "session_id": "optional-session-id",
  "temperature": 0.7,
  "top_p": 0.9
}
```

请求头：
```
X-API-Key: your-api-key
```

### Dify 兼容API

```
POST /dify/chat-messages
```

请求体：
```json
{
  "query": "你好，请问如何使用这个系统？",
  "response_mode": "streaming",
  "conversation_id": "optional-conversation-id",
  "user": "optional-user-id"
}
```

请求头：
```
Authorization: Bearer your-api-key
```

### 系统状态API

```
GET /rag-status      # 检查RAG服务状态
GET /rag-agent-status  # 检查RAG Agent状态
```

## 日志系统

项目使用了统一的日志管理系统，通过 `utils/logger.py` 模块提供日志记录功能。

## 日志配置

日志配置在 `config.toml` 文件的 `[logging]` 部分：

```toml
[logging]
level = "INFO"                # 日志级别：DEBUG, INFO, WARNING, ERROR, CRITICAL
format = "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"  # 标准日志格式
debug_format = "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d:%(funcName)s] - %(message)s"  # DEBUG级别日志格式
log_dir = "./logs"            # 日志目录
max_bytes = 10485760          # 单个日志文件最大大小（10MB）
backup_count = 5              # 保留的日志文件数量
encoding = "utf-8"            # 日志文件编码
console_output = true         # 是否输出到控制台
file_output = true            # 是否输出到文件
session_logs = true           # 是否启用会话日志
```

日志格式说明：
- `%(asctime)s`: 时间戳
- `%(name)s`: 日志记录器名称
- `%(levelname)s`: 日志级别
- `%(filename)s:%(lineno)d`: 文件名和行号，方便定位日志来源
- `%(funcName)s`: 函数名称（仅在DEBUG级别时显示）
- `%(message)s`: 日志消息内容

当日志级别设置为 `DEBUG` 时，系统会自动使用 `debug_format` 格式，显示更详细的信息，包括函数名称，便于调试。

## 使用方法

### 获取模块日志记录器

```python
from utils.logger import get_logger

# 获取模块特定的日志记录器
logger = get_logger("module_name")

# 记录日志
logger.debug("调试信息")
logger.info("普通信息")
logger.warning("警告信息")
logger.error("错误信息")
logger.critical("严重错误信息")
```

### 会话日志记录

对于聊天会话，可以使用 `SessionLogger` 类记录会话相关的日志：

```python
from utils.logger import SessionLogger

# 记录用户问题
SessionLogger.log_question(session_id, question)

# 记录AI回答
SessionLogger.log_answer_chunk(session_id, answer, is_final=True)
```

## 日志文件结构

- `logs/app.log`：主应用日志文件
- `logs/module_name.log`：模块特定的日志文件
- `logs/session_logs/session_{session_id}_{timestamp}.log`：会话特定的日志文件

## 性能优化

1. 向量检索优化：
   - 调整 `top_k` 参数
   - 开启重排序功能
   - 选择合适的嵌入模型

2. LLM响应优化：
   - 调整 temperature 和 top_p 参数
   - 优化系统提示词
   - 选择合适的模型

3. 系统性能：
   - 合理设置并发限制
   - 配置超时参数
   - 启用流式响应

## 常见问题

1. 如何提高回答准确性？
   - 确保文档预处理质量
   - 调整 RAG 参数
   - 优化提示词模板

2. 如何处理大规模文档？
   - 使用多线程处理
   - 分批导入数据
   - 优化向量数据库配置

3. 连接断开问题？
   - 检查 `timeout_keep_alive` 设置
   - 配置 `continue_on_disconnect`
   - 使用合适的客户端超时设置

## 贡献指南

欢迎提交 Issue 和 Pull Request 来帮助改进项目。

## 许可证

本项目采用 MIT 许可证。 