# RAG 工具集

这个目录包含了一系列用于实现检索增强生成（Retrieval-Augmented Generation, RAG）的工具。本工具集设计用于处理和管理大规模文档知识库，支持多种文档格式和处理策略。

## 功能特性

- 📄 支持多种文件格式：
  - 文本文件：TXT、Markdown
  - 办公文档：PDF、DOCX、PPTX
  - 结构化数据：CSV、Excel、JSON
  - 网页内容：HTML、XML
- 🧹 文本清洗和预处理
  - 去除特殊字符
  - 统一编码格式
  - 段落重组
- ✂️ 智能文本分块
  - 基于语义的分块
  - 重叠分块策略
  - 自定义分块规则
- 🔤 文本嵌入
  - 支持多种嵌入模型
  - 批量处理优化
  - 缓存机制
- 💾 向量数据库管理
  - 高效存储和检索
  - 元数据管理
  - 增量更新支持
- 🎯 检索优化
  - 语义重排序
  - 相关度评分
  - 上下文扩展

## 配置说明

在 `config.toml` 中配置 RAG 相关参数：

```toml
[rag]
vector_db_path = "./data/vector_db"
collection_name = "default"
embedding_model = "shibing624/text2vec-base-chinese"
rerank_model = "maidalun1020/bce-reranker-base_v1"
use_reranker = true
top_k = 5

[rag_server]
host = "127.0.0.1"
port = 65533
```

## 使用方法

### 处理单个文件

```python
from tools.rag_tools.document_processor import DocumentProcessor

# 创建文档处理器
processor = DocumentProcessor(collection_name="my_collection")

# 处理单个文件
result = processor.process_file("path/to/file.txt", delete_existing=True)
```

### 处理多个文件

```python
from tools.rag_tools.document_processor import DocumentProcessor

# 创建文档处理器
processor = DocumentProcessor(collection_name="my_collection")

# 处理多个文件
file_paths = ["path/to/file1.txt", "path/to/file2.pdf", "path/to/file3.docx"]
results = processor.process_files(file_paths, delete_existing=True, single_thread=True)
```

### 处理文件夹

```python
from tools.rag_tools.document_processor import DocumentProcessor

# 创建文档处理器
processor = DocumentProcessor(collection_name="my_collection")

# 处理文件夹
results = processor.process_directory(
    "path/to/directory", 
    extensions=['.txt', '.md', '.pdf'], 
    delete_existing=True,
    single_thread=True
)
```

### 检索文档（推荐方式）

推荐直接使用VectorRetriever进行检索，这样可以避免不必要的资源消耗：

```python
from tools.rag_tools.vector_retriever import VectorRetriever

# 创建检索器
retriever = VectorRetriever()

# 检索
query = "你的查询"
results = retriever.retrieve(query, collection_name="my_collection", top_k=3)

# 输出结果
for i, (doc, meta, score) in enumerate(zip(results["documents"], results["metadatas"], results["scores"])):
    print(f"结果 {i+1} (分数: {score:.4f}):")
    print(f"来源: {meta.get('source', '未知')}")
    print(f"内容: {doc}")
    print()
```

### 检索文档（通过DocumentProcessor）

也可以通过DocumentProcessor进行检索，但这会初始化更多组件：

```python
from tools.rag_tools.document_processor import DocumentProcessor

# 创建文档处理器
processor = DocumentProcessor(collection_name="my_collection")

# 检索
query = "你的查询"
results = processor.retrieve(query, top_k=3)

# 输出结果
for i, (doc, meta, score) in enumerate(zip(results["documents"], results["metadatas"], results["scores"])):
    print(f"结果 {i+1} (分数: {score:.4f}):")
    print(f"来源: {meta.get('source', '未知')}")
    print(f"内容: {doc}")
    print()
```

### 直接嵌入文本

```python
from tools.rag_tools.text_embedder import TextEmbedder

# 创建嵌入器
embedder = TextEmbedder()

# 嵌入文本
text = "这是一段示例文本"
metadata = {"source": "示例来源"}
embedder.embed_text(text, collection_name="my_collection", metadata=metadata)
```

### 命令行使用

也可以通过命令行使用这些工具：

```bash
# 处理文件
python -m tools.rag_tools.document_processor --action process_file --input path/to/file.txt --collection_name my_collection

# 处理目录
python -m tools.rag_tools.document_processor --action process_directory --input path/to/directory --collection_name my_collection --extensions .txt .md .pdf

# 检索文档
python -m tools.rag_tools.document_processor --action retrieve --query "你的查询" --collection_name my_collection

# 列出所有集合
python -m tools.rag_tools.document_processor --action list_collections

# 删除集合
python -m tools.rag_tools.document_processor --action delete_collection --collection_name my_collection
```

### 使用REST API服务

我们提供了一个基于FastAPI的REST API服务，可以通过HTTP请求进行文件处理、向量化和检索：

#### 启动服务

```bash
python -m tools.rag_tools.rag_server --host 127.0.0.1 --port 8000
```

服务启动后，可以通过浏览器访问API文档：`http://127.0.0.1:8000/docs`

#### API接口

1. **健康检查**
   ```
   GET /health
   ```

2. **列出所有集合**
   ```
   GET /collections
   ```

3. **处理单个文件**
   ```
   POST /process
   ```
   参数：
   - `file`: 文件（表单数据）
   - `collection_name`: 集合名称（可选）
   - `delete_existing`: 是否删除已存在的集合（可选）

4. **处理目录**
   ```
   POST /process_directory
   ```
   参数：
   - `directory_path`: 目录路径（表单数据）
   - `collection_name`: 集合名称（可选）
   - `delete_existing`: 是否删除已存在的集合（可选）
   - `extensions`: 文件扩展名列表（可选）
   - `single_thread`: 是否使用单线程处理（可选）

5. **处理多个文件**
   ```
   POST /process_files
   ```
   参数：
   - `file_paths`: 文件路径列表（JSON数据）
   - `collection_name`: 集合名称（可选）
   - `delete_existing`: 是否删除已存在的集合（可选）
   - `single_thread`: 是否使用单线程处理（可选）

6. **直接嵌入文本**
   ```
   POST /embed_text
   ```
   参数：
   - `text`: 文本内容（JSON数据）
   - `collection_name`: 集合名称（可选）
   - `metadata`: 元数据（可选）

7. **检索**
   ```
   POST /retrieve
   ```
   参数：
   - `query`: 查询文本
   - `collection_name`: 集合名称（可选）
   - `top_k`: 返回结果数量（可选）

8. **删除集合**
   ```
   DELETE /collection/{collection_name}
   ```
   参数：
   - `collection_name`: 集合名称（路径参数）

#### 使用示例

**Python示例**：

```python
import requests

# 处理文件
with open("path/to/file.txt", "rb") as f:
    response = requests.post(
        "http://127.0.0.1:8000/process",
        files={"file": f},
        data={"collection_name": "my_collection", "delete_existing": "true"}
    )
print(response.json())

# 处理目录
response = requests.post(
    "http://127.0.0.1:8000/process_directory",
    data={
        "directory_path": "path/to/directory",
        "collection_name": "my_collection",
        "delete_existing": "true",
        "extensions": [".txt", ".md", ".pdf"],
        "single_thread": "true"
    }
)
print(response.json())

# 检索
response = requests.post(
    "http://127.0.0.1:8000/retrieve",
    json={"query": "你的查询", "collection_name": "my_collection", "top_k": 3}
)
results = response.json()
for i, (doc, meta, score) in enumerate(zip(results["documents"], results["metadatas"], results["scores"])):
    print(f"结果 {i+1} (分数: {score:.4f}):")
    print(f"来源: {meta.get('source', '未知')}")
    print(f"内容: {doc}")
    print()
```

**curl示例**：

```bash
# 处理文件
curl -X POST http://127.0.0.1:8000/process \
  -F "file=@path/to/file.txt" \
  -F "collection_name=my_collection" \
  -F "delete_existing=true"

# 处理目录
curl -X POST http://127.0.0.1:8000/process_directory \
  -F "directory_path=path/to/directory" \
  -F "collection_name=my_collection" \
  -F "delete_existing=true" \
  -F "extensions=.txt" \
  -F "extensions=.md" \
  -F "extensions=.pdf" \
  -F "single_thread=true"

# 检索
curl -X POST http://127.0.0.1:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "你的查询", "collection_name": "my_collection", "top_k": 3}'
```

## 测试结果

我们进行了多个文件嵌入到同一个向量数据库的测试，结果表明：

1. 系统能够成功地将多个文件嵌入到同一个向量数据库。
2. 通用查询能够返回所有相关文档。
3. 特定查询能够准确地返回最相关的文档，并且分数很高。

测试脚本位于 `tools/rag_tools/test_multiple_files.py`，可以运行该脚本查看测试结果。

## 注意事项

1. 在处理多个文件时，建议使用 `single_thread=True` 参数，避免多线程冲突。
2. 如果需要处理大量文件，建议使用批处理方式，避免内存溢出。
3. 在嵌入文本时，建议使用唯一的 ID，避免 ID 冲突。
4. 检索文档时，推荐直接使用VectorRetriever，这样可以避免不必要的资源消耗。
5. 使用REST API服务时，模型只会加载一次，大大提高了检索效率。

## 最佳实践

### 文档处理

1. 文件预处理
   ```python
   from tools.rag_tools.document_processor import DocumentProcessor
   
   processor = DocumentProcessor(
       collection_name="knowledge_base",
       chunk_size=500,
       chunk_overlap=50,
       use_async=True  # 启用异步处理
   )
   
   # 处理大型文档时使用多线程
   processor.process_directory(
       "docs/",
       extensions=['.pdf', '.docx', '.txt'],
       single_thread=False,
       batch_size=10
   )
   ```

2. 自定义分块策略
   ```python
   from tools.rag_tools.text_splitter import TextSplitter
   
   splitter = TextSplitter(
       chunk_size=500,
       chunk_overlap=50,
       separator="\n\n",  # 按段落分割
       min_chunk_size=100  # 最小分块大小
   )
   ```

### 向量检索优化

1. 使用重排序提高精度
   ```python
   from tools.rag_tools.vector_retriever import VectorRetriever
   
   retriever = VectorRetriever(
       use_reranker=True,
       rerank_top_k=10,
       use_cache=True  # 启用缓存
   )
   
   results = retriever.retrieve(
       query="查询内容",
       collection_name="knowledge_base",
       top_k=3
   )
   ```

2. 批量检索优化
   ```python
   # 批量查询
   queries = ["问题1", "问题2", "问题3"]
   results = retriever.batch_retrieve(
       queries,
       collection_name="knowledge_base",
       top_k=3,
       batch_size=5  # 控制批处理大小
   )
   ```

### REST API 最佳实践

1. 文件上传（支持异步）
   ```python
   import aiohttp
   import asyncio
   
   async def upload_file(file_path: str, collection_name: str):
       async with aiohttp.ClientSession() as session:
           with open(file_path, "rb") as f:
               response = await session.post(
                   "http://localhost:8000/process",
                   data={
                       "collection_name": collection_name,
                       "chunk_size": 500,
                       "chunk_overlap": 50,
                       "use_async": "true"
                   },
                   files={"file": f}
               )
           return await response.json()
   ```

2. 批量处理（支持异步）
   ```python
   async def process_directory(dir_path: str, collection_name: str):
       async with aiohttp.ClientSession() as session:
           response = await session.post(
               "http://localhost:8000/process_directory",
               json={
                   "directory_path": dir_path,
                   "collection_name": collection_name,
                   "extensions": [".pdf", ".docx", ".txt"],
                   "single_thread": False,
                   "batch_size": 10,
                   "use_async": True
               }
           )
           return await response.json()
   ```

## 性能优化建议

1. 文档处理
   - 使用异步处理大文件
   - 启用多线程处理大量文件
   - 使用适当的分块大小（推荐300-500字）
   - 根据文档类型选择合适的分块策略
   - 启用批处理模式
   - 使用流式处理避免内存溢出

2. 向量检索
   - 使用异步批处理
   - 启用结果缓存
   - 适当调整top_k参数
   - 针对特定场景优化重排序
   - 使用向量数据库索引
   - 定期优化向量数据库

3. 系统配置
   - 合理设置并发限制
   - 优化内存使用
   - 定期清理缓存
   - 监控系统性能
   - 使用连接池
   - 启用压缩传输

## 故障排除

1. 内存使用过高
   - 减小批处理大小
   - 使用流式处理
   - 清理向量缓存
   - 使用内存监控
   - 定期释放资源

2. 处理速度慢
   - 检查文件大小
   - 优化分块策略
   - 调整并发参数
   - 使用性能分析工具
   - 优化数据库查询

3. 检索质量问题
   - 验证文档分块
   - 调整重排序参数
   - 检查嵌入模型
   - 分析相似度分数
   - 优化提示词

## 开发计划

- [ ] 支持更多文件格式
- [ ] 添加增量更新功能
- [ ] 优化检索算法
- [ ] 添加文档质量评估
- [ ] 支持分布式处理
- [ ] 添加向量数据库迁移工具
- [ ] 实现自动化测试
- [ ] 优化内存管理
- [ ] 添加性能监控
- [ ] 支持多语言处理

## 更新日志

### v1.0.0
- 初始版本发布
- 支持基本文档处理
- 实现向量检索功能

### v1.1.0
- 添加重排序功能
- 优化文本分块
- 改进API接口
- 添加异步处理支持
- 优化内存使用
- 添加缓存机制

## 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 发起 Pull Request

## 许可证

MIT License 