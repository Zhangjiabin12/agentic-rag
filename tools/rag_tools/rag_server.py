import os
from typing import Dict, List, Optional, Any
import sys
import time
from datetime import datetime
from pathlib import Path

# 首先导入配置
from config import config

# 导入日志工具
from tools.logger import get_logger

# 获取RAG服务器日志记录器
logger = get_logger("rag_server")

from fastapi import FastAPI, HTTPException, Query, Body, File, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from contextlib import asynccontextmanager
import shutil
import chromadb
import stat
import tomli

# 定义生命周期事件
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时执行
    global retriever, processor
    
    # 预加载模型
    logger.info("预加载嵌入模型...")
    get_embedder_model(config.DEFAULT_EMBEDDING_MODEL)
    
    logger.info("预加载重排序模型...")
    get_reranker_model(config.DEFAULT_RERANK_MODEL)
    
    # 初始化检索器
    logger.info("初始化检索器...")
    retriever = VectorRetriever()
    
    # 初始化文档处理器
    logger.info("初始化文档处理器...")
    processor = DocumentProcessor()
    
    # 确保上传目录存在
    os.makedirs("./uploads", exist_ok=True)
    
    logger.info("初始化完成，服务已准备就绪")
    
    yield
    
    # 关闭时执行
    logger.info("服务关闭...")

# 创建FastAPI应用
app = FastAPI(title="RAG API", description="检索增强生成API服务", lifespan=lifespan)

# 然后导入其他模块
from tools.rag_tools.vector_retriever import VectorRetriever, get_embedder_model, get_reranker_model
from tools.rag_tools.document_processor import DocumentProcessor

# 添加中间件记录请求
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # 记录请求信息
    logger.debug(f"请求开始: {request.method} {request.url.path}")
    
    # 处理请求
    response = await call_next(request)
    
    # 计算处理时间
    process_time = time.time() - start_time
    
    # 记录响应信息
    logger.debug(f"请求完成: {request.method} {request.url.path} - 状态码: {response.status_code} - 处理时间: {process_time:.4f}秒")
    
    return response

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头
)

# 全局变量
retriever = None
processor = None

# 请求和响应模型
class RetrieveRequest(BaseModel):
    query: str
    collection_name: Optional[str] = None
    top_k: Optional[int] = None

class RetrieveResponse(BaseModel):
    query: str
    documents: List[str]
    metadatas: List[Dict]
    scores: List[float]

class CollectionsResponse(BaseModel):
    collections: List[str]

class HealthResponse(BaseModel):
    status: str

class ProcessResponse(BaseModel):
    file_path: str
    collection_name: str
    chunk_count: int
    message: str

# 定义配置更新请求模型
class ConfigUpdateRequest(BaseModel):
    """配置更新请求模型"""
    section: str
    key: str
    value: Any

# 定义配置更新响应模型
class ConfigUpdateResponse(BaseModel):
    """配置更新响应模型"""
    success: bool
    message: str
    config: Dict[str, Any]

def handle_config_update(config_loader):
    """处理配置更新
    
    Args:
        config_loader: 配置加载器实例
    """
    # 重新初始化检索器
    global retriever, processor
    logger.info("重新初始化检索器...")
    retriever = VectorRetriever()
    
    # 重新初始化文档处理器
    logger.info("重新初始化文档处理器...")
    processor = DocumentProcessor()
    
    logger.info("配置更新处理完成")

# 注册配置更新观察者
config.add_observer(handle_config_update)

@app.get("/health", response_model=HealthResponse)
def health_check():
    """健康检查接口"""
    logger.debug("健康检查请求")
    return {"status": "ok"}

@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(request: RetrieveRequest):
    """检索接口"""
    global retriever
    
    # 获取请求参数
    query = request.query
    collection_name = request.collection_name or config.COLLECTION_NAME
    top_k = request.top_k or config.TOP_K
    
    logger.info(f"检索请求: 查询='{query}', 集合='{collection_name}', top_k={top_k}")
    
    # 检索
    try:
        results = retriever.retrieve(query, collection_name, top_k)
        logger.info(f"检索成功: 找到 {len(results['documents'])} 条结果")
        return results
    except Exception as e:
        error_msg = f"检索失败: {str(e)}"
        logger.error(f"检索错误: {error_msg}, 详细错误: {repr(e)}")
        # 抛出HTTP异常，详细的错误信息会被记录并返回
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/collections", response_model=CollectionsResponse)
def list_collections():
    """列出所有集合"""
    global retriever
    
    logger.debug("列出集合请求")
    
    try:
        collections = retriever.list_collections()
        logger.info(f"列出集合成功: 找到 {len(collections)} 个集合")
        return {"collections": collections}
    except Exception as e:
        error_msg = f"列出集合失败: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/process", response_model=ProcessResponse)
async def process_file(
    file: UploadFile = File(...),
    collection_name: str = Form(None),
    delete_existing: bool = Form(False)
):
    """处理文件并向量化"""
    global processor
    
    # 使用默认集合名称
    if not collection_name:
        collection_name = config.COLLECTION_NAME
    
    logger.info(f"处理文件请求: 文件='{file.filename}', 集合='{collection_name}', 删除现有={delete_existing}")
    
    try:
        # 保存上传的文件
        file_path = f"./uploads/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"文件已保存: {file_path}")
        
        # 创建一个新的处理器实例，使用指定的集合名称
        file_processor = DocumentProcessor(collection_name=collection_name)
        
        # 处理文件
        start_time = time.time()
        result = file_processor.process_file(file_path, delete_existing=delete_existing)
        process_time = time.time() - start_time
        
        logger.info(f"文件处理成功: 集合='{result['collection_name']}', 分块数={result['chunk_count']}, 处理时间={process_time:.2f}秒")
        
        return {
            "file_path": file_path,
            "collection_name": result["collection_name"],
            "chunk_count": result["chunk_count"],
            "message": f"文件处理成功，生成了 {result['chunk_count']} 个分块"
        }
    except Exception as e:
        error_msg = f"处理文件失败: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.delete("/collection/{collection_name}")
def delete_collection(collection_name: str):
    """删除集合"""
    global retriever
    
    logger.info(f"删除集合请求: 集合='{collection_name}'")
    
    try:
        # 使用ChromaDB的根路径
        settings = chromadb.Settings(allow_reset=True)
        client = chromadb.PersistentClient(path=config.VECTOR_DB_PATH, settings=settings)
        
        try:
            # 删除集合
            client.delete_collection(name=collection_name)
            logger.info(f"已通过ChromaDB删除集合: {collection_name}")
        except Exception as e:
            logger.warning(f"删除集合失败: {e}")
        
        # 关闭客户端连接
        client.reset()
        del client
        
        # 等待一小段时间确保连接完全关闭
        time.sleep(1)
        
        # 删除集合目录
        collection_path = os.path.join(config.VECTOR_DB_PATH, collection_name)
        if os.path.exists(collection_path):
            def handle_remove_readonly(func, path, exc):
                os.chmod(path, stat.S_IWRITE)
                func(path)
            
            shutil.rmtree(collection_path, onexc=handle_remove_readonly)
            logger.info(f"集合删除成功: {collection_name}")
            return {"message": f"成功删除集合: {collection_name}"}
        else:
            logger.warning(f"集合目录不存在: {collection_path}")
            return {"message": f"集合目录不存在: {collection_name}"}
            
    except Exception as e:
        error_msg = f"删除集合失败: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/process_directory", response_model=ProcessResponse)
async def process_directory(
    directory_path: str = Form(...),
    collection_name: str = Form(None),
    delete_existing: bool = Form(False),
    extensions: List[str] = Form(None),
    single_thread: bool = Form(False)
):
    """处理目录中的所有文件并向量化"""
    global processor
    
    # 使用默认集合名称
    if not collection_name:
        collection_name = config.COLLECTION_NAME
    
    # 使用默认扩展名
    if not extensions:
        extensions = ['.txt', '.md', '.pdf', '.docx', '.html', '.csv', '.xlsx', '.xls']
    
    logger.info(f"处理目录请求: 目录='{directory_path}', 集合='{collection_name}', 删除现有={delete_existing}, 扩展名={extensions}, 单线程={single_thread}")
    
    try:
        # 创建一个新的处理器实例，使用指定的集合名称
        dir_processor = DocumentProcessor(collection_name=collection_name)
        
        # 处理目录
        start_time = time.time()
        results = dir_processor.process_directory(
            directory_path, 
            extensions=extensions, 
            delete_existing=delete_existing,
            single_thread=single_thread
        )
        process_time = time.time() - start_time
        
        # 计算总分块数
        total_chunks = sum(result.get('chunk_count', 0) for result in results)
        
        logger.info(f"目录处理成功: 集合='{dir_processor.collection_name}', 文件数={len(results)}, 分块数={total_chunks}, 处理时间={process_time:.2f}秒")
        
        return {
            "file_path": directory_path,
            "collection_name": dir_processor.collection_name,
            "chunk_count": total_chunks,
            "message": f"目录处理成功，共处理了 {len(results)} 个文件，生成了 {total_chunks} 个分块"
        }
    except Exception as e:
        error_msg = f"处理目录失败: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/process_files", response_model=ProcessResponse)
async def process_files(
    files: List[UploadFile] = File(...),
    collection_name: str = Form(None),
    delete_existing: bool = Form(False),
    single_thread: bool = Form(False),
    group_by_type: bool = Form(True),  # 默认按文件类型分组
    use_filename_prefix: bool = Form(True)  # 默认使用文件名作为集合前缀
):
    """处理多个文件并向量化
    
    Args:
        files: 要处理的文件列表
        collection_name: 集合名称（可选）
        delete_existing: 是否删除现有集合
        single_thread: 是否使用单线程处理
        group_by_type: 是否按文件类型分组创建集合
        use_filename_prefix: 是否使用文件名作为集合前缀
    """
    logger.info(f"处理多个文件请求: 文件数={len(files)}, 集合='{collection_name}', "
               f"删除现有={delete_existing}, 单线程={single_thread}, "
               f"按类型分组={group_by_type}, 使用文件名前缀={use_filename_prefix}")
    
    try:
        # 保存上传的文件并按类型分组
        file_groups = {}
        for file in files:
            file_path = f"./uploads/{file.filename}"
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            if group_by_type:
                # 获取文件扩展名
                ext = os.path.splitext(file.filename)[1].lower()
                if ext not in file_groups:
                    file_groups[ext] = []
                file_groups[ext].append((file_path, file.filename))
            else:
                # 不分组，所有文件放在一起
                if 'all' not in file_groups:
                    file_groups['all'] = []
                file_groups['all'].append((file_path, file.filename))
        
        total_chunks = 0
        results = []
        
        # 处理每个分组
        for group_key, group_files in file_groups.items():
            # 确定集合名称
            group_collection_name = collection_name or config.COLLECTION_NAME
            
            if group_by_type:
                # 使用文件类型作为集合后缀
                group_collection_name = f"{group_collection_name}_{group_key[1:]}"  # 去掉扩展名的点
            
            if use_filename_prefix and len(group_files) == 1:
                # 单个文件使用文件名作为集合前缀
                filename_prefix = os.path.splitext(group_files[0][1])[0]
                group_collection_name = f"{filename_prefix}_{group_collection_name}"
            
            # 创建处理器实例
            processor = DocumentProcessor(collection_name=group_collection_name)
            
            # 获取该组的文件路径
            group_file_paths = [f[0] for f in group_files]
            
            # 处理文件组
            logger.info(f"处理文件组 {group_key}: 集合名称={group_collection_name}, 文件数={len(group_file_paths)}")
            group_results = processor.process_files(
                group_file_paths,
                delete_existing=delete_existing,
                single_thread=single_thread
            )
            
            # 累计分块数
            group_chunks = sum(r.get('chunk_count', 0) for r in group_results)
            total_chunks += group_chunks
            
            results.extend(group_results)
            logger.info(f"文件组 {group_key} 处理完成: 生成 {group_chunks} 个分块")
        
        # 返回处理结果
        return {
            "file_path": str([f[0] for group in file_groups.values() for f in group]),
            "collection_name": "multiple_collections" if group_by_type else collection_name or config.COLLECTION_NAME,
            "chunk_count": total_chunks,
            "message": f"文件处理成功，共处理了 {len(results)} 个文件，生成了 {total_chunks} 个分块，"
                      f"分布在 {len(file_groups)} 个集合中" if group_by_type else f"在同一个集合中"
        }
        
    except Exception as e:
        error_msg = f"处理文件失败: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/embed_text", response_model=Dict)
async def embed_text(
    text: str = Body(...),
    collection_name: str = Body(None),
    metadata: Dict = Body(None)
):
    """直接嵌入文本"""
    from tools.rag_tools.text_embedder import TextEmbedder
    
    # 使用默认集合名称
    if not collection_name:
        collection_name = config.COLLECTION_NAME
    
    # 使用默认元数据
    if metadata is None:
        metadata = {"source": "API嵌入"}
    
    logger.info(f"嵌入文本请求: 文本长度={len(text)}, 集合='{collection_name}'")
    
    try:
        # 创建嵌入器
        embedder = TextEmbedder()
        
        # 嵌入文本
        start_time = time.time()
        embedder.embed_text(text, collection_name=collection_name, metadata=metadata)
        process_time = time.time() - start_time
        
        logger.info(f"文本嵌入成功: 集合='{collection_name}', 文本长度={len(text)}, 处理时间={process_time:.2f}秒")
        
        return {
            "message": f"文本嵌入成功，集合名称: {collection_name}",
            "collection_name": collection_name,
            "text_length": len(text)
        }
    except Exception as e:
        error_msg = f"嵌入文本失败: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)


def init_app():
    """初始化应用"""
    global retriever, processor
    
    # 预加载模型
    logger.info("预加载嵌入模型...")
    get_embedder_model(config.DEFAULT_EMBEDDING_MODEL)
    
    logger.info("预加载重排序模型...")
    get_reranker_model(config.DEFAULT_RERANK_MODEL)
    
    # 初始化检索器
    logger.info("初始化检索器...")
    retriever = VectorRetriever()
    
    # 初始化文档处理器
    logger.info("初始化文档处理器...")
    processor = DocumentProcessor()
    
    # 确保上传目录存在
    os.makedirs("./uploads", exist_ok=True)
    
    logger.info("初始化完成，服务已准备就绪")
    
    return app

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='RAG服务')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='主机地址')
    parser.add_argument('--port', type=int, default=8000, help='端口号')
    parser.add_argument('--reload', action='store_true', help='是否开启热重载')
    
    args = parser.parse_args()
    
    # 记录启动信息
    logger.info(f"启动RAG服务: host={args.host}, port={args.port}, reload={args.reload}")
    
    # 启动服务
    try:
        uvicorn.run("tools.rag_tools.rag_server:app", host=args.host, port=args.port, reload=args.reload)
    except Exception as e:
        logger.error(f"服务启动失败: {str(e)}")
        raise 