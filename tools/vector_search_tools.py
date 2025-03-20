"""使用该工具需要先运行api_server.py,此工具是基于已经持久化的HTTP的向量检索工具"""

import os
import time
import asyncio
import requests
import logfire
from typing import Dict, List, Any, Optional, Union, Tuple

from agents import function_tool
from tools.logger import get_logger

# 获取向量检索工具日志记录器
logger = get_logger("vector_search_tools")

# 默认配置
DEFAULT_CONFIG = {
    "host": "127.0.0.1",
    "port": 8000,
    "top_k": 5,
    "auto_collection_name": True,
    "vector_collection_name": "default_collection"
}

# 尝试从配置文件中加载设置
try:
    import tomli
    with open("config.toml", "rb") as f:
        config = tomli.load(f)
    
    # 从配置文件中提取配置
    RAG_SERVER_CONFIG = config.get("rag_server", {})
    TRACING_CONFIG = config.get("tracing", {})
    
    # 更新默认配置
    for key in DEFAULT_CONFIG:
        if key in RAG_SERVER_CONFIG:
            DEFAULT_CONFIG[key] = RAG_SERVER_CONFIG[key]
            
    # 获取跟踪配置
    USE_LOGFIRE = TRACING_CONFIG.get("use_logfire", False)
except Exception as e:
    logger.warning(f"加载配置文件失败: {e}，使用默认配置")
    USE_LOGFIRE = False

# 构建RAG服务器URL
RAG_SERVER_HOST = DEFAULT_CONFIG.get("host")
RAG_SERVER_PORT = DEFAULT_CONFIG.get("port")
RAG_SERVER_URL = f"http://{RAG_SERVER_HOST}:{RAG_SERVER_PORT}"
RAG_SERVER_AUTO_COLLECTION_NAME = DEFAULT_CONFIG.get("auto_collection_name")
RAG_SERVER_VECTOR_COLLECTION_NAME = DEFAULT_CONFIG.get("vector_collection_name")
TOP_K = DEFAULT_CONFIG.get("top_k")

# 配置Logfire（如果启用）
if USE_LOGFIRE:
    try:
        # 配置Logfire
        logfire_project = TRACING_CONFIG.get("logfire_project")
        logfire_api_key = TRACING_CONFIG.get("logfire_api_key")
        
        # 设置环境变量
        os.environ["LOGFIRE_PROJECT"] = logfire_project
        os.environ["LOGFIRE_API_KEY"] = logfire_api_key
        
        # 配置Logfire
        logfire.configure()
        
        # 启用HTTPX追踪
        logfire.instrument_httpx(capture_all=True)
        
        logger.info("Logfire追踪已启用")
    except Exception as e:
        logger.error(f"Logfire配置失败: {e}")
        USE_LOGFIRE = False


class HTTPRetriever:
    """HTTP检索器，通过HTTP请求获取向量检索结果"""
    
    def __init__(self, 
                 server_url: str = RAG_SERVER_URL,
                 collection_name: str = '',
                 top_k: int = TOP_K,
                 check_health: bool = True):
        """初始化HTTP检索器
        
        Args:
            server_url: RAG服务器URL
            collection_name: 集合名称
            top_k: 返回结果数量
            check_health: 是否在初始化时检查服务器健康状态
        """
        self.server_url = server_url
        self.collection_name = collection_name
        self.top_k = top_k
        self.health_checked = False
        self.last_health_check_time = 0
        self.health_check_interval = 60  # 健康检查间隔，单位为秒
        
        # 检查服务器是否可用
        if check_health:
            try:
                response = requests.get(f"{self.server_url}/health")
                if response.status_code != 200:
                    logger.warning(f"RAG服务器健康检查失败: {response.status_code}")
                else:
                    logger.info("RAG服务器连接成功")
                    self.health_checked = True
                    self.last_health_check_time = time.time()
            except Exception as e:
                logger.warning(f"RAG服务器连接失败: {e}")
    
    def _check_health(self) -> bool:
        """检查服务器健康状态
        
        Returns:
            服务器是否健康
        """
        # 如果已经检查过健康状态且未超过检查间隔，直接返回
        current_time = time.time()
        if self.health_checked and (current_time - self.last_health_check_time) < self.health_check_interval:
            return True
        
        try:
            logger.debug(f"检查RAG服务器健康状态: {self.server_url}/health")
            response = requests.get(f"{self.server_url}/health")
            if response.status_code != 200:
                logger.warning(f"RAG服务器健康检查失败: {response.status_code}")
                return False
            else:
                logger.debug("RAG服务器健康检查成功")
                self.health_checked = True
                self.last_health_check_time = current_time
                return True
        except Exception as e:
            logger.warning(f"RAG服务器健康检查失败: {e}")
            return False
    
    async def _async_check_health(self) -> bool:
        """异步检查服务器健康状态
        
        Returns:
            服务器是否健康
        """
        # 如果已经检查过健康状态且未超过检查间隔，直接返回
        current_time = time.time()
        if self.health_checked and (current_time - self.last_health_check_time) < self.health_check_interval:
            return True
        
        try:
            from httpx import AsyncClient
            logger.debug(f"异步检查RAG服务器健康状态: {self.server_url}/health")
            async with AsyncClient() as client:
                response = await client.get(f"{self.server_url}/health")
                if response.status_code != 200:
                    logger.warning(f"RAG服务器健康检查失败: {response.status_code}")
                    return False
                else:
                    logger.debug("RAG服务器健康检查成功")
                    self.health_checked = True
                    self.last_health_check_time = current_time
                    return True
        except Exception as e:
            logger.warning(f"RAG服务器健康检查失败: {e}")
            return False
    
    def retrieve(self, query: str, collection_name: str = None, top_k: int = None) -> Dict[str, Any]:
        """从知识库中检索信息
        
        Args:
            query: 查询文本
            collection_name: 集合名称，如果为None则使用默认集合
            top_k: 返回结果数量，如果为None则使用默认值
            
        Returns:
            检索结果
        """
        # 先检查服务器健康状态
        if not self._check_health():
            return {"error": "RAG服务器不可用", "documents": [], "metadatas": [], "scores": []}
        
        # 使用传入的参数或默认值
        collection = collection_name or self.collection_name
        k = top_k or self.top_k
        
        # 构建请求URL
        url = f"{self.server_url}/retrieve"
        
        # 构建请求参数
        params = {
            "query": query,
            "top_k": k
        }
        
        # 如果指定了集合名称，添加到参数中
        if collection:
            params["collection_name"] = collection
        
        try:
            logger.info(f"发送检索请求: {url}, 参数: {params}")
            # 发送请求
            response = requests.post(url, json=params)
            
            # 检查响应状态
            if response.status_code != 200:
                logger.error(f"检索失败: {response.status_code}, {response.text}")
                return {"error": f"检索失败: {response.status_code}", "documents": [], "metadatas": [], "scores": []}
            
            # 解析响应
            result = response.json()
            
            # 检查响应格式
            if "documents" not in result or "metadatas" not in result or "scores" not in result:
                logger.error(f"检索结果格式错误: {result}")
                return {"error": "检索结果格式错误", "documents": [], "metadatas": [], "scores": []}
            
            return result
        except Exception as e:
            logger.error(f"检索请求失败: {e}")
            return {"error": f"检索请求失败: {e}", "documents": [], "metadatas": [], "scores": []}
    
    async def async_retrieve(self, query: str, collection_name: str = None, top_k: int = None) -> Dict[str, Any]:
        """异步从知识库中检索信息
        
        Args:
            query: 查询文本
            collection_name: 集合名称，如果为None则使用默认集合
            top_k: 返回结果数量，如果为None则使用默认值
            
        Returns:
            检索结果
        """
        # 先检查服务器健康状态
        if not await self._async_check_health():
            return {"error": "RAG服务器不可用", "documents": [], "metadatas": [], "scores": []}
        
        # 使用传入的参数或默认值
        collection = collection_name or self.collection_name
        k = top_k or self.top_k
        
        # 构建请求URL
        url = f"{self.server_url}/retrieve"
        
        # 构建请求参数
        params = {
            "query": query,
            "top_k": k
        }
        
        # 如果指定了集合名称，添加到参数中
        if collection:
            params["collection_name"] = collection
        
        try:
            from httpx import AsyncClient, ReadTimeout
            logger.info(f"异步发送检索请求: {url}, 参数: {params}")
            # 添加重试逻辑
            max_retries = 3
            for retry in range(max_retries):
                try:
                    # 发送请求
                    async with AsyncClient(timeout=30.0) as client:
                        response = await client.post(url, json=params)
                        
                        # 检查响应状态
                        if response.status_code != 200:
                            try:
                                error_detail = response.json().get("detail", "未知错误")
                            except:
                                error_detail = response.text
                            
                            logger.error(f"检索失败: 状态码={response.status_code}, 错误详情={error_detail}")
                            return {"error": f"检索失败: {error_detail}", "documents": [], "metadatas": [], "scores": []}
                        
                        # 解析响应
                        result = response.json()
                        
                        # 检查响应格式
                        if "documents" not in result or "metadatas" not in result or "scores" not in result:
                            logger.error(f"检索结果格式错误: {result}")
                            return {"error": "检索结果格式错误", "documents": [], "metadatas": [], "scores": []}
                        
                        return result
                except ReadTimeout:
                    if retry < max_retries - 1:  # 不是最后一次尝试
                        await asyncio.sleep(1)  # 等待1秒后重试
                    else:
                        raise  # 重试次数用完，继续抛出异常
        except Exception as e:
            error_str = str(e)
            error_repr = repr(e)
            logger.error(f"检索请求失败: {error_str}, 详细错误: {error_repr}")
            return {"error": f"检索请求失败: {error_str}", "documents": [], "metadatas": [], "scores": []}
    
    def list_collections(self) -> List[str]:
        """获取所有集合列表
        
        Returns:
            集合列表
        """
        # 先检查服务器健康状态
        if not self._check_health():
            return []
        
        try:
            logger.info(f"获取集合列表: {self.server_url}/collections")
            response = requests.get(f"{self.server_url}/collections")
            
            # 检查响应状态
            if response.status_code != 200:
                logger.error(f"获取集合列表失败: {response.status_code}, {response.text}")
                return []
            
            # 解析响应
            result = response.json()
            return result.get("collections", [])
        except Exception as e:
            logger.error(f"获取集合列表请求失败: {e}")
            return []
    
    async def async_list_collections(self) -> List[str]:
        """异步获取所有集合列表
        
        Returns:
            集合列表
        """
        # 先检查服务器健康状态
        if not await self._async_check_health():
            return []
        
        try:
            from httpx import AsyncClient
            logger.info(f"异步获取集合列表: {self.server_url}/collections")
            async with AsyncClient() as client:
                response = await client.get(f"{self.server_url}/collections")
                
                # 检查响应状态
                if response.status_code != 200:
                    logger.error(f"获取集合列表失败: {response.status_code}, {response.text}")
                    return []
                
                # 解析响应
                result = response.json()
                return result.get("collections", [])
        except Exception as e:
            logger.error(f"获取集合列表请求失败: {e}")
            return []


# 全局变量，用于存储检索器实例
_retriever_instance = None

def get_retriever(server_url: str = None, collection_name: str = None, top_k: int = None, check_health: bool = True) -> HTTPRetriever:
    """获取检索器实例（单例模式）
    
    Args:
        server_url: RAG服务器URL，如果为None则使用默认URL
        collection_name: 集合名称，如果为None则使用默认集合
        top_k: 返回结果数量，如果为None则使用默认值
        check_health: 是否在初始化时检查服务器健康状态
        
    Returns:
        检索器实例
    """
    global _retriever_instance
    
    if _retriever_instance is None:
        logger.info(f"初始化检索器: URL={server_url or RAG_SERVER_URL}, 集合={collection_name or '默认'}")
        _retriever_instance = HTTPRetriever(
            server_url=server_url or RAG_SERVER_URL,
            collection_name=collection_name or '',
            top_k=top_k or TOP_K,
            check_health=check_health
        )
    
    return _retriever_instance


@function_tool
def retriever_tool(query: str, collection_name: str = None, top_k: int = None) -> str:
    """从知识库中检索信息
    
    Args:
        query: 查询文本
        collection_name: 集合名称，如果为None则使用默认集合
        top_k: 返回结果数量，如果为None则使用默认值
        
    Returns:
        检索结果
    """
    retriever = get_retriever()
    
    # 记录检索请求，使用长等号框起来
    logger.info("=" * 50)
    logger.info(f"开始检索: 查询='{query}', 集合='{collection_name or '默认'}', top_k={top_k or TOP_K}")
    logger.info("=" * 50)
    
    # 使用Logfire追踪检索过程（如果启用）
    if USE_LOGFIRE:
        try:
            with logfire.span(f"检索: {query}", tags={"query": query, "collection": collection_name or "default", "top_k": top_k or TOP_K}):
                # 使用HTTP检索方法
                results = retriever.retrieve(query, collection_name, top_k)
                
                # 记录检索结果数量
                doc_count = len(results.get("documents", []))
                logger.info(f"检索到 {doc_count} 条结果，查询: {query}")
        except Exception as e:
            logger.warning(f"Logfire追踪检索过程失败: {e}")
            # 如果追踪失败，仍然执行检索
            results = retriever.retrieve(query, collection_name, top_k)
    else:
        # 使用HTTP检索方法
        results = retriever.retrieve(query, collection_name, top_k)
    
    # 检查是否有错误
    if "error" in results and results["error"]:
        error_msg = f"检索失败: {results['error']}"
        logger.error(error_msg)
        return error_msg
    
    # 记录检索结果详情
    doc_count = len(results.get("documents", []))
    logger.info(f"检索成功: 找到 {doc_count} 条结果，查询: '{query}'")
    
    # 记录每个检索结果的基本信息
    for i, (doc, meta, score) in enumerate(zip(results['documents'], results['metadatas'], results['scores'])):
        source = meta.get('source', '未知')
        doc_preview = doc[:100] + "..." if len(doc) > 100 else doc
        logger.info(f"检索结果 #{i+1}: 来源='{source}', 分数={score:.4f}, 内容预览='{doc_preview}'")
    
    # 结束检索，使用长等号框起来
    logger.info("=" * 50)
    logger.info(f"检索完成: 查询='{query}'")
    logger.info("=" * 50)
    
    # 构建结果字符串
    result_str = f"检索到 {len(results['documents'])} 条结果:\n\n"
    
    for i, (doc, meta, score) in enumerate(zip(results['documents'], results['metadatas'], results['scores'])):
        result_str += f"[{i+1}] 来源: {meta.get('source', '未知')} (分数: {score:.4f})\n"
        result_str += f"{doc}\n\n"
    
    return result_str


@function_tool
def list_collections_tool() -> str:
    """获取所有可用的知识库集合列表
    
    Returns:
        集合列表
    """
    retriever = get_retriever()
    
    # 使用Logfire追踪获取集合列表过程（如果启用）
    if USE_LOGFIRE:
        try:
            with logfire.span("获取集合列表", tags={"operation": "list_collections"}):
                # 获取集合列表
                collections = retriever.list_collections()
                
                # 记录集合数量
                logger.info(f"获取到 {len(collections)} 个集合")
        except Exception as e:
            logger.warning(f"Logfire追踪获取集合列表过程失败: {e}")
            # 如果追踪失败，仍然执行获取集合列表
            collections = retriever.list_collections()
    else:
        # 获取集合列表
        collections = retriever.list_collections()
    
    # 检查是否有集合
    if not collections:
        return "当前没有可用的知识库集合。"
    
    # 构建结果字符串
    result_str = f"找到 {len(collections)} 个知识库集合:\n\n"
    
    for i, collection in enumerate(collections):
        result_str += f"[{i+1}] {collection}\n"
    
    return result_str


@function_tool
def final_answer(answer: str) -> str:
    """提供最终答案
    
    Args:
        answer: 最终答案文本
        
    Returns:
        最终答案
    """
    # 记录最终答案
    logger.info(f"生成最终答案: {answer}")
    
    # 使用Logfire追踪最终答案过程（如果启用）
    if USE_LOGFIRE:
        try:
            with logfire.span("生成最终答案", tags={"answer_length": len(answer)}):
                logger.info(f"生成最终答案，长度: {len(answer)}")
                return answer
        except Exception as e:
            logger.warning(f"Logfire追踪最终答案过程失败: {e}")
            return answer
    else:
        return answer


@function_tool
async def async_retriever_tool(query: str, collection_name: str = None, top_k: int = None) -> str:
    """从知识库中检索信息（异步版本）
    
    Args:
        query: 查询文本
        collection_name: 集合名称，如果为None则使用默认集合
        top_k: 返回结果数量，如果为None则使用默认值
        
    Returns:
        检索结果
    """
    retriever = get_retriever()
    
    # 记录检索请求
    logger.debug(f"异步检索请求: 查询='{query}', 集合='{collection_name or '默认'}', top_k={top_k or '默认'}")
    
    # 执行检索
    start_time = time.time()
    result = await retriever.async_retrieve(query, collection_name, top_k)
    elapsed_time = time.time() - start_time
    
    # 检查是否有错误
    if "error" in result and result["error"]:
        error_msg = f"检索失败: {result['error']}"
        logger.error(error_msg)
        return error_msg
    
    # 获取检索结果
    documents = result.get("documents", [])
    metadatas = result.get("metadatas", [])
    scores = result.get("scores", [])
    
    # 检查结果是否为空
    if not documents:
        no_results_msg = "未找到相关信息"
        logger.warning(f"检索结果为空: {no_results_msg}")
        return no_results_msg
    
    # 构建响应
    response = []
    for i, (doc, meta, score) in enumerate(zip(documents, metadatas, scores)):
        # 添加文档信息
        doc_info = f"文档 {i+1} (相关度: {score:.4f}):"
        response.append(doc_info)
        
        # 添加元数据信息（如果有）
        if meta:
            meta_info = []
            for key, value in meta.items():
                if key in ["source", "title", "filename", "page", "chunk_id"]:
                    meta_info.append(f"{key}: {value}")
            if meta_info:
                response.append("元数据: " + ", ".join(meta_info))
        
        # 添加文档内容
        response.append(f"内容: {doc}")
        response.append("---")
    
    # 添加检索统计信息
    response.append(f"检索到 {len(documents)} 条结果，耗时 {elapsed_time:.2f} 秒")
    
    # 记录检索完成
    logger.info(f"异步检索完成: 查询='{query}', 找到 {len(documents)} 条结果, 耗时 {elapsed_time:.2f} 秒")
    
    return "\n".join(response)


@function_tool
async def async_list_collections_tool() -> str:
    """获取所有可用的知识库集合列表（异步版本）
    
    Returns:
        集合列表
    """
    retriever = get_retriever()
    
    # 记录请求
    logger.info("获取集合列表请求")
    
    # 获取集合列表
    start_time = time.time()
    collections = await retriever.async_list_collections()
    elapsed_time = time.time() - start_time
    
    # 检查结果是否为空
    if not collections:
        no_collections_msg = "未找到任何知识库集合"
        logger.warning(f"集合列表为空: {no_collections_msg}")
        return no_collections_msg
    
    # 构建响应
    response = [f"找到 {len(collections)} 个知识库集合:"]
    for i, collection in enumerate(collections):
        response.append(f"{i+1}. {collection}")
    
    # 添加统计信息
    response.append(f"\n获取集合列表耗时 {elapsed_time:.2f} 秒")
    
    # 记录完成
    logger.info(f"获取集合列表完成: 找到 {len(collections)} 个集合, 耗时 {elapsed_time:.2f} 秒")
    
    return "\n".join(response)
