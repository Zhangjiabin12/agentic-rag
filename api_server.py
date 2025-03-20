import json
import os
import uuid
import tomli
import toml  # 添加toml模块导入
from datetime import datetime
import time
import asyncio
from pathlib import Path
import sys
from typing import Dict, Any, List, Callable, Optional
import signal
import subprocess
import socket

from fastapi import FastAPI, HTTPException, Request, Depends, Header, Response, Body, Query, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, AsyncGenerator, Annotated, Any, Union
import uvicorn
from contextlib import asynccontextmanager

# 首先导入配置
from config import config

# 导入日志工具
from tools.logger import get_logger, SessionLogger

# 获取API服务器日志记录器
logger = get_logger("api_server")

# 导入其他模块
from chat_llm.llm import AsyncChatSession
from chat_llm.rag_agent import RAGAgent

# 定义请求模型
class ConversationRequest(BaseModel):
    """会话请求模型"""
    questions: List[str]
    session_id: Optional[str] = None
    stream: bool = False

# 定义API密钥头部
API_KEY_NAME = "Authorization"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# 验证API密钥函数
async def validate_api_key(api_key: Optional[str] = Header(None, alias=API_KEY_NAME)):
    """验证API密钥并返回用户角色"""
    if not api_key:
        raise HTTPException(status_code=401, detail="缺少API密钥")
    
    # 去掉Bearer前缀
    if api_key.startswith("Bearer "):
        api_key = api_key[7:]
    
    # 从配置文件读取有效密钥
    valid_keys = {v: k for k, v in config.API_KEYS_CONFIG.items()}
    
    if api_key not in valid_keys:
        raise HTTPException(status_code=403, detail="无效的API密钥")
    
    return valid_keys[api_key]

# 定义lifespan上下文管理器
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI应用的生命周期管理器
    
    处理应用启动和关闭时的操作
    """
    # 启动时的操作
    logger.info("服务正在启动...")
    
    # 设置进程标题，便于在任务管理器中识别
    try:
        import setproctitle
        setproctitle.setproctitle("RAG API Server")
    except ImportError:
        logger.warning("未安装setproctitle模块，无法设置进程标题")
    
    # 记录启动信息
    logger.info(f"服务启动于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Python版本: {sys.version}")
    logger.info(f"进程ID: {os.getpid()}")
    
    # 初始化资源
    yield
    
    # 关闭时的操作
    logger.info("服务正在关闭...")
    
    # 关闭资源
    # 这里可以添加关闭数据库连接、释放资源等操作
    
    # 取消所有正在运行的任务
    try:
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        if tasks:
            logger.info(f"正在取消 {len(tasks)} 个正在运行的任务...")
            for task in tasks:
                task.cancel()
            
            try:
                # 等待所有任务完成取消，忽略取消错误
                await asyncio.gather(*tasks, return_exceptions=True)
                logger.info("所有任务已取消")
            except asyncio.CancelledError:
                # 忽略取消错误，这是预期的行为
                logger.info("任务取消过程中被中断，这是正常的")
    except Exception as e:
        logger.error(f"取消任务时出错: {e}")
    
    logger.info(f"服务关闭于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 创建FastAPI应用
app = FastAPI(
    title="聊天API服务",
    lifespan=lifespan  # 添加lifespan上下文管理器
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有源，生产环境中应限制为特定域名
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有HTTP方法
    # allow_headers=["*"],  # 允许所有请求头
    allow_headers=["*", "Authorization"],
)

# 在文件末尾导入和挂载RAG服务
from tools.rag_tools.rag_server import app as rag_app
from tools.rag_tools.rag_server import init_app as init_rag_app

# 初始化并挂载RAG服务
rag_app = init_rag_app()
# 将RAG服务挂载到/rag路径下
app.mount("/rag", rag_app)

# 确保Uvicorn不缓冲响应 - 直接添加中间件，不检查middleware_stack
@app.middleware("http")
async def no_buffering_middleware(request: Request, call_next):
    response = await call_next(request)
    if isinstance(response, StreamingResponse):
        response.headers["Cache-Control"] = "no-cache, no-transform"
        response.headers["X-Accel-Buffering"] = "no"  # 禁用Nginx缓冲
        response.headers["Transfer-Encoding"] = "chunked"
    return response

# 日志中间件 - 简化版，减少日志量
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # 记录请求详情
    request_id = f"{time.time()}-{id(request)}"
    
    # 记录请求头 - 只记录关键头部
    headers_dict = {k.decode('utf-8'): v.decode('utf-8') for k, v in request.headers.raw 
                   if k.decode('utf-8').lower() in ['content-type', 'accept']}
    
    path = request.url.path
    # 仅记录API调用，不记录静态资源等请求
    if '/api/' in path or '/chat/' in path:
        logger.info(f"请求: {request.method} {path} - 头部: {headers_dict}")
    
    # 处理请求
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    
    # 仅记录API调用的响应状态
    if '/api/' in path or '/chat/' in path:
        logger.info(f"响应: {path} - 状态码: {response.status_code} - 用时: {process_time:.2f}ms")
    
    return response

# 添加一个简单的端点来测试RAG服务是否正常工作
@app.get("/rag-status")
async def rag_status():
    """检查RAG服务状态"""
    try:
        # 尝试访问RAG服务的健康检查端点
        from httpx import AsyncClient
        # 使用127.0.0.1而不是配置中的host，因为0.0.0.0不能用于客户端连接
        host = "127.0.0.1"  # 始终使用127.0.0.1作为客户端连接地址
        port = int(os.getenv("API_PORT", config.SERVER_CONFIG.get("port", 8000)))
        
        async with AsyncClient() as client:
            response = await client.get(f"http://{host}:{port}/rag/health")
            if response.status_code == 200:
                return {"status": "RAG服务正常运行", "details": response.json()}
            else:
                return {"status": "RAG服务响应异常", "details": response.text}
    except Exception as e:
        logger.error(f"检查RAG服务状态时出错: {str(e)}")
        return {"status": "RAG服务检查失败", "error": str(e)}

# 添加一个端点来检查RAG Agent的状态
@app.get("/rag-agent-status")
async def rag_agent_status():
    """检查RAG Agent状态"""
    try:
        # 获取RAG Agent实例
        rag_agent = await get_rag_agent()
        
        # 获取健康状态
        status = await rag_agent.get_health_status()
        
        # 添加额外信息
        status["agent_type"] = rag_agent.__class__.__name__
        status["model"] = rag_agent.llm_model
        status["top_k"] = rag_agent.top_k
        status["use_async"] = rag_agent.use_async
        
        return {
            "status": "RAG Agent正常运行" if status["is_healthy"] else "RAG Agent连接异常",
            "details": status
        }
    except Exception as e:
        logger.error(f"检查RAG Agent状态时出错: {str(e)}")
        return {"status": "RAG Agent检查失败", "error": str(e)}

# 聊天端点
@app.post("/chat/conversation")
async def create_conversation(
    request: ConversationRequest,
    user_role: str = Depends(validate_api_key)
):
    """
    处理会话聊天请求
    
    Args:
        request: 包含问题列表和流式选项的请求
        user_role: 从API密钥验证中获取的用户角色
        
    Returns:
        流式响应或常规JSON响应
    """
    # 生成或使用会话ID
    session_id = request.session_id or f"session_{uuid.uuid4().hex[:8]}_{int(time.time())}"
    
    # 从配置文件获取LLM配置
    api_key = config.LLM_CONFIG.get("api_key")
    base_url = config.LLM_CONFIG.get("base_url")
    model = config.LLM_CONFIG.get("model")
    system_prompt = config.LLM_CONFIG.get("system_prompt", "你是著名的作词家。")
    
    # 检查API密钥是否有效
    if not api_key:
        logger.error("未配置API密钥")
        raise HTTPException(status_code=500, detail="服务配置错误")
    
    try:
        # 创建聊天会话
        session = AsyncChatSession(
            system_prompt=system_prompt, 
            max_messages_length=10, 
            api_key=api_key, 
            base_url=base_url, 
            model=model
        )
        
        # 根据请求类型返回流式或常规响应
        if request.stream:
            logger.info(f"会话 {session_id}: 使用流式响应模式")
            # 使用适当的方式处理异步生成器
            response = StreamingResponse(
                generate_from_async_generator(handle_streaming(session, request.questions, session_id)),
                media_type="text/event-stream"
            )
            # 设置必要的响应头，确保内容不被缓冲
            response.headers["Cache-Control"] = "no-cache, no-transform"
            response.headers["X-Accel-Buffering"] = "no"  # 禁用Nginx缓冲
            response.headers["Connection"] = "keep-alive"
            response.headers["Transfer-Encoding"] = "chunked"
            response.headers["Content-Type"] = "text/event-stream"
            # 返回会话ID
            response.headers["X-Session-ID"] = session_id
            return response
        else:
            result = await handle_regular(session, request.questions, session_id)
            # 添加会话ID到结果中
            return {
                "session_id": session_id,
                "results": result
            }
    except Exception as e:
        logger.error(f"处理会话请求时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"处理请求失败: {str(e)}")

async def handle_streaming(session: AsyncChatSession, questions: List[str], session_id: str) -> AsyncGenerator[str, None]:
    """
    处理流式响应
    
    Args:
        session: 聊天会话
        questions: 用户问题列表
        session_id: 会话ID
        
    Yields:
        流式响应数据
    """
    try:
        for i, question in enumerate(questions):
            # 记录问题到会话日志
            SessionLogger.log_question(session_id, question)
            
            await session.add_message("user", question)
            yield f"data: 问题: {question}\n\n"
            
            # 记录完整回答
            full_response = []
            
            async for chunk in session.stream_response():
                full_response.append(chunk)
                yield f"data: {chunk}\n\n"
                # 添加小延迟，确保数据被正确发送
                await asyncio.sleep(0.01)
            
            complete_answer = ''.join(full_response)
            # 记录完整回答到会话日志
            SessionLogger.log_answer_chunk(session_id, complete_answer, is_final=True)
            
            yield "data: [END_OF_QUESTION]\n\n"
            
    except Exception as e:
        error_msg = f"流式响应生成错误: {str(e)}"
        logger.error(error_msg)
        yield f"data: 错误: {str(e)}\n\n"

async def handle_regular(session: AsyncChatSession, questions: List[str], session_id: str) -> List[Dict]:
    """
    处理常规响应
    
    Args:
        session: 聊天会话
        questions: 用户问题列表
        session_id: 会话ID
        
    Returns:
        包含问题和回答的响应列表
    """
    results = []
    try:
        for i, question in enumerate(questions):
            # 记录问题到会话日志
            SessionLogger.log_question(session_id, question)
            
            await session.add_message("user", question)
            response = await session.get_response()
            
            # 记录回答到会话日志
            SessionLogger.log_answer_chunk(session_id, response, is_final=True)
            
            results.append({
                "question": question,
                "response": response,
                "messages": session.messages.copy()
            })
        
        return results
    except Exception as e:
        error_msg = f"常规响应生成错误: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=f"生成响应失败: {str(e)}")

# 添加此辅助函数，用于将异步生成器转换为适合StreamingResponse的格式
async def generate_from_async_generator(generator):
    """
    将异步生成器转换为可用于StreamingResponse的同步生成器
    
    Args:
        generator: 异步生成器
        
    Yields:
        从异步生成器中产生的数据
    """
    try:
        async for item in generator:
            # 确保每个数据项都被立即发送
            yield item
            # 强制刷新缓冲区并使用更长的延迟
            await asyncio.sleep(0.1)
            # 发送一个额外的空行来强制传输
            if item.startswith("data: 问题:") or "END_OF_QUESTION" in item:
                yield "\n"
    except Exception as e:
        logger.error(f"异步生成器处理错误: {str(e)}")
        yield f"data: 内部错误: {str(e)}\n\n"

# 定义Dify API的请求模型
class DifyFileInfo(BaseModel):
    type: str  # document, image, audio, video, custom
    transfer_method: str  # remote_url, local_file
    url: Optional[str] = None
    upload_file_id: Optional[str] = None

class DifyChatRequest(BaseModel):
    query: str
    inputs: Dict[str, Any] = Field(default_factory=dict)
    response_mode: str = "streaming"  # streaming 或 blocking
    user: Optional[str] = None
    conversation_id: Optional[str] = None
    files: Optional[List[DifyFileInfo]] = None
    auto_generate_name: Optional[bool] = True

# 定义Dify API的响应模型
class DifyUsage(BaseModel):
    prompt_tokens: int
    prompt_unit_price: str
    prompt_price_unit: str
    prompt_price: str
    completion_tokens: int
    completion_unit_price: str
    completion_price_unit: str
    completion_price: str
    total_tokens: int
    total_price: str
    currency: str
    latency: float

class DifyRetrieverResource(BaseModel):
    position: int
    dataset_id: str
    dataset_name: str
    document_id: str
    document_name: str
    segment_id: str
    score: float
    content: str

class DifyMetadata(BaseModel):
    usage: DifyUsage
    retriever_resources: List[DifyRetrieverResource] = []

class DifyChatResponse(BaseModel):
    message_id: str
    conversation_id: str
    mode: str = "chat"
    answer: str
    metadata: DifyMetadata
    created_at: int

# 创建API密钥验证器，用于Dify API
DIFY_API_KEY_NAME = "Authorization"
dify_api_key_header = APIKeyHeader(name=DIFY_API_KEY_NAME, auto_error=False)

async def validate_dify_api_key(api_key: Optional[str] = Header(None, alias=DIFY_API_KEY_NAME)):
    """验证Dify API密钥并返回用户角色"""
    if not api_key:
        raise HTTPException(status_code=401, detail="缺少API密钥")
    
    # 去掉Bearer前缀
    if api_key.startswith("Bearer "):
        api_key = api_key[7:]
    
    # 从配置文件读取有效密钥
    valid_keys = {v: k for k, v in config.API_KEYS_CONFIG.items()}
    
    if api_key not in valid_keys:
        raise HTTPException(status_code=403, detail="无效的API密钥")
    
    return valid_keys[api_key]

# 创建Dify API路由
# 创建一个子应用用于Dify API
dify_app = FastAPI(title="Dify API兼容服务")

# 添加CORS中间件
dify_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 创建RAG Agent实例
rag_agent_instance = None

async def get_rag_agent():
    """获取或创建RAG Agent实例"""
    global rag_agent_instance
    if rag_agent_instance is None:
        logger.info("无全局Agent,初始化RAG Agent")
        # 获取当前服务器的主机和端口
        # 注意：即使配置中使用0.0.0.0，客户端连接时也应该使用127.0.0.1
        host = "127.0.0.1"  # 始终使用127.0.0.1作为客户端连接地址，不使用配置中的地址
        port = int(os.getenv("API_PORT", config.SERVER_CONFIG.get("port", 8000)))
        # 使用挂载的RAG服务路径
        server_url = f"http://{host}:{port}/rag"
        logger.info(f"初始化RAG Agent，连接到RAG服务: {server_url}")
        
        # 在初始化RAG Agent之前，先检查RAG服务是否可用
        from httpx import AsyncClient
        import asyncio
        
        # 添加延迟初始化，等待服务器完全启动
        # await asyncio.sleep(2)  # 等待2秒，确保服务器已经启动
        
        # 尝试多次连接，直到成功或达到最大重试次数
        max_retries = 5  # 增加重试次数
        retry_interval = 1.0  # 增加初始重试间隔
        timeout = 3.0  # 增加超时时间
        
        for i in range(max_retries):
            try:
                # 尝试连接RAG服务的健康检查端点
                async with AsyncClient() as client:
                    logger.info(f"尝试连接RAG服务 ({i+1}/{max_retries}): {server_url}/health")
                    response = await asyncio.wait_for(
                        client.get(f"{server_url}/health"),
                        timeout=timeout
                    )
                    if response.status_code == 200:
                        logger.info(f"RAG服务连接成功: {server_url}")
                        break
                    else:
                        logger.warning(f"RAG服务响应异常: {response.status_code}, {response.text}")
            except asyncio.TimeoutError:
                logger.warning(f"RAG服务连接超时 ({timeout}秒)")
            except Exception as e:
                logger.warning(f"RAG服务连接失败，将在 {retry_interval} 秒后重试 ({i+1}/{max_retries}): {e}")
            
            if i < max_retries - 1:
                logger.info(f"等待 {retry_interval} 秒后重试连接RAG服务...")
                await asyncio.sleep(retry_interval)
                retry_interval *= 1.5  # 指数退避
        
        # 初始化RAG Agent - 即使健康检查失败也继续初始化，但设置check_health=False避免再次检查
        logger.info(f"创建RAG Agent实例，服务URL: {server_url}")
        rag_agent_instance = RAGAgent(
            server_url=server_url,
            check_health=False,  # 禁用初始健康检查，避免再次检查导致延迟
            use_async=True      # 使用异步工具函数
        )
    else:
        logger.info("已存在全局Agent,直接返回")
    return rag_agent_instance

@dify_app.post("/chat-messages")
async def dify_chat_messages(
    request: DifyChatRequest,
    user_role: str = Depends(validate_dify_api_key)
):
    """兼容Dify API的聊天消息接口"""
    logger.info(f"收到Dify API请求: {request.query}")
    
    # 生成或使用会话ID
    conversation_id = request.conversation_id or f"conv_{uuid.uuid4().hex[:8]}_{int(time.time())}"
    message_id = f"msg_{uuid.uuid4().hex}"
    
    try:
        # 获取RAG Agent
        try:
            rag_agent = await get_rag_agent()
            
            # 检查RAG Agent是否健康
            health_status = await rag_agent.get_health_status()
            logger.debug(f"RAG健康状态检查结果: {health_status}")
            
            if not health_status["is_healthy"]:
                error_details = f"服务URL: {health_status['server_url']}, 集合数量: {health_status['collection_count']}"
                logger.warning(f"RAG服务不健康: {error_details}")
                
                if request.response_mode == "streaming":
                    # 创建流式错误响应
                    async def error_generator():
                        task_id = f"task_{uuid.uuid4().hex[:8]}"
                        yield f"data: {json.dumps({'event': 'error', 'task_id': task_id, 'message_id': message_id, 'status': 503, 'code': 'service_unavailable', 'message': f'RAG服务当前不可用，请稍后重试。{error_details}'})}\n\n"
                    
                    response = StreamingResponse(
                        error_generator(),
                        media_type="text/event-stream",
                        status_code=503
                    )
                    
                    # 设置必要的响应头
                    response.headers["Cache-Control"] = "no-cache, no-transform"
                    response.headers["X-Accel-Buffering"] = "no"
                    response.headers["Connection"] = "keep-alive"
                    response.headers["Transfer-Encoding"] = "chunked"
                    response.headers["Content-Type"] = "text/event-stream"
                    
                    return response
                else:
                    # 返回常规错误响应
                    raise HTTPException(
                        status_code=503,
                        detail=f"RAG服务当前不可用，请稍后重试。{error_details}"
                    )
        except Exception as e:
            logger.error(f"获取RAG Agent时出错: {str(e)}")
            if request.response_mode == "streaming":
                # 创建流式错误响应
                async def error_generator():
                    task_id = f"task_{uuid.uuid4().hex[:8]}"
                    yield f"data: {json.dumps({'event': 'error', 'task_id': task_id, 'message_id': message_id, 'status': 500, 'code': 'internal_error', 'message': f'RAG服务初始化失败: {str(e)}'})}\n\n"
                
                response = StreamingResponse(
                    error_generator(),
                    media_type="text/event-stream",
                    status_code=500
                )
                
                # 设置必要的响应头
                response.headers["Cache-Control"] = "no-cache, no-transform"
                response.headers["X-Accel-Buffering"] = "no"
                response.headers["Connection"] = "keep-alive"
                response.headers["Transfer-Encoding"] = "chunked"
                response.headers["Content-Type"] = "text/event-stream"
                
                return response
            else:
                # 返回常规错误响应
                raise HTTPException(
                    status_code=500,
                    detail=f"RAG服务初始化失败: {str(e)}"
                )
        
        # 处理文件（如果有）
        files_info = ""
        if request.files and len(request.files) > 0:
            files_info = "附带文件: "
            for file in request.files:
                if file.transfer_method == "remote_url" and file.url:
                    files_info += f"[{file.type}:{file.url}] "
                elif file.transfer_method == "local_file" and file.upload_file_id:
                    files_info += f"[{file.type}:{file.upload_file_id}] "
        
        # 构建查询
        query = request.query
        if files_info:
            query = f"{query}\n{files_info}"
        
        # 根据请求类型返回流式或常规响应
        if request.response_mode == "streaming":
            logger.info(f"Dify API会话 {conversation_id}: 使用流式响应模式")
            
            # 检查是否配置为断开连接时继续执行查询
            continue_on_disconnect = config.SERVER_CONFIG.get("continue_on_disconnect", False)
            
            # 如果配置为断开连接时继续执行查询，则创建后台任务
            # if continue_on_disconnect:
            #     # 创建一个后台任务字典，用于存储正在进行的查询任务
            #     if not hasattr(dify_streaming_generator, "background_tasks"):
            #         dify_streaming_generator.background_tasks = {}
                
            #     # 创建一个事件，用于通知后台任务客户端已断开连接
            #     client_disconnected = asyncio.Event()
                
            #     # 创建一个Future对象，用于存储查询结果
            #     result_future = asyncio.Future()
                
            #     async def background_query_task():
            #         """后台查询任务，即使客户端断开连接也会继续执行"""
            #         try:
            #             # 检查RAG Agent是否健康
            #             try:
            #                 health_status = await rag_agent.get_health_status()
            #                 logger.info(f"后台任务 - RAG健康状态检查结果: {health_status}")
                            
            #                 if not health_status["is_healthy"]:
            #                     error_details = f"服务URL: {health_status['server_url']}, 集合数量: {health_status['collection_count']}"
            #                     logger.warning(f"后台任务 - RAG服务不健康: {error_details}")
            #                     result_future.set_exception(Exception(f"RAG服务当前不可用: {error_details}"))
            #                     return
            #             except Exception as e:
            #                 logger.error(f"后台任务 - 检查RAG服务健康状态失败: {str(e)}")
            #                 # 继续尝试执行查询，因为健康检查可能只是暂时失败
                        
            #             # 执行RAG查询
            #             try:
            #                 # 使用RAG Agent处理查询，添加超时处理
            #                 timeout_seconds = 180  # 设置30秒超时
            #                 try:
            #                     # 使用asyncio.wait_for添加超时
            #                     logger.info(f"后台任务 - 开始执行RAG查询: {query}")
            #                     answer = await asyncio.wait_for(
            #                         rag_agent.run(query),
            #                         timeout=timeout_seconds
            #                     )
                                
            #                     logger.info(f"后台任务 - RAG查询完成，回答长度: {len(answer)}")
                                
            #                     # 设置查询结果
            #                     if not result_future.done():
            #                         result_future.set_result(answer)
                                
            #                 except asyncio.TimeoutError:
            #                     error_msg = f"RAG查询超时 ({timeout_seconds}秒)，请稍后重试或简化您的问题"
            #                     logger.error(f"后台任务 - {error_msg}")
            #                     if not result_future.done():
            #                         result_future.set_exception(asyncio.TimeoutError(error_msg))
            #                 except Exception as e:
            #                     # 如果是连接错误，尝试重新检查健康状态
            #                     logger.error(f"后台任务 - RAG查询执行错误: {str(e)}")
                                
            #                     # 尝试重新检查健康状态
            #                     try:
            #                         await rag_agent.check_health()
            #                     except Exception:
            #                         pass  # 忽略健康检查错误
                                
            #                     if not result_future.done():
            #                         result_future.set_exception(Exception(f"RAG查询失败: {str(e)}"))
            #             except Exception as e:
            #                 logger.error(f"后台任务 - RAG查询执行错误: {str(e)}")
            #                 if not result_future.done():
            #                     result_future.set_exception(Exception(f"RAG查询失败: {str(e)}"))
            #         except Exception as e:
            #             logger.error(f"后台任务 - 未处理的错误: {str(e)}")
            #             if not result_future.done():
            #                 result_future.set_exception(e)
            #         finally:
            #             # 从后台任务字典中移除此任务
            #             if task_id in dify_streaming_generator.background_tasks:
            #                 del dify_streaming_generator.background_tasks[task_id]
                        
            #             # 如果客户端已断开连接，记录查询结果
            #             if client_disconnected.is_set():
            #                 try:
            #                     if result_future.done():
            #                         if result_future.exception() is None:
            #                             answer = result_future.result()
            #                             logger.info(f"客户端已断开连接，但查询已完成。查询: '{query}', 回答长度: {len(answer)}")
            #                             # 这里可以将结果保存到数据库或缓存中，以便客户端重新连接时获取
            #                         else:
            #                             logger.error(f"客户端已断开连接，查询失败: {result_future.exception()}")
            #                 except Exception as e:
            #                     logger.error(f"处理断开连接后的查询结果时出错: {str(e)}")
                
            #     # 启动后台任务
            #     background_task = asyncio.create_task(background_query_task())
            #     dify_streaming_generator.background_tasks[task_id] = background_task
            
            # 创建流式响应
            response = StreamingResponse(
                dify_streaming_generator(rag_agent, query, message_id, conversation_id),
                media_type="text/event-stream"
            )
            
            # 设置必要的响应头，确保内容不被缓冲
            response.headers["Cache-Control"] = "no-cache, no-transform"
            response.headers["X-Accel-Buffering"] = "no"  # 禁用Nginx缓冲
            response.headers["Connection"] = "keep-alive"
            response.headers["Transfer-Encoding"] = "chunked"
            response.headers["Content-Type"] = "text/event-stream"
            
            return response
        else:
            logger.info(f"Dify API会话 {conversation_id}: 使用阻塞响应模式")
            
            # 执行RAG查询
            try:
                start_time = time.time()
                
                # 添加超时处理
                timeout_seconds = 1800  # 设置30秒超时，查询时间长很正常，与模型有关
                try:
                    # 使用asyncio.wait_for添加超时
                    answer = await asyncio.wait_for(
                        rag_agent.run(query),
                        timeout=timeout_seconds
                    )
                except asyncio.TimeoutError:
                    logger.error(f"RAG查询超时 ({timeout_seconds}秒)")
                    raise HTTPException(
                        status_code=504,  # Gateway Timeout
                        detail=f"RAG查询处理超时，请稍后重试或简化您的问题"
                    )
                
                latency = time.time() - start_time
                
                # 构建响应 - 使用新的格式
                response = {
                    "event": "message",
                    "message_id": message_id,
                    "conversation_id": conversation_id,
                    "mode": "chat",
                    "answer": answer,
                    "metadata": {
                        "usage": {
                            "prompt_tokens": 100,  # 示例值，实际应从模型响应中获取
                            "prompt_unit_price": "0.001",
                            "prompt_price_unit": "0.001",
                            "prompt_price": "0.0001000",
                            "completion_tokens": len(answer) // 4,  # 粗略估计
                            "completion_unit_price": "0.002",
                            "completion_price_unit": "0.001",
                            "completion_price": f"{(len(answer) // 4) * 0.002 / 1000:.7f}",
                            "total_tokens": 100 + (len(answer) // 4),
                            "total_price": f"{(100 * 0.001 + (len(answer) // 4) * 0.002) / 1000:.7f}",
                            "currency": "USD",
                            "latency": latency
                        },
                        "retriever_resources": []  # 可以从RAG结果中提取
                    },
                    "created_at": int(time.time())
                }
                
                return response
            except Exception as e:
                logger.error(f"处理Dify API请求时出错: {str(e)}")
                raise HTTPException(status_code=500, detail=f"处理请求失败: {str(e)}")
    except Exception as e:
        logger.error(f"处理Dify API请求时出错: {str(e)}")
        # 返回一个友好的错误消息
        if request.response_mode == "streaming":
            # 创建流式错误响应
            async def error_generator():
                task_id = f"task_{uuid.uuid4().hex[:8]}"
                yield f"data: {json.dumps({'event': 'error', 'task_id': task_id, 'message_id': message_id, 'status': 500, 'code': 'internal_error', 'message': f'处理请求失败: {str(e)}'})}\n\n"
            
            response = StreamingResponse(
                error_generator(),
                media_type="text/event-stream",
                status_code=500
            )
            
            # 设置必要的响应头
            response.headers["Cache-Control"] = "no-cache, no-transform"
            response.headers["X-Accel-Buffering"] = "no"
            response.headers["Connection"] = "keep-alive"
            response.headers["Transfer-Encoding"] = "chunked"
            response.headers["Content-Type"] = "text/event-stream"
            
            return response
        else:
            # 返回常规错误响应
            raise HTTPException(status_code=500, detail=f"处理请求失败: {str(e)}")

async def dify_streaming_generator(rag_agent, query, message_id, conversation_id):
    """生成Dify API兼容的流式响应"""
    task_id = f"task_{uuid.uuid4().hex[:8]}"
    start_time = time.time()
    full_answer = ""
    
    # 检查是否配置为断开连接时继续执行查询
    continue_on_disconnect = config.SERVER_CONFIG.get("continue_on_disconnect", False)
    
    # 如果配置为断开连接时继续执行查询，则创建后台任务
    # if continue_on_disconnect:
    #     # 创建一个后台任务字典，用于存储正在进行的查询任务
    #     if not hasattr(dify_streaming_generator, "background_tasks"):
    #         dify_streaming_generator.background_tasks = {}
        
    #     # 创建一个事件，用于通知后台任务客户端已断开连接
    #     client_disconnected = asyncio.Event()
        
    #     # 创建一个Future对象，用于存储查询结果
    #     result_future = asyncio.Future()
        
    #     async def background_query_task():
    #         """后台查询任务，即使客户端断开连接也会继续执行"""
    #         try:
    #             # 检查RAG Agent是否健康
    #             try:
    #                 health_status = await rag_agent.get_health_status()
    #                 logger.info(f"后台任务 - RAG健康状态检查结果: {health_status}")
                    
    #                 if not health_status["is_healthy"]:
    #                     error_details = f"服务URL: {health_status['server_url']}, 集合数量: {health_status['collection_count']}"
    #                     logger.warning(f"后台任务 - RAG服务不健康: {error_details}")
    #                     result_future.set_exception(Exception(f"RAG服务当前不可用: {error_details}"))
    #                     return
    #             except Exception as e:
    #                 logger.error(f"后台任务 - 检查RAG服务健康状态失败: {str(e)}")
    #                 # 继续尝试执行查询，因为健康检查可能只是暂时失败
                
    #             # 执行RAG查询
    #             try:
    #                 # 使用RAG Agent处理查询，添加超时处理
    #                 timeout_seconds = 180  # 设置30秒超时
    #                 try:
    #                     # 使用asyncio.wait_for添加超时
    #                     logger.info(f"后台任务 - 开始执行RAG查询: {query}")
    #                     answer = await asyncio.wait_for(
    #                         rag_agent.run(query),
    #                         timeout=timeout_seconds
    #                     )
                        
    #                     logger.info(f"后台任务 - RAG查询完成，回答长度: {len(answer)}")
                        
    #                     # 设置查询结果
    #                     if not result_future.done():
    #                         result_future.set_result(answer)
                        
    #                 except asyncio.TimeoutError:
    #                     error_msg = f"RAG查询超时 ({timeout_seconds}秒)，请稍后重试或简化您的问题"
    #                     logger.error(f"后台任务 - {error_msg}")
    #                     if not result_future.done():
    #                         result_future.set_exception(asyncio.TimeoutError(error_msg))
    #                 except Exception as e:
    #                     # 如果是连接错误，尝试重新检查健康状态
    #                     logger.error(f"后台任务 - RAG查询执行错误: {str(e)}")
                        
    #                     # 尝试重新检查健康状态
    #                     try:
    #                         await rag_agent.check_health()
    #                     except Exception:
    #                         pass  # 忽略健康检查错误
                        
    #                     if not result_future.done():
    #                         result_future.set_exception(Exception(f"RAG查询失败: {str(e)}"))
    #             except Exception as e:
    #                 logger.error(f"后台任务 - RAG查询执行错误: {str(e)}")
    #                 if not result_future.done():
    #                     result_future.set_exception(Exception(f"RAG查询失败: {str(e)}"))
    #         except Exception as e:
    #             logger.error(f"后台任务 - 未处理的错误: {str(e)}")
    #             if not result_future.done():
    #                 result_future.set_exception(e)
    #         finally:
    #             # 从后台任务字典中移除此任务
    #             if task_id in dify_streaming_generator.background_tasks:
    #                 del dify_streaming_generator.background_tasks[task_id]
                
    #             # 如果客户端已断开连接，记录查询结果
    #             if client_disconnected.is_set():
    #                 try:
    #                     if result_future.done():
    #                         if result_future.exception() is None:
    #                             answer = result_future.result()
    #                             logger.info(f"客户端已断开连接，但查询已完成。查询: '{query}', 回答长度: {len(answer)}")
    #                             # 这里可以将结果保存到数据库或缓存中，以便客户端重新连接时获取
    #                         else:
    #                             logger.error(f"客户端已断开连接，查询失败: {result_future.exception()}")
    #                     else:
    #                         logger.warning(f"客户端已断开连接，查询被取消")
    #                 except Exception as e:
    #                     logger.error(f"处理断开连接后的查询结果时出错: {str(e)}")
        
    #     # 启动后台任务
    #     background_task = asyncio.create_task(background_query_task())
    #     dify_streaming_generator.background_tasks[task_id] = background_task
    
    try:
        # 发送开始事件
        yield f"data: {json.dumps({'event': 'start', 'task_id': task_id, 'message_id': message_id, 'conversation_id': conversation_id})}\n\n"
        
        if continue_on_disconnect:
            # 等待查询结果
            try:
                return
            except Exception as e:
                logger.error(f"等待查询结果时出错: {str(e)}")
                yield f"data: {json.dumps({'event': 'error', 'task_id': task_id, 'message_id': message_id, 'status': 500, 'code': 'internal_error', 'message': f'处理查询结果时出错: {str(e)}'})}\n\n"
                return
        else:
            # 直接执行查询，不使用后台任务
            # 检查RAG Agent是否健康
            try:
                health_status = await rag_agent.get_health_status()
                logger.info(f"RAG健康状态检查结果: {health_status}")
                
                if not health_status["is_healthy"]:
                    error_details = f"服务URL: {health_status['server_url']}, 集合数量: {health_status['collection_count']}"
                    logger.warning(f"流式响应中RAG服务不健康: {error_details}")
                    yield f"data: {json.dumps({'event': 'error', 'task_id': task_id, 'message_id': message_id, 'status': 503, 'code': 'service_unavailable', 'message': f'RAG服务当前不可用，请稍后重试。{error_details}'})}\n\n"
                    return
            except Exception as e:
                logger.error(f"流式响应中检查RAG服务健康状态失败: {str(e)}")
                # 继续尝试执行查询，因为健康检查可能只是暂时失败
                yield f"data: {json.dumps({'event': 'message', 'task_id': task_id, 'message_id': message_id, 'conversation_id': conversation_id, 'answer': '正在尝试连接知识库服务...'})}\n\n"
            
            # 执行RAG查询
            try:
                # 使用RAG Agent处理查询，添加超时处理
                timeout_seconds = 180  # 设置30秒超时
                try:
                    # 使用asyncio.wait_for添加超时
                    logger.debug(f"开始执行RAG查询: {query}")
                    answer = await asyncio.wait_for(
                        rag_agent.run(query),
                        timeout=timeout_seconds
                    )
                    
                    logger.info(f"RAG查询完成，回答长度: {len(answer)}")
                    
                    # 模拟流式输出 - 每次发送一个字符
                    for char in answer:
                        full_answer += char
                        yield f"data: {json.dumps({'event': 'message', 'task_id': task_id, 'message_id': message_id, 'conversation_id': conversation_id, 'answer': char})}\n\n"
                        await asyncio.sleep(0.01)  # 添加小延迟，模拟真实的流式输出
                except asyncio.TimeoutError:
                    error_msg = f"RAG查询超时 ({timeout_seconds}秒)，请稍后重试或简化您的问题"
                    logger.error(error_msg)
                    yield f"data: {json.dumps({'event': 'error', 'task_id': task_id, 'message_id': message_id, 'status': 504, 'code': 'timeout', 'message': error_msg})}\n\n"
                    return
                except Exception as e:
                    # 如果是连接错误，尝试重新检查健康状态
                    logger.error(f"RAG查询执行错误: {str(e)}")
                    
                    # 尝试重新检查健康状态
                    try:
                        await rag_agent.check_health()
                    except Exception:
                        pass  # 忽略健康检查错误
                    
                    yield f"data: {json.dumps({'event': 'error', 'task_id': task_id, 'message_id': message_id, 'status': 500, 'code': 'rag_error', 'message': f'RAG查询失败: {str(e)}'})}\n\n"
                    return
            except Exception as e:
                logger.error(f"RAG查询执行错误: {str(e)}")
                yield f"data: {json.dumps({'event': 'error', 'task_id': task_id, 'message_id': message_id, 'status': 500, 'code': 'rag_error', 'message': f'RAG查询失败: {str(e)}'})}\n\n"
                return
        
        # 计算耗时
        elapsed_time = time.time() - start_time
        
        # 发送消息结束事件
        metadata = {
            "usage": {
                "prompt_tokens": len(query),
                "prompt_unit_price": "0.001",
                "prompt_price_unit": "0.001",
                "prompt_price": f"{len(query) * 0.001 / 1000:.7f}",
                "completion_tokens": len(full_answer),
                "completion_unit_price": "0.002",
                "completion_price_unit": "0.001",
                "completion_price": f"{len(full_answer) * 0.002 / 1000:.7f}",
                "total_tokens": len(query) + len(full_answer),
                "total_price": f"{(len(query) * 0.001 + len(full_answer) * 0.002) / 1000:.7f}",
                "currency": "USD",
                "latency": elapsed_time
            },
            "retriever_resources": []  # 可以从RAG结果中提取
        }
        
        yield f"data: {json.dumps({'event': 'message_end', 'task_id': task_id, 'message_id': message_id, 'conversation_id': conversation_id, 'metadata': metadata})}\n\n"
        
    except Exception as e:
        logger.error(f"生成Dify流式响应时出错: {str(e)}")
        yield f"data: {json.dumps({'event': 'error', 'task_id': task_id, 'message_id': message_id, 'status': 500, 'code': 'internal_error', 'message': str(e)})}\n\n"
    except asyncio.CancelledError:
        # 客户端断开连接
        disconnect_time = time.time()
        elapsed_time = disconnect_time - start_time
        logger.warning(
            f"客户端断开连接，查询{'将继续在后台执行' if continue_on_disconnect else '已取消'}:\n"
            f"  - 任务ID: {task_id}\n"
            f"  - 会话ID: {conversation_id}\n"
            f"  - 消息ID: {message_id}\n"
            f"  - 查询内容: '{query}'\n"
            f"  - 断开时间: {datetime.fromtimestamp(disconnect_time).strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"  - 查询持续时间: {elapsed_time:.2f}秒"
        )
        
        # 如果配置为断开连接时继续执行查询，则设置断开连接事件
        # if continue_on_disconnect:
        #     client_disconnected.set()
        
        # 重新抛出异常，让FastAPI处理
        raise

# 将Dify API挂载到主应用的/v1路径下
app.mount("/v1", dify_app)

async def check_rag_service_health(host="127.0.0.1", port=8000, max_retries=3, retry_interval=1.0):
    """检查RAG服务的健康状态
    
    Args:
        host: 主机地址
        port: 端口号
        max_retries: 最大重试次数
        retry_interval: 重试间隔（秒）
        
    Returns:
        (bool, str): 是否健康，详细信息
    """
    from httpx import AsyncClient
    import asyncio
    
    # 构建URL
    url = f"http://{host}:{port}/rag/health"
    logger.debug(f"检查RAG服务健康状态: {url}")
    
    # 尝试多次连接
    for i in range(max_retries):
        try:
            async with AsyncClient() as client:
                response = await asyncio.wait_for(
                    client.get(url),
                    timeout=3.0
                )
                
                if response.status_code == 200:
                    logger.info(f"RAG服务健康检查成功: {url}")
                    return True, f"RAG服务健康检查成功: {response.json()}"
                else:
                    logger.warning(f"RAG服务响应异常: {response.status_code}, {response.text}")
        except asyncio.TimeoutError:
            logger.warning(f"RAG服务连接超时 (3.0秒)")
        except Exception as e:
            logger.warning(f"RAG服务连接失败: {e}")
        
        if i < max_retries - 1:
            logger.info(f"等待 {retry_interval} 秒后重试...")
            await asyncio.sleep(retry_interval)
            retry_interval *= 1.5  # 指数退避
    
    return False, "RAG服务健康检查失败，请确保服务已启动"

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
    needs_restart: bool = False

class RebootResponse(BaseModel):
    """重启响应模型"""
    success: bool
    message: str

# 定义需要重启的配置项
RESTART_REQUIRED_CONFIGS = {
    "server": ["host", "port", "timeout_keep_alive", "timeout_graceful_shutdown", "limit_concurrency"],
    "api_keys": ["admin", "user"],
    "llm": ["api_key", "base_url"],
    "rag_server": ["host", "port"]
}

def handle_config_update(config_manager):
    """处理配置更新
    
    Args:
        config_manager: 配置管理器实例
    """
    # 重新初始化RAG Agent
    global rag_agent_instance
    rag_agent_instance = None
    logger.info("配置已更新，RAG Agent将在下次请求时重新初始化")

# 注册配置更新观察者
config.add_observer(handle_config_update)

def restart_server():
    """重启服务器"""
    logger.info("正在重启服务器...")
    # 获取当前脚本的路径和参数
    args = [sys.executable] + sys.argv
    
    # 创建新进程，并确保它能够接收信号
    # 在Windows上，创建进程时需要设置CREATE_NEW_PROCESS_GROUP标志
    # 在Unix/Linux上，需要确保新进程不会继承父进程的信号处理程序
    if os.name == 'nt':  # Windows
        # 使用subprocess.CREATE_NEW_PROCESS_GROUP和subprocess.CREATE_NEW_CONSOLE标志创建新进程组和新控制台
        process = subprocess.Popen(
            args,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NEW_CONSOLE
        )
    else:  # Unix/Linux
        # 在Unix/Linux上，使用preexec_fn参数重置信号处理
        def preexec_fn():
            # 重置所有信号处理为默认值
            for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGHUP]:
                try:
                    signal.signal(sig, signal.SIG_DFL)
                except:
                    pass
            # 创建新的进程组
            os.setpgrp()
        
        process = subprocess.Popen(
            args,
            preexec_fn=preexec_fn
        )
    
    logger.info(f"新进程已启动，PID: {process.pid}")
    
    # 使用os._exit直接退出，避免触发异常和清理过程
    os._exit(0)

async def delayed_restart():
    """延迟重启服务器，给客户端一些时间接收响应"""
    try:
        logger.info("服务将在5秒后重启...")
        await asyncio.sleep(5)  # 等待5秒
        
        # 在主线程中执行重启，避免在异步任务中直接退出
        loop = asyncio.get_running_loop()
        loop.call_soon_threadsafe(restart_server)
    except asyncio.CancelledError:
        logger.warning("重启任务被取消")
    except Exception as e:
        logger.error(f"重启过程中出错: {e}")
        # 尝试直接重启
        restart_server()

@app.post("/config/update", response_model=ConfigUpdateResponse)
async def update_config(request: ConfigUpdateRequest):
    """更新配置
    
    Args:
        request: 配置更新请求
        
    Returns:
        配置更新响应
    """
    try:
        # 读取当前配置
        with open(config.config_path, "rb") as f:
            current_config = tomli.load(f)
        
        # 更新配置
        if request.section not in current_config:
            current_config[request.section] = {}
        current_config[request.section][request.key] = request.value
        
        # 保存配置
        with open(config.config_path, "w", encoding="utf-8") as f:
            toml.dump(current_config, f)
        
        # 重新加载配置
        if config.reload_config():
            # 检查是否需要重启
            needs_restart = False
            restart_message = ""
            
            if request.section in RESTART_REQUIRED_CONFIGS:
                if request.key in RESTART_REQUIRED_CONFIGS[request.section]:
                    needs_restart = True
                    restart_message = f"注意：更新了 {request.section}.{request.key}，服务将在5秒后自动重启。"
                    # 创建异步任务在返回响应后重启服务器
                    asyncio.create_task(delayed_restart())

            return ConfigUpdateResponse(
                success=True,
                message=f"配置更新成功。{restart_message}",
                config=current_config,
                needs_restart=needs_restart
            )
        else:
            return ConfigUpdateResponse(
                success=False,
                message="配置未发生变化",
                config=current_config,
                needs_restart=False
            )
    except Exception as e:
        logger.error(f"更新配置失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/config", response_model=Dict[str, Any])
async def get_config():
    """获取当前配置
    
    Returns:
        当前配置
    """
    try:
        with open(config.config_path, "rb") as f:
            return tomli.load(f)
    except Exception as e:
        logger.error(f"获取配置失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/reboot", response_model=RebootResponse)
async def reboot():
    """重启服务器"""
    logger.info("正在重启服务器...")
    asyncio.create_task(delayed_restart())
    return RebootResponse(
        success=True,
        message="服务器将在5秒后重启"
    )


if __name__ == "__main__":
    import uvicorn
    import asyncio
    
    # 从配置文件获取主机和端口
    host = os.getenv("API_HOST", config.SERVER_CONFIG.get("host", "0.0.0.0"))
    port = int(os.getenv("API_PORT", config.SERVER_CONFIG.get("port", 8000)))
    
    # 检查端口是否被占用，如果被占用则自动切换到其他端口
    def is_port_in_use(port, host='0.0.0.0'):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return False
            except socket.error:
                return True
    
    # 查找可用端口
    original_port = port
    max_port_attempts = 10
    for attempt in range(max_port_attempts):
        if not is_port_in_use(port):
            break
        logger.warning(f"端口 {port} 已被占用，尝试使用端口 {port + 1}")
        port += 1
    
    if port != original_port:
        logger.info(f"自动切换到可用端口: {port}")
    
    # 在Windows上设置控制台处理
    if os.name == 'nt':
        try:
            import ctypes
            import msvcrt
            
            # 获取控制台句柄
            kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
            
            # 设置控制台模式，启用ENABLE_PROCESSED_INPUT以处理Ctrl+C
            STD_INPUT_HANDLE = -10
            ENABLE_PROCESSED_INPUT = 0x0001
            
            handle = kernel32.GetStdHandle(STD_INPUT_HANDLE)
            mode = ctypes.c_ulong()
            kernel32.GetConsoleMode(handle, ctypes.byref(mode))
            kernel32.SetConsoleMode(handle, mode.value | ENABLE_PROCESSED_INPUT)
            
            logger.info("已设置Windows控制台模式以处理Ctrl+C信号")
        except Exception as e:
            logger.warning(f"设置Windows控制台模式失败: {e}")
    
    # 添加信号处理程序
    def signal_handler(sig, frame):
        logger.info(f"接收到终止信号 {sig}，正在优雅关闭服务...")
        # 使用sys.exit而不是os._exit，允许正常的清理过程
        sys.exit(0)
    
    # 注册SIGINT和SIGTERM信号处理程序
    # 在Windows环境中，只有有限的信号可用，主要是SIGINT和SIGBREAK
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    if hasattr(signal, 'SIGBREAK'):  # Windows特有的信号
        signal.signal(signal.SIGBREAK, signal_handler)  # Ctrl+Break
    if hasattr(signal, 'SIGTERM'):  # 可能在某些环境中不可用
        signal.signal(signal.SIGTERM, signal_handler)  # kill命令
    
    # 检查RAG服务是否已经启动
    logger.info("正在检查RAG服务是否已经启动...")
    
    # 创建一个事件循环来运行异步函数
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # 检查RAG服务健康状态
    is_healthy, message = loop.run_until_complete(check_rag_service_health(host="127.0.0.1", port=port))
    
    if is_healthy:
        logger.info(f"RAG服务已经启动: {message}")
    else:
        logger.warning(f"RAG服务尚未启动: {message}")
        logger.info("API服务器将继续启动，RAG服务将在API服务器启动后自动挂载")
    
    logger.info(f"启动API服务器于 {host}:{port}")
    logger.info(f"RAG服务已挂载到 /rag 路径")
    logger.info(f"Dify API兼容服务已挂载到 /v1 路径")
    logger.info(f"可通过 http://{host}:{port}/v1/chat-messages 访问Dify API")
    
    # 使用配置文件中的Uvicorn配置
    uvicorn.run(
        app, 
        host=host, 
        port=port,
        log_level="info",
        timeout_keep_alive=config.SERVER_CONFIG.get("timeout_keep_alive", 120),
        limit_concurrency=config.SERVER_CONFIG.get("limit_concurrency", 20),
        # 添加以下配置以确保正确处理信号
        reload=False,  # 禁用自动重载，因为我们有自己的重启逻辑
        workers=1,     # 使用单个工作进程，简化信号处理
        loop="asyncio",  # 使用asyncio事件循环
        access_log=False,  # 禁用访问日志，减少控制台输出
        use_colors=True,   # 使用彩色输出
        # 设置关闭超时时间
        timeout_graceful_shutdown=30
    )