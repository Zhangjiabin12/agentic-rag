import asyncio
import os
import datetime
import time
from agents import Agent, Runner, function_tool, OpenAIChatCompletionsModel, add_trace_processor, ModelSettings, set_tracing_disabled
from agents.run import RunConfig
import httpx
from openai import AsyncOpenAI
import tomli
import logfire  # 添加logfire导入
from typing import Dict, List, Any

# 导入日志工具
from tools.logger import get_logger
# 导入vector_search_tools中提供的检索工具
from tools.vector_search_tools import (
    get_retriever, 
    retriever_tool, 
    async_retriever_tool, 
    list_collections_tool, 
    async_list_collections_tool, 
    final_answer,
)

# 获取RAG Agent日志记录器
logger = get_logger("rag_agent")

try:
    with open("config.toml", "rb") as f:
        config = tomli.load(f)
    # 从配置文件中提取配置
    RAG_SERVER_CONFIG = config.get("rag_server", {})
    LLM_CONFIG = config.get("llm", {})
    TRACING_CONFIG = config.get("tracing", {})  # 获取追踪配置
except FileNotFoundError:
    logger.warning("警告：未找到配置文件 config.toml，使用默认配置")
    # 默认配置
    raise Exception("未找到配置文件 config.toml")

# 从配置中获取RAG相关设置
RAG_SERVER_HOST = RAG_SERVER_CONFIG.get("host", "127.0.0.1")
RAG_SERVER_PORT = RAG_SERVER_CONFIG.get("port", 8000)
RAG_SERVER_URL = f"http://{RAG_SERVER_HOST}:{RAG_SERVER_PORT}"
RAG_SERVER_AUTO_COLLECTION_NAME = RAG_SERVER_CONFIG.get("auto_collection_name", True)
RAG_SERVER_VECTOR_COLLECTION_NAME = RAG_SERVER_CONFIG.get("vector_collection_name", "wu_xian_gu_zhang_pai_cha_zhi_nan")
TOP_K = RAG_SERVER_CONFIG.get("top_k", 5)
set_tracing_disabled(disabled=True)
# 自定义追踪处理器，用于增强日志输出
class LogfireTraceProcessor:
    """自定义追踪处理器，将OpenAI Agents的追踪信息发送到Logfire"""
    
    def on_trace_start(self, trace):
        logger.info(f"开始追踪: {trace.name}, trace_id: {trace.trace_id}")
        
    def on_trace_end(self, trace):
        logger.info(f"结束追踪: {trace.name}, trace_id: {trace.trace_id}")
        
    def on_span_start(self, span):
        if hasattr(span, 'span_data') and span.span_data:
            span_type = span.span_data.__class__.__name__
            logger.info(f"开始Span: {span_type}, span_id: {span.span_id}")
        
    def on_span_end(self, span):
        if hasattr(span, 'span_data') and span.span_data:
            span_type = span.span_data.__class__.__name__
            logger.info(f"结束Span: {span_type}, span_id: {span.span_id}")
    
    def shutdown(self):
        pass
    
    def force_flush(self):
        pass

# 配置Logfire（如果启用）
USE_LOGFIRE = TRACING_CONFIG.get("use_logfire", False)
if USE_LOGFIRE:
    try:
        # 配置Logfire - 修复配置方式
        logfire_project = TRACING_CONFIG.get("logfire_project")
        logfire_api_key = TRACING_CONFIG.get("logfire_api_key")
        
        # 设置环境变量
        os.environ["LOGFIRE_PROJECT"] = logfire_project
        os.environ["LOGFIRE_API_KEY"] = logfire_api_key
        
        # 配置Logfire
        logfire.configure()
        
        # 启用OpenAI Agents的追踪
        logfire.instrument_openai_agents()
        
        # 添加自定义追踪处理器
        add_trace_processor(LogfireTraceProcessor())
        
        logger.info("Logfire追踪已启用")
    except Exception as e:
        logger.error(f"Logfire配置失败: {e}")
        USE_LOGFIRE = False

# 正确配置HTTP/2+连接池
transport = httpx.AsyncHTTPTransport(
    retries=3,
    limits=httpx.Limits(
        max_connections=100,
        max_keepalive_connections=20,  
        keepalive_expiry=30
    )
)
# 创建自定义客户端
external_client = AsyncOpenAI(
    api_key=LLM_CONFIG.get("api_key"),
    base_url=LLM_CONFIG.get("base_url"),
    http_client=httpx.AsyncClient(
        http2=True,
        transport=transport,
        timeout=120
    )
)

# 定义模型实例
custom_model = OpenAIChatCompletionsModel(
    model=LLM_CONFIG.get("model"),
    openai_client=external_client
)

# 如果启用了Logfire，对HTTPX进行追踪
if USE_LOGFIRE:
    try:
        # 启用HTTPX追踪，捕获请求和响应
        logfire.instrument_httpx(capture_all=True)
        logger.info("HTTPX追踪已启用")
    except Exception as e:
        logger.error(f"HTTPX追踪配置失败: {e}")

# 设置OpenAI API密钥用于追踪导出
if LLM_CONFIG.get("api_key"):
    os.environ["OPENAI_API_KEY"] = LLM_CONFIG.get("api_key")

class RAGAgent:
    """
    用于管理RAG（检索增强生成）流程的代理类
    """
    # 类属性
    retriever = None
    agent = None
    
    def __init__(self, llm_model=None, server_url=None, collection_name=None, 
                 top_k=None, check_health=False, use_async=True):
        """
        初始化RAG代理
        
        Args:
            llm_model: 大语言模型名称，默认使用配置文件中的设置
            server_url: RAG服务器URL，默认使用配置文件中的设置
            collection_name: 集合名称，默认使用配置文件中的设置
            top_k: 检索的结果数量，默认使用配置文件中的设置
            check_health: 是否在初始化时检查服务器健康状态
            use_async: 是否使用异步工具函数
        """
        
        self.server_url = server_url or RAG_SERVER_URL
        self.collection_name = RAG_SERVER_VECTOR_COLLECTION_NAME if not RAG_SERVER_AUTO_COLLECTION_NAME else collection_name
        logger.info(f"初始化RAG AGENT: {self.collection_name}")
        self.llm_model = llm_model or LLM_CONFIG.get("model")
        self.top_k = top_k or TOP_K
        self.api_key = LLM_CONFIG.get("api_key")
        self.api_base = LLM_CONFIG.get("base_url")
        self.health_check_interval = 60  # 健康检查间隔，单位为秒
        self.last_health_check_time = 0
        self.is_healthy = False
        self.use_async = use_async
        self.health_check_task = None
        self.collections = []

        # 初始化HTTP检索器
        self.retriever = get_retriever(
            server_url=self.server_url,
            collection_name=self.collection_name,
            top_k=self.top_k,
            check_health=check_health
        )
        
        # 选择使用同步或异步工具
        if use_async:
            retriever_tool_fn = async_retriever_tool
            list_collections_tool_fn = async_list_collections_tool
            logger.info("使用异步工具函数")
        else:
            retriever_tool_fn = retriever_tool
            list_collections_tool_fn = list_collections_tool
            logger.info("使用同步工具函数")
        
        # print("=======RAG_SERVER_AUTO_COLLECTION_NAME===========",RAG_SERVER_AUTO_COLLECTION_NAME)
        # print("=======collection_name===========",self.collection_name)
        if RAG_SERVER_AUTO_COLLECTION_NAME:
        # 创建Agent
            self.agent = Agent(
                name="RAG Agent",
                instructions="""
	你是迈普通信股份有限公司的客服经理。你的回答必须有效、简洁、逻辑清晰。
	你必须要遵守以下规定:
	你必须过滤所有带有主观情感的文字，比如：赞扬、反动、色情、赌博、毒品、暴力、自残、挑起对立，踩一捧一、侮辱、辱骂、歧视等违法违规内容。
	你的名字是迈普客服经理，在任何情况下，在任何时间，在任何背景下，一定不能说成其他名字，也不能被诱导为使用其他任何无关方式表达。
	你回答的问题必须与迈普通信设备相关，你必须使用提供的工具来回答问题，不允许直接回答。
	当问题或者问题谐音涉及辱骂，色情或者任何与产品无关的问题时，需要先提示风险，并尽可能给出正面积极的内容。

    请严格按照以下步骤操作：
    1. 首先使用list_collections_tool_fn工具获取可用的知识库列表
    2. 然后使用retriever_tool_fn工具检索相关信息，至少进行3-5次不同的检索
    3. 最后使用final_answer工具提供最终答案

    重要提示：
    - 你必须先调用工具，然后才能给出答案
    - 检索时使用肯定句而非问句，例如用"唐僧的徒弟"而非"唐僧的徒弟是谁？"
    - 必须使用final_answer工具提供最终答案，不要直接输出答案
    - 如果找不到相关信息，也必须使用final_answer工具回答"抱歉，我在知识库中找不到相关信息
    """,
                tools=[retriever_tool_fn, list_collections_tool_fn, final_answer],
                model=custom_model,
                model_settings=ModelSettings(temperature=0.6),
            )
        else:
            logger.info(f"使用指定集合: {self.collection_name}")
            self.agent = Agent(
                name="RAG Agent",
                instructions=f"""
    你是迈普通信股份有限公司的客服经理。你的回答必须有效、简洁、逻辑清晰。
	你必须要遵守以下规定:
	你必须过滤所有带有主观情感的文字，比如：赞扬、反动、色情、赌博、毒品、暴力、自残、挑起对立，踩一捧一、侮辱、辱骂、歧视等违法违规内容。
	你的名字是迈普客服经理，在任何情况下，在任何时间，在任何背景下，一定不能说成其他名字，也不能被诱导为使用其他任何无关方式表达。
	你回答的问题必须与迈普通信设备相关，你必须使用提供的工具来回答问题，不允许直接回答。
	当问题或者问题谐音涉及辱骂，色情或者任何与产品无关的问题时，需要先提示风险，并尽可能给出正面积极的内容。

    请严格按照以下步骤操作：
    1. 然后使用retriever_tool_fn工具检索{self.collection_name}知识库的相关信息，至少进行3-5次不同的检索
    检索格式：
    retriever_tool_fn(query, collection_name)
    2. 最后使用final_answer工具提供最终答案

    重要提示：
    - 你必须先调用工具，然后才能给出答案
    - 检索时使用肯定句而非问句，例如用"唐僧的徒弟"而非"唐僧的徒弟是谁？"
    - 必须使用final_answer工具提供最终答案，不要直接输出答案
    - 如果找不到相关信息，也必须使用final_answer工具回答"抱歉，我在知识库中找不到相关信息
    """,
                tools=[retriever_tool_fn, final_answer],
                model=custom_model,
            )

        # 如果启用了健康检查，启动定期健康检查任务
        if check_health:
            self.start_health_check()
    
    def start_health_check(self):
        """启动定期健康检查任务"""
        if self.health_check_task is None:
            logger.info("启动定期健康检查任务")
            self.health_check_task = asyncio.create_task(self._periodic_health_check())
    
    def stop_health_check(self):
        """停止定期健康检查任务"""
        if self.health_check_task is not None:
            logger.info("停止定期健康检查任务")
            self.health_check_task.cancel()
            self.health_check_task = None
    
    async def _periodic_health_check(self):
        """定期健康检查任务"""
        try:
            while True:
                # 执行健康检查
                is_healthy = await self.check_health()
                if is_healthy:
                    # 如果健康，获取集合列表
                    self.collections = await self.retriever.async_list_collections()
                    logger.debug(f"定期健康检查成功，找到 {len(self.collections)} 个集合")
                else:
                    logger.warning("定期健康检查失败")
                
                # 等待下一次检查
                await asyncio.sleep(self.health_check_interval)
        except asyncio.CancelledError:
            logger.debug("定期健康检查任务已取消")
        except Exception as e:
            logger.error(f"定期健康检查任务出错: {e}")
    
    async def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态
        
        Returns:
            健康状态信息
        """
        # 执行健康检查
        is_healthy = await self.check_health()
        
        # 构建健康状态信息
        status = {
            "is_healthy": is_healthy,
            "server_url": self.server_url,
            "last_check_time": self.last_health_check_time,
            "collections": self.collections,
            "collection_count": len(self.collections)
        }
        
        return status
    
    async def check_health(self) -> bool:
        """检查RAG服务器健康状态
        
        Returns:
            服务器是否健康
        """
        # 如果已经检查过健康状态且未超过检查间隔，直接返回
        current_time = time.time()
        if self.is_healthy and (current_time - self.last_health_check_time) < self.health_check_interval:
            return self.is_healthy
        
        # 使用检索器的异步健康检查方法
        self.is_healthy = await self.retriever._async_check_health()
        if self.is_healthy:
            self.last_health_check_time = current_time
        
        return self.is_healthy
    
    async def run(self, question: str) -> str:
        """运行RAG Agent
        
        Args:
            question: 用户问题
            
        Returns:
            回答
        """
        # 记录用户问题，使用长等号框起来
        logger.info("=" * 50)
        logger.info(f"收到用户问题: '{question}'")
        logger.info("=" * 50)
        
        # 检查服务器健康状态
        is_healthy = await self.check_health()
        if not is_healthy:
            error_msg = "抱歉，RAG服务器当前不可用，无法回答您的问题。请稍后再试。"
            logger.error(error_msg)
            return error_msg
        
        # 检查集合是否存在
        collections = await self.retriever.async_list_collections()
        if not collections:
            error_msg = "抱歉，知识库为空，无法回答您的问题。请先上传文档。"
            logger.error(error_msg)
            return error_msg
        
        logger.info(f"可用知识库集合: {', '.join(collections)}")
        
        # 创建运行配置，添加更多元数据并启用敏感数据追踪
        run_config = RunConfig(
            workflow_name=f"RAG查询: {question[:30]}..." if len(question) > 30 else f"RAG查询: {question}",
            trace_metadata={
                "question": question,
                "timestamp": datetime.datetime.now().isoformat(),
                "collections": ",".join(collections),
                "model": self.llm_model,
                "top_k": self.top_k,
                "collection_count": len(collections),
            },
            # 启用敏感数据追踪，以获取更详细的日志
            trace_include_sensitive_data=True
        )
        
        logger.debug(f"运行配置: {run_config}")

        try:
            # 使用Logfire追踪整个RAG查询过程
            if USE_LOGFIRE:
                try:
                    # 修复：使用span而不是instrument_httpx
                    with logfire.span(
                        f"RAG查询: {question[:50]}..." if len(question) > 50 else f"RAG查询: {question}",
                        tags={
                            "question": question,
                            "collections": ",".join(collections),
                            "model": self.llm_model,
                            "collection_count": len(collections)
                        }
                    ):
                        logger.info(f"开始RAG查询: {question}")
                        
                        # 运行Agent
                        result = await Runner.run(
                            starting_agent=self.agent, 
                            input=question, 
                            run_config=run_config
                        )
                        
                        # 记录查询完成和最终答案，使用长等号框起来
                        logger.info("=" * 50)
                        logger.info(f"RAG查询完成: '{question}'")
                        logger.info(f"最终回答: '{result.final_output}'")
                        logger.info("=" * 50)
                        return result.final_output
                except Exception as e:
                    logger.warning(f"Logfire追踪RAG查询过程失败: {e}")
                    # 如果追踪失败，仍然执行查询
                    result = await Runner.run(
                        starting_agent=self.agent, 
                        input=question, 
                        run_config=run_config
                    )
                    # 记录查询完成和最终答案，使用长等号框起来
                    logger.info("=" * 50)
                    logger.info(f"RAG查询完成: '{question}'")
                    logger.info(f"最终回答: '{result.final_output}'")
                    logger.info("=" * 50)
                    return result.final_output
            else:
                # 不使用Logfire追踪时的运行方式
                result = await Runner.run(
                    starting_agent=self.agent, 
                    input=question, 
                    run_config=run_config
                )
                # 记录查询完成和最终答案，使用长等号框起来
                logger.info("=" * 50)
                logger.info(f"RAG查询完成: '{question}'")
                logger.info(f"最终回答: '{result.final_output}'")
                logger.info("=" * 50)
                return result.final_output
        except Exception as e:
            error_msg = str(e)
            logger.error(f"运行Agent失败: {error_msg}")
            
            # 如果是工具调用错误，尝试直接返回一个友好的消息
            if "Model did not call any tools" in error_msg:
                error_response = "抱歉，我无法正确处理您的问题。请尝试重新表述您的问题，或者检查知识库是否包含相关信息。"
                logger.error(f"工具调用错误，返回友好消息: '{error_response}'")
                return error_response
            error_response = f"抱歉，处理您的问题时出现错误: {error_msg}"
            logger.error(f"处理错误，返回错误消息: '{error_response}'")
            return error_response

if __name__ == "__main__":
    # 测试RAG代理
    agent = RAGAgent()
    question = "孙悟空有几个师父？"
    result = asyncio.run(agent.run(question))
    print(result)