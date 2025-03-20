"""
LLM输出格式化工具

此模块提供了用于处理大语言模型(LLM)输出的格式化功能，
能处理多种常见的输出格式问题，如代码块、JSON格式错误等。
"""

import re
import json
import ast
import logging
from typing import Any, Dict, List, Union, Type, TypeVar, Optional

# 创建日志记录器
logger = logging.getLogger("llm_formatter")

# 定义泛型类型变量用于输出类型
T = TypeVar('T')

def format_llm_output(content: str, output_type: Type[T] = None) -> Union[Dict, List, str, T]:
    """
    格式化LLM输出内容，处理各种常见格式问题
    
    Args:
        content: LLM返回的原始内容
        output_type: 可选，期望的输出类型（Pydantic模型类）
        
    Returns:
        格式化后的内容，如果提供了output_type，则返回该类型的实例
    """
    logger.debug(f"开始格式化LLM输出: {content[:100]}...")
    
    # 去除首尾空白
    content = content.strip()
    
    # 移除思考过程
    if "<think>" in content and "</think>" in content:
        end_of_think = content.find("</think>") + len("</think>")
        content = content[end_of_think:].strip()
    
    # 移除markdown代码块
    if content.startswith("```") and content.endswith("```"):
        # 处理各种代码块情况
        if content.startswith("```python"):
            content = content[9:-3].strip()
        elif content.startswith("```json"):
            content = content[7:-3].strip()
        elif content.startswith("```str"):
            content = content[6:-3].strip()
        elif content.startswith("```\n"):
            content = content[4:-3].strip()
        else:
            # 通用代码块
            match = re.match(r"```\w*\n?(.*?)\n?```", content, re.DOTALL)
            if match:
                content = match.group(1).strip()
    
    # 移除注释
    content = re.sub(r'\s*#.*$', '', content, flags=re.MULTILINE)
    
    # 移除空行，保留格式
    content = '\n'.join(line for line in content.splitlines() if line.strip())
    
    # 尝试解析为Python对象
    try:
        # 先尝试作为JSON解析
        result = json.loads(content)
        logger.debug(f"成功使用JSON解析: {type(result)}")
    except json.JSONDecodeError:
        try:
            # 尝试用ast.literal_eval解析
            result = ast.literal_eval(content)
            logger.debug(f"成功使用ast.literal_eval解析: {type(result)}")
        except (ValueError, SyntaxError):
            # 尝试从文本中提取JSON或字典
            try:
                matches = re.findall(r"(\[.*?\]|\{.*?\})", content, re.DOTALL)
                if matches:
                    result = ast.literal_eval(matches[0])
                    logger.debug(f"成功从文本中提取并解析: {type(result)}")
                else:
                    # 如果无法解析，返回原始内容
                    result = content
                    logger.debug("无法解析，返回原始内容")
            except Exception as e:
                logger.warning(f"从文本提取解析失败: {e}")
                result = content
    
    # 如果提供了输出类型，尝试转换
    if output_type and hasattr(output_type, "parse_obj") and isinstance(result, dict):
        try:
            result = output_type.parse_obj(result)
            logger.debug(f"成功转换为目标类型: {output_type.__name__}")
        except Exception as e:
            logger.error(f"转换为目标类型失败: {e}")
    
    return result

def extract_json(content: str) -> Optional[Dict]:
    """
    从文本中提取JSON对象
    
    Args:
        content: 包含JSON的文本
        
    Returns:
        提取的JSON对象，如果提取失败则返回None
    """
    # 使用正则表达式查找JSON对象
    matches = re.findall(r"\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}", content, re.DOTALL)
    
    if not matches:
        return None
    
    # 尝试解析找到的每个可能的JSON
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    
    return None

def clean_llm_text(content: str) -> str:
    """
    清理LLM生成的文本，移除可能的前缀和后缀
    
    Args:
        content: LLM生成的原始文本
        
    Returns:
        清理后的文本
    """
    # 移除常见的前缀
    prefixes = [
        "好的，", "好的！", "以下是", "Here's", "Here is", "Sure,", "Sure!", 
        "Of course,", "Of course!", "I'll", "Below is"
    ]
    
    for prefix in prefixes:
        if content.startswith(prefix):
            content = content[len(prefix):].lstrip()
    
    # 移除常见的后缀
    suffixes = [
        "希望这对您有所帮助！", "希望这对你有帮助。", "Let me know if you need anything else!",
        "如果您有任何问题，请告诉我。", "请问还有其他问题吗？", "如需更多帮助，请随时告诉我。"
    ]
    
    for suffix in suffixes:
        if content.endswith(suffix):
            content = content[:-len(suffix)].rstrip()
    
    return content 