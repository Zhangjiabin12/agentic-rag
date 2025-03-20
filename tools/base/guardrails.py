"""
自定义安全防护机制

此模块提供了用于检查用户输入和系统输出的安全防护机制。
"""

from agents import InputGuardrail, OutputGuardrail, RunContextWrapper
from typing import Tuple, List, Any, Callable, Awaitable

class ContentSafetyGuard(InputGuardrail):
    """
    检查用户输入是否包含不适当内容的防护机制
    """
    
    def __init__(self, blocked_keywords: List[str] = None):
        """
        初始化内容安全防护
        
        Args:
            blocked_keywords: 阻止的关键词列表，默认为None则使用内置列表
        """
        self.blocked_keywords = blocked_keywords or [
            "自杀", "自残", "违法", "毒品", "赌博", "色情", "暴力",
            "恐怖", "歧视", "侮辱", "仇恨", "政治敏感"
        ]
    
    async def check(self, user_input: str, run_context: RunContextWrapper) -> Tuple[bool, str]:
        """
        检查用户输入
        
        Args:
            user_input: 用户输入文本
            run_context: 运行上下文
            
        Returns:
            (是否通过, 拒绝原因)
        """
        # 检查是否包含阻止的关键词
        for keyword in self.blocked_keywords:
            if keyword in user_input:
                return False, f"您的输入包含不适当的内容 '{keyword}'，请修改后重试。"
        
        # 检查输入长度
        if len(user_input) < 5:
            return False, "您的输入太短，请提供更详细的信息。"
        
        if len(user_input) > 1000:
            return False, "您的输入太长，请简化您的请求。"
        
        return True, ""


# OutputGuardrail需要一个guardrail_function参数，所以我们定义一个函数而不是类
async def check_story_quality(output: str, run_context: RunContextWrapper) -> Tuple[bool, str]:
    """
    检查生成的故事是否符合质量要求
    
    Args:
        output: 生成的内容
        run_context: 运行上下文
        
    Returns:
        (是否通过, 修改建议)
    """
    # 检查输出长度
    if len(output) < 100:
        return False, "生成的故事太短，请提供更丰富的内容。"
    
    # 检查是否包含对话
    if '"' not in output and "'" not in output and "：" not in output:
        return False, "故事缺少对话内容，请增加一些对话使故事更生动。"
    
    # 检查是否有明显的开始和结束
    paragraphs = output.split("\n\n")
    if len(paragraphs) < 3:
        return False, "故事结构不完整，请包含明确的开始、中间和结束。"
    
    return True, ""

# 创建StoryQualityGuard实例
def create_story_quality_guard() -> OutputGuardrail:
    """
    创建故事质量检查防护实例
    
    Returns:
        OutputGuardrail实例
    """
    return OutputGuardrail(guardrail_function=check_story_quality) 