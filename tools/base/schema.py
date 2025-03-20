from enum import Enum
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field


class AgentState(str, Enum):
    """代理状态枚举"""

    IDLE = "idle"
    RUNNING = "running"
    FINISHED = "finished"


class MessageRole(str, Enum):
    """消息角色枚举"""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    """消息模型"""

    role: str
    content: str
    name: Optional[str] = None

    @classmethod
    def system_message(cls, content: str) -> "Message":
        return cls(role=MessageRole.SYSTEM, content=content)

    @classmethod
    def user_message(cls, content: str) -> "Message":
        return cls(role=MessageRole.USER, content=content)

    @classmethod
    def assistant_message(cls, content: str) -> "Message":
        return cls(role=MessageRole.ASSISTANT, content=content)

    @classmethod
    def tool_message(cls, content: str, name: str) -> "Message":
        return cls(role=MessageRole.TOOL, content=content, name=name)


class ToolCall(BaseModel):
    """工具调用模型"""

    name: str
    arguments: Dict[str, Union[str, int, float, bool, list, dict]] = Field(
        default_factory=dict
    )


class Tool(BaseModel):
    """工具基础模型"""

    name: str
    description: str

    async def execute(self, **kwargs) -> str:
        """执行工具操作"""
        raise NotImplementedError("子类必须实现此方法")
