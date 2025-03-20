from typing import List, Optional
from pydantic import BaseModel, Field

from simple_react_agent.schema import Message


class Memory(BaseModel):
    """代理记忆管理类"""

    messages: List[Message] = Field(default_factory=list)
    max_messages: int = Field(default=100)

    def add_message(self, message: Message) -> None:
        """Add a message to memory"""
        self.messages.append(message)
        # Optional: Implement message limit
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

    def _trim_messages(self) -> None:
        """如果消息数量超过上限，修剪最旧的消息"""
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

    def get_last_user_message(self) -> Optional[Message]:
        """获取最后一条用户消息"""
        for message in reversed(self.messages):
            if message.role == "user":
                return message
        return None

    def get_last_n_messages(self, n: int) -> List[Message]:
        """获取最后n条消息"""
        return self.messages[-n:] if n < len(self.messages) else self.messages

    def clear(self) -> None:
        """清空记忆"""
        self.messages = []

    def has_duplicate_exchanges(self, threshold: int = 2) -> bool:
        """检测是否有重复对话，用于防止循环"""
        if len(self.messages) < 4:
            return False

        # 获取最近的交互对
        recent_pairs = []
        for i in range(len(self.messages) - 1):
            if (
                self.messages[i].role == "user"
                and self.messages[i + 1].role == "assistant"
            ):
                pair = (self.messages[i].content, self.messages[i + 1].content)
                recent_pairs.append(pair)

        # 检查是否有重复对超过阈值
        for pair in recent_pairs:
            if recent_pairs.count(pair) >= threshold:
                return True

        return False
