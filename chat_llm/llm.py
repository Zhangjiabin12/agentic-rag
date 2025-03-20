
from typing import List, Dict, AsyncGenerator

from openai import AsyncOpenAI



class AsyncChatSession:
    def __init__(self, system_prompt: str,  max_messages_length: int = 10,api_key: str = None,base_url:str='', model:str=''):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]
        # 异步客户端初始化
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        self.max_messages_length = max_messages_length

    async def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        if len(self.messages) > self.max_messages_length:
            self.messages.pop(0)

    async def stream_response(self, temperature: float = 0.7) -> AsyncGenerator[str, None]:
        try:
            response = await self.client.chat.completions.create(
                messages=self.messages,
                model=self.model,
                temperature=temperature,
                stream=True,
                max_tokens=8192
            )
            
            full_response = []
            async for chunk in response:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response.append(content)
                    yield content
            
            await self.add_message("assistant", "".join(full_response))

        except Exception as e:
            yield f"API Error: {str(e)}"

    async def get_response(self, temperature: float = 0.7) -> str:
        try:
            response = await self.client.chat.completions.create(
                messages=self.messages,
                model=self.model,
                temperature=temperature,
                stream=False,
                max_tokens=8192
            )
            
            content = response.choices[0].message.content
            await self.add_message("assistant", content)
            return content
            
        except Exception as e:
            return f"API Error: {str(e)}"



        
