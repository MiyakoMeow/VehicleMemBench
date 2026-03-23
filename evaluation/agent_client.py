import traceback
from openai import OpenAI
import os
from typing import List, Dict, Any, Optional

class AgentClient:
    """
    Client for interacting with LLM API using OpenAI-like interface.
    """
    
    def __init__(self, 
                 api_base: str = "",
                 api_key: Optional[str] = "",
                 model: str = "",
                 temperature: float = 0.0,
                 max_tokens: int = 8192):
        """
        Initialize the AgentClient.
        
        Args:
            api_base: Base URL for the API
            api_key: API key (reads from environment if None)
            model: Model name to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens in the response
        """
        self.api_base = api_base
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        if not self.api_key:
            raise ValueError("API key must be provided either in constructor or as OPENAI_API_KEY environment variable")
        self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)
    

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Send a chat completion request to the LLM API.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
                    (e.g., [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}])

        Returns:
            The assistant's response content as a string, along with token usage information
        """
        response = ""
        prompt_token_length = 0
        completion_token_length = 0
        total_token_length = 0

        retry_num = 3

        while True:
            try:
                chat_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=300
                )
                
                response = chat_response.choices[0].message.content.strip()
                total_token_length += chat_response.usage.total_tokens
                prompt_token_length += chat_response.usage.prompt_tokens
                completion_token_length += chat_response.usage.completion_tokens
                
                # Successfully obtained response, break out of loop
                break
                    
            except Exception as e:
                retry_num -= 1
                if retry_num == 0:
                    stack_trace = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
                    print(f"<<<<<<Agent Error>>>>>>\n{stack_trace}")
                    break

        # print(f"<<<<<<Agent Response>>>>>>\n{response}")
        return response, total_token_length, prompt_token_length, completion_token_length
