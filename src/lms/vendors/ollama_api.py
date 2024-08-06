import asyncio
from typing import Dict, List

import backoff
from ollama import AsyncClient, Client

from src.lms.cache_init import cache
from src.lms.vendors.openai_like import OpenAIStandardProvider

BACKOFF_TOLERANCE = 0

class OllamaAPIProvider(OpenAIStandardProvider):
    def __init__(self):
        self.sync_client = Client()
        self.async_client = AsyncClient()
        self.supports_response_model = False
    
    async def hit_ollama_async(self, messages: List[Dict], model_name: str, temperature: float = 0) -> str:

        response = await self.async_client.chat(model=model_name, messages=messages)
        return response['message']['content']
    
    def hit_ollama_sync(self, messages: List[Dict], model_name: str, temperature: float = 0) -> str:
        response = self.sync_client.chat(model=model_name, messages=messages)
        return response['message']['content']
      

    async def async_chat_completion(self, messages, model, temperature, max_tokens):
        hit = cache.hit_cache(messages, model, temperature, None)
        if hit:
            return hit
        result = await self.hit_ollama_async(
            messages=messages,
            model_name=model,
            temperature=temperature,
        )
        cache.add_to_cache(messages, model, temperature, None, result)
        return result

    # @backoff.on_exception(
    #     backoff.expo,
    #     (Exception),  # Replace with specific exceptions if known
    #     max_tries=BACKOFF_TOLERANCE,
    # )
    def sync_chat_completion(self, messages, model, temperature, max_tokens):
        hit = cache.hit_cache(messages, model, temperature, None)
        if hit:
            return hit
        print("Here")
        result = self.hit_ollama_sync(
            messages=messages,
            model_name=model,
            temperature=temperature,
        )
        cache.add_to_cache(messages, model, temperature, None, result)
        return result
        

if __name__ == "__main__":
    import asyncio
    messages = [
        {
            "role": "system",
            "content": "You are a helpful a ssistant that can answer questions abou t the capital of France."
        },
        {
            "role": "user",
            "content": "What  is the capital of France?   "
        }
    ]
    print("Hitting Ollama")
    response = asyncio.run(OllamaAPIProvider().async_chat_completion(
        messages=messages,
        model="gemma2:2b",
        temperature=0.0,
        max_tokens=150,
    ))
    print(response)