import asyncio
from typing import Dict, List

import instructor
from ollama import AsyncClient, Client
from openai import AsyncOpenAI, OpenAI

from apropos.src.core.lms.cache_init import cache
from apropos.src.core.lms.vendors.openai_like import OpenAIStandardProvider

BACKOFF_TOLERANCE = 0


class OllamaAPIProvider(OpenAIStandardProvider):
    def __init__(self):
        self.sync_client_structured = instructor.from_openai(
            OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama",  # required, but unused
            ),
            mode=instructor.Mode.JSON,
        )
        self.async_client_structured = instructor.from_openai(
            AsyncOpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama",  # required, but unused
            ),
            mode=instructor.Mode.JSON,
        )
        self.sync_client = Client()
        self.async_client = AsyncClient()
        self.supports_response_model = True

    async def hit_ollama_async_structured(
        self,
        messages: List[Dict],
        model_name: str,
        temperature: float = 0,
        response_model=None,
    ) -> str:
        response = await self.async_client_structured.chat.completions.create(
            model=model_name, messages=messages, response_model=response_model
        )
        return response

    def hit_ollama_sync_structured(
        self,
        messages: List[Dict],
        model_name: str,
        temperature: float = 0,
        response_model=None,
    ) -> str:
        response = self.sync_client_structured.chat.completions.create(
            model=model_name, messages=messages, response_model=response_model
        )
        return response

    async def hit_ollama_async(
        self,
        messages: List[Dict],
        model_name: str,
        temperature: float = 0,
        response_model=None,
    ) -> str:
        if response_model:
            response = await self.hit_ollama_async_structured(
                messages, model_name, temperature, response_model
            )
        else:
            response = await self.async_client.chat(model=model_name, messages=messages)
            response = response["message"]["content"]
        return response

    def hit_ollama_sync(
        self,
        messages: List[Dict],
        model_name: str,
        temperature: float = 0,
        response_model=None,
    ) -> str:
        if response_model:
            response = self.hit_ollama_sync_structured(
                messages, model_name, temperature, response_model
            )
        else:
            response = self.sync_client.chat(model=model_name, messages=messages)
            response = response["message"]["content"]
        return response

    async def async_chat_completion(
        self, messages, model, temperature, max_tokens, response_model=None
    ):
        hit = cache.hit_cache(messages, model, temperature, None)
        if hit:
            return hit
        result = await self.hit_ollama_async(
            messages=messages,
            model_name=model,
            temperature=temperature,
            response_model=response_model,
        )
        cache.add_to_cache(messages, model, temperature, None, result)
        return result

    def sync_chat_completion(
        self, messages, model, temperature, max_tokens, response_model=None
    ):
        hit = cache.hit_cache(messages, model, temperature, None)
        if hit:
            return hit
        result = self.hit_ollama_sync(
            messages=messages,
            model_name=model,
            temperature=temperature,
            response_model=response_model,
        )
        cache.add_to_cache(messages, model, temperature, None, result)
        return result

    async def async_chat_completion_with_response_model(
        self, messages, model, temperature, max_tokens, response_model
    ):
        return await self.async_chat_completion(
            messages, model, temperature, max_tokens, response_model
        )

    def sync_chat_completion_with_response_model(
        self, messages, model, temperature, max_tokens, response_model
    ):
        return self.sync_chat_completion(
            messages, model, temperature, max_tokens, response_model
        )


if __name__ == "__main__":
    import asyncio

    from pydantic import BaseModel

    class CapitalResponse(BaseModel):
        capital: str

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that can answer questions about the capital of France.",
        },
        {"role": "user", "content": "What is the capital of France? "},
    ]
    response = asyncio.run(
        OllamaAPIProvider().async_chat_completion(
            messages=messages,
            model="gemma2:2b",
            temperature=0.0,
            max_tokens=150,  #
            # response_model=CapitalResponse
        )
    )
    # print(response)
    print(response)
