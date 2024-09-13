import os

import backoff
import groq
import instructor
import pydantic_core
from groq import AsyncGroq, Groq

from apropos.src.core.lms.cache_init import cache
from apropos.src.core.lms.vendors.openai_like import OpenAIStandardProvider

BACKOFF_TOLERANCE = 30


class GroqAPIProvider(OpenAIStandardProvider):
    def __init__(self, force_structured_output=False, multi_threaded=False):
        if force_structured_output:
            self.sync_client = instructor.patch(
                Groq(api_key=os.environ.get("GROQ_API_KEY"))
            )
            self.async_client = instructor.patch(
                AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))
            )
        else:
            self.sync_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
            self.async_client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))
        self.supports_response_model = True
        self.multi_threaded = multi_threaded

    @backoff.on_exception(
        backoff.expo,
        (
            groq.RateLimitError,
            groq.APITimeoutError,
            pydantic_core._pydantic_core.ValidationError,
        ),
        max_tries=BACKOFF_TOLERANCE,
        logger=None,
        on_giveup=lambda e: print(e)
        if isinstance(e, pydantic_core._pydantic_core.ValidationError)
        else None,
    )
    def sync_chat_completion_with_response_model(
        self, messages, model, temperature, max_tokens, response_model
    ):
        hit = cache.hit_cache(messages, model, temperature, None)
        if hit:
            return hit
        output = self.sync_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_model=response_model,
        )
        cache.add_to_cache(messages, model, temperature, None, output)
        return output

    @backoff.on_exception(
        backoff.expo,
        (
            groq.RateLimitError,
            groq.APITimeoutError,
            pydantic_core._pydantic_core.ValidationError,
        ),
        max_tries=BACKOFF_TOLERANCE,
        logger=None,
        on_giveup=lambda e: print(e)
        if isinstance(e, pydantic_core._pydantic_core.ValidationError)
        else None,
    )
    async def async_chat_completion_with_response_model(
        self, messages, model, temperature, max_tokens, response_model
    ):
        hit = cache.hit_cache(messages, model, temperature, None)
        if hit:
            return hit
        output = await self.async_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_model=response_model,
        )
        cache.add_to_cache(messages, model, temperature, None, output)
        return output


if __name__ == "__main__":
    import asyncio

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that can answer questions about the capital of France.",
        },
        {"role": "user", "content": "What is the capital of France?"},
    ]
    response = asyncio.run(
        GroqAPIProvider().async_chat_completion(
            messages=messages,
            model="llama3-8b-8192",
            temperature=0.0,
            max_tokens=150,
        )
    )
    print(response)
