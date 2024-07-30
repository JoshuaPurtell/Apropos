import asyncio
import os
from typing import Dict, List, Optional, Type

import groq
from groq import AsyncGroq
from pydantic import BaseModel

from src.lms.caching import (
    cache,
    generate_cache_key,
    generate_cache_key_with_response_model,
)

MAX_RETRIES = 1000
DELAY = 60

client = AsyncGroq(
    api_key=os.environ.get("GROQ_API_KEY"),
)


def retry_on_ratelimit(
    max_retries=MAX_RETRIES,
    delay=DELAY,
    exceptions=(groq.RateLimitError, groq.APIConnectionError),
):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_retries:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts < max_retries:
                        await asyncio.sleep(delay)
                    else:
                        raise Exception

        return wrapper

    return decorator


@retry_on_ratelimit(max_retries=MAX_RETRIES, delay=DELAY)
async def async_groq_chat_completion(
    messages: list[dict[str, str]],
    model: str,
    temperature=0,
    response_model: Optional[Type[BaseModel]] = None,
) -> dict[str, str]:
    if response_model:
        key = generate_cache_key_with_response_model(
            messages, model, temperature, response_model
        )
        if key in cache:
            return response_model.parse_raw(cache[key])

        result = await client.chat.completions.create(
            messages=messages,
            model=model,
            response_model=response_model,
            temperature=temperature,
        )
        cache[key] = result.json()
        return response_model.parse_raw(cache[key])

    key = generate_cache_key(messages, model, temperature)
    if key in cache:
        return cache[key]
    result = await client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=temperature,
    )
    # print("Done")
    cache[key] = result.choices[0].message.content
    return result.choices[0].message.content


def groq_chat_completion(
    messages: List[Dict[str, str]],
    model: str,
    response_model: Optional[Type[BaseModel]] = None,
):
    if response_model:
        return asyncio.run(
            async_groq_chat_completion(
                messages=messages,
                model=model,
                response_model=response_model,
            )
        )

    return asyncio.run(
        async_groq_chat_completion(
            messages=messages,
            model=model,
        )
    )
