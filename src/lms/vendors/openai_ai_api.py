import asyncio
from os import sync
from typing import Dict, List, Type
import backoff
import loguru
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel
import openai

logger = loguru.logger
from src.lms.caching import (
    cache,
    generate_cache_key,
    generate_cache_key_with_response_model,
)
import instructor

client = instructor.patch(AsyncOpenAI())


## Using Instructor
async def hit_oai_with_model(
    messages: List[Dict],
    response_model: Type[BaseModel],
    model: str,
    temperature,
    max_tokens,
):
    r = await client.chat.completions.create(
        model=model,
        response_model=response_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return r


async def async_openai_chat_completion_with_response_model(
    messages: List[Dict],
    model: str = "gpt-3.5-turbo-0125",
    response_model: Type[BaseModel] = None,
    temperature=0.0,
    max_tokens=150,
) -> BaseModel:
    key = generate_cache_key_with_response_model(
        messages, model, temperature, response_model
    )
    if key in cache:
        return response_model.parse_raw(cache[key])
    output = await hit_oai_with_model(
        messages, response_model, model, temperature, max_tokens
    )
    cache[key] = output.json()
    return response_model.parse_raw(cache[key])


# TODO: make this default sync, not asyncio.run
def sync_openai_chat_completion_with_response_model(
    messages: List[Dict],
    model: str = "gpt-3.5-turbo-0125",
    response_model: Type[BaseModel] = None,
    temperature=0.0,
    max_tokens=150,
) -> BaseModel:
    return asyncio.run(
        async_openai_chat_completion_with_response_model(
            messages=messages,
            model=model,
            response_model=response_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    )


## Using Text Response
async def async_openai_chat_completion(
    messages: List[Dict],
    model: str = "gpt-3.5-turbo-0125",
    temperature=0.0,
    max_tokens=150,
) -> BaseModel:
    key = generate_cache_key(messages, model, temperature)
    if key in cache:
        return cache[key]
    output = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    cache[key] = output.choices[0].message.content
    return output.choices[0].message.content


@backoff.on_exception(
    backoff.expo,
    openai.RateLimitError,
    max_tries=3,
    giveup=lambda e: e.response.status_code != 429,
)
def sync_openai_chat_completion(
    messages: List[Dict],
    model: str = "gpt-3.5-turbo-0125",
    temperature=0.0,
    max_tokens=150,
) -> str:
    key = generate_cache_key(messages, model, temperature)
    if key in cache:
        return cache[key]
    c = OpenAI()
    output = c.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    cache[key] = output.choices[0].message.content
    return output.choices[0].message.content
