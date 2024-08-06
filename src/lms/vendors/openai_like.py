import backoff
import openai
import together
import groq
from src.lms.cache_init import cache

import instructor
from src.lms.vendors.base import BaseProvider

BACKOFF_TOLERANCE = 10

class OpenAIStandardProvider(BaseProvider):
    def __init__(self, sync_client, async_client):
        self.sync_client = sync_client
        self.async_client = async_client
        self.supports_response_model = False

    @backoff.on_exception(
        backoff.expo,
        (openai.RateLimitError, together.error.RateLimitError, groq.RateLimitError, instructor.exceptions.InstructorRetryException, together.error.APIConnectionError, together.error.APIError),#pydantic_core._pydantic_core.ValidationError
        max_tries=BACKOFF_TOLERANCE,
    )
    def sync_chat_completion_with_response_model(self, messages, model, temperature, max_tokens, response_model):
        if not self.supports_response_model:
            raise ValueError("Code for this provider does not yet support response models")
        hit = cache.hit_cache(messages, model, temperature, response_model)
        if hit:
            return hit
        output = self.sync_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_model=response_model
        )
        cache.add_to_cache(messages, model, temperature, response_model, output.choices[0].message.content)
        return output.choices[0].message.content

    @backoff.on_exception(
        backoff.expo,
        (openai.RateLimitError, together.error.RateLimitError, together.error.Timeout, groq.RateLimitError, instructor.exceptions.InstructorRetryException, together.error.APIConnectionError, together.error.APIError),#pydantic_core._pydantic_core.ValidationError, 
        max_tries=BACKOFF_TOLERANCE,
    )
    async def async_chat_completion_with_response_model(self, messages, model, temperature, max_tokens, response_model):
        if not self.supports_response_model:
            raise ValueError("Code for this provider does not yet support response models")
        hit = cache.hit_cache(messages, model, temperature, response_model)
        if hit:
            return hit
        output = await self.async_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_model=response_model
        )
        cache.add_to_cache(messages, model, temperature, response_model, output.choices[0].message.content)
        return output.choices[0].message.content
    
    @backoff.on_exception(
        backoff.expo,
        (openai.RateLimitError, together.error.RateLimitError, together.error.Timeout, groq.RateLimitError, instructor.exceptions.InstructorRetryException, together.error.APIConnectionError, together.error.APIError),
        max_tries=BACKOFF_TOLERANCE,
    )
    def sync_chat_completion(self, messages, model, temperature, max_tokens):
        hit = cache.hit_cache(messages, model, temperature, None)
        if hit:
            return hit
        output = self.sync_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        cache.add_to_cache(messages, model, temperature, None, output.choices[0].message.content)
        return output.choices[0].message.content
    
    @backoff.on_exception(
        backoff.expo,
        (openai.RateLimitError, together.error.RateLimitError, together.error.Timeout, groq.RateLimitError, instructor.exceptions.InstructorRetryException, together.error.APIConnectionError, together.error.APIError),
        max_tries=BACKOFF_TOLERANCE,
    )
    async def async_chat_completion(self, messages, model, temperature, max_tokens):
        hit = cache.hit_cache(messages, model, temperature, None)
        if hit:
            return hit
        output = await self.async_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        cache.add_to_cache(messages, model, temperature, None, output.choices[0].message.content)
        return output.choices[0].message.content

