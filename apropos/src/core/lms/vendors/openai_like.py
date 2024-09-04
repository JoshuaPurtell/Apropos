import time

import backoff
import groq
import instructor
import openai
import together

# from apropos.src.core.lms.api_caching_fxns_diskcache import safecache_wrapper
# from apropos.src.core.lms.cache_init_diskcache import cache, old_cache
# from apropos.src.core.lms.api_caching_fxns_sqlite import #safecache_wrapper
from apropos.src.core.lms.cache_init import cache  # , old_cache
from apropos.src.core.lms.vendors.base import BaseProvider

BACKOFF_TOLERANCE = 200


class OpenAIStandardProvider(BaseProvider):
    def __init__(self, sync_client, async_client):
        self.sync_client = sync_client
        self.async_client = async_client
        self.supports_response_model = False

    @backoff.on_exception(
        backoff.expo,
        (
            openai.RateLimitError,
            together.error.RateLimitError,
            groq.RateLimitError,
            instructor.exceptions.InstructorRetryException,
            together.error.APIConnectionError,
            together.error.APIError,
        ),  # pydantic_core._pydantic_core.ValidationError
        max_tries=BACKOFF_TOLERANCE,
    )
    def sync_chat_completion_with_response_model(
        self, messages, model, temperature, max_tokens, response_model
    ):
        if not self.supports_response_model:
            raise ValueError(
                "Code for this provider does not yet support response models"
            )
        # hit = safecache_wrapper.hit_cache(messages, model, temperature, response_model, cache, old_cache)
        hit = cache.hit_cache(messages, model, temperature, response_model)
        if hit:
            return hit
        output = self.sync_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_model=response_model,
        )
        cache.add_to_cache(
            messages,
            model,
            temperature,
            response_model,
            output.choices[0].message.content,
        )
        # safecache_wrapper.add_to_cache(messages, model, temperature, response_model, output.choices[0].message.content, cache)
        return output.choices[0].message.content

    @backoff.on_exception(
        backoff.expo,
        (
            openai.RateLimitError,
            together.error.RateLimitError,
            together.error.Timeout,
            groq.RateLimitError,
            instructor.exceptions.InstructorRetryException,
            together.error.APIConnectionError,
            together.error.APIError,
        ),  # pydantic_core._pydantic_core.ValidationError,
        max_tries=BACKOFF_TOLERANCE,
    )
    async def async_chat_completion_with_response_model(
        self, messages, model, temperature, max_tokens, response_model
    ):
        if not self.supports_response_model:
            raise ValueError(
                "Code for this provider does not yet support response models"
            )
        # hit = safecache_wrapper.hit_cache(messages, model, temperature,  response_model, cache, old_cache)
        hit = cache.hit_cache(messages, model, temperature, response_model)
        # print("Hit true/false: ",hit is not None)
        if hit:
            return hit
        output = await self.async_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_model=response_model,
        )
        # safecache_wrapper.add_to_cache(messages, model, temperature,  response_model, output.choices[0].message.content, cache)
        cache.add_to_cache(
            messages,
            model,
            temperature,
            response_model,
            output.choices[0].message.content,
        )
        return output.choices[0].message.content

    @backoff.on_exception(
        backoff.expo,
        (
            openai.RateLimitError,
            together.error.RateLimitError,
            together.error.Timeout,
            groq.RateLimitError,
            instructor.exceptions.InstructorRetryException,
            together.error.APIConnectionError,
            together.error.APIError,
        ),
        max_tries=BACKOFF_TOLERANCE,
    )
    def sync_chat_completion(self, messages, model, temperature, max_tokens):
        t0 = time.time()
        # hit = safecache_wrapper.hit_cache(messages, model, temperature, None, cache, old_cache)
        hit = cache.hit_cache(messages, model, temperature, None)
        t1 = time.time()
        print(f"Time taken to hit cache: {t1-t0:.2e} seconds")
        if hit:
            return hit
        output = self.sync_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        # safecache_wrapper.add_to_cache(messages, model, temperature, None, output.choices[0].message.content, cache)
        cache.add_to_cache(
            messages, model, temperature, None, output.choices[0].message.content
        )
        return output.choices[0].message.content

    @backoff.on_exception(
        backoff.expo,
        (
            openai.RateLimitError,
            together.error.RateLimitError,
            together.error.Timeout,
            groq.RateLimitError,
            instructor.exceptions.InstructorRetryException,
            together.error.APIConnectionError,
            together.error.APIError,
        ),
        max_tries=BACKOFF_TOLERANCE,
    )
    async def async_chat_completion(self, messages, model, temperature, max_tokens):
        # hit = safecache_wrapper.hit_cache(messages, model, temperature, None, cache, old_cache)
        hit = cache.hit_cache(messages, model, temperature, None)
        if hit:
            return hit
        output = await self.async_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        # safecache_wrapper.add_to_cache(messages, model, temperature, None, output.choices[0].message.content, cache)
        cache.add_to_cache(
            messages, model, temperature, None, output.choices[0].message.content
        )
        return output.choices[0].message.content
