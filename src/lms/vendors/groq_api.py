from src.lms.vendors.openai_like import OpenAIStandardProvider
from groq import AsyncGroq, Groq
import instructor
import os
import backoff
import pydantic_core
from src.lms.cache_init import cache, old_cache
from src.lms.api_caching_fxns import safecache_wrapper
import groq

BACKOFF_TOLERANCE = 20

class GroqAPIProvider(OpenAIStandardProvider):
    def __init__(self):
        self.sync_client = instructor.patch(Groq(api_key=os.environ.get("GROQ_API_KEY"),))
        self.async_client = instructor.patch(AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"),))
        self.supports_response_model = True
    
    @backoff.on_exception(
            backoff.expo,
            (groq.RateLimitError, pydantic_core._pydantic_core.ValidationError),
            max_tries=BACKOFF_TOLERANCE,
            logger=None,
            on_giveup=lambda e: print(e) if isinstance(e, pydantic_core._pydantic_core.ValidationError) else None
        )
    def sync_chat_completion_with_response_model(self, messages, model, temperature, max_tokens, response_model):
        if not self.supports_response_model:
            raise ValueError("Code for this provider does not yet support response models")
        hit = safecache_wrapper.hit_cache(messages, model, temperature, response_model, cache, old_cache)
        if hit:
            return hit
        output = self.sync_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_model=response_model
        )
        safecache_wrapper.add_to_cache(messages, model, temperature, response_model, output, cache)
        return output

    @backoff.on_exception(
        backoff.expo,
        (groq.RateLimitError, pydantic_core._pydantic_core.ValidationError),
        max_tries=BACKOFF_TOLERANCE,
        logger=None,
        on_giveup=lambda e: print(e) if isinstance(e, pydantic_core._pydantic_core.ValidationError) else None
    )
    async def async_chat_completion_with_response_model(self, messages, model, temperature, max_tokens, response_model):
        if not self.supports_response_model:
            raise ValueError("Code for this provider does not yet support response models")
        hit = safecache_wrapper.hit_cache(messages, model, temperature,  response_model, cache, old_cache)
        if hit:
            return hit
        output = await self.async_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_model=response_model
        )
        safecache_wrapper.add_to_cache(messages, model, temperature,  response_model, output, cache)
        return output
