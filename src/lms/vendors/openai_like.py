import backoff
import openai

from src.lms.api_caching_fxns import safecache_wrapper
from src.lms.cache_init import cache, old_cache
from src.lms.vendors.base import BaseProvider

class OpenAIStandardProvider(BaseProvider):
    def __init__(self, sync_client, async_client):
        self.sync_client = sync_client
        self.async_client = async_client
        self.supports_response_model = False
    
    @backoff.on_exception(
        backoff.expo,
        openai.RateLimitError,
        max_tries=20,
        logger=None
    )
    def sync_chat_completion(self, messages, model, temperature, max_tokens):
        hit = safecache_wrapper.hit_cache(messages, model, temperature, None, cache, old_cache)
        if hit:
            return hit
        output = self.sync_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        safecache_wrapper.add_to_cache(messages, model, temperature, None, output.choices[0].message.content, cache)
        return output.choices[0].message.content
    
    @backoff.on_exception(
        backoff.expo,
        openai.RateLimitError,
        max_tries=20,
        giveup=lambda e: e.response.status_code != 429,
        logger=None
    )
    async def async_chat_completion(self, messages, model, temperature, max_tokens):
        hit = safecache_wrapper.hit_cache(messages, model, temperature, None, cache, old_cache)
        if hit:
            return hit
        output = await self.async_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        safecache_wrapper.add_to_cache(messages, model, temperature, None, output.choices[0].message.content, cache)
        return output.choices[0].message.content

