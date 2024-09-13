import backoff
import groq
import instructor
import openai
import together
from apropos.src.core.lms.cache_init import cache, threaded_cache
from apropos.src.core.lms.vendors.base import BaseProvider

BACKOFF_TOLERANCE = 0  # 200


class OpenAIStandardProvider(BaseProvider):
    def __init__(self, sync_client, async_client, multi_threaded=False):
        self.sync_client = sync_client
        self.async_client = async_client
        self.supports_response_model = False
        self.multi_threaded = multi_threaded

    def hit_cache(self, messages, model, temperature, response_model):
        if self.multi_threaded:
            return threaded_cache.hit_cache(
                messages, model, temperature, response_model
            )
        else:
            return cache.hit_cache(messages, model, temperature, response_model)

    def add_to_cache(self, messages, model, temperature, response_model, content):
        if self.multi_threaded:
            return threaded_cache.add_to_cache(
                messages, model, temperature, response_model, content
            )
        else:
            return cache.add_to_cache(
                messages, model, temperature, response_model, content
            )

    @backoff.on_exception(
        backoff.expo,
        (
            openai.RateLimitError,
            together.error.RateLimitError,
            groq.RateLimitError,
            instructor.exceptions.InstructorRetryException,
            together.error.APIConnectionError,
            together.error.APIError,
        ),
        max_tries=BACKOFF_TOLERANCE,
    )
    def sync_chat_completion_with_response_model(
        self, messages, model, temperature, max_tokens, response_model
    ):
        if "o1" in model:
            message = f"<instructions>{messages[0]['content']}</instructions>\n\n<information>{messages[1]['content']}</information>"
            messages = [{"role": "user", "content": message}]
            temperature = 1
        if not self.supports_response_model:
            raise ValueError(
                "Code for this provider does not yet support response models"
            )
        hit = self.hit_cache(messages, model, temperature, response_model)
        if hit:
            return hit
        output = self.sync_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            # max_tokens=max_tokens,
            response_model=response_model,
        )
        self.add_to_cache(
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
        on_backoff=lambda details: print(
            f"Backing off: {details['tries']} tries, {details['wait']} seconds"
        )
        if details["tries"] > 3
        else None,
    )
    async def async_chat_completion_with_response_model(
        self, messages, model, temperature, max_tokens, response_model
    ):
        if not self.supports_response_model:
            raise ValueError(
                "Code for this provider does not yet support response models"
            )
        hit = self.hit_cache(messages, model, temperature, response_model)
        if hit:
            return hit
        output = await self.async_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            # max_tokens=max_tokens,
            response_model=response_model,
        )
        self.add_to_cache(
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
        if "o1" in model:
            message = f"<instructions>{messages[0]['content']}</instructions>\n\n<information>{messages[1]['content']}</information>"
            messages = [{"role": "user", "content": message}]
            temperature = 1
        hit = self.hit_cache(messages, model, temperature, None)
        if hit:
            return hit
        output = self.sync_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            # max_completion_tokens=max_tokens,
        )
        self.add_to_cache(
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
        if "o1" in model:
            message = f"<instructions>{messages[0]['content']}</instructions>\n\n<information>{messages[1]['content']}</information>"
            messages = [{"role": "user", "content": message}]
            temperature = 1
        hit = self.hit_cache(messages, model, temperature, None)
        if hit:
            return hit
        output = await self.async_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            # max_completion_tokens=max_tokens,
        )
        self.add_to_cache(
            messages, model, temperature, None, output.choices[0].message.content
        )
        return output.choices[0].message.content
