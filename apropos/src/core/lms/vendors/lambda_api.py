from openai import OpenAI, AsyncOpenAI
import os
import backoff
import openai
from apropos.src.core.lms.vendors.openai_like import OpenAIStandardProvider
import os
from apropos.src.core.lms.vendors.json_structured_outputs.core import (
    add_json_instructions_to_messages,
    extract_pydantic_model_from_response_sync,
    extract_pydantic_model_from_response_async,
)

BACKOFF_TOLERANCE = 10


class LambdaAPIProvider(OpenAIStandardProvider):
    def __init__(self):
        openai_api_key = os.getenv("LAMBDA_API_KEY")
        openai_api_base = "https://api.lambdalabs.com/v1"
        self.sync_client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        self.async_client = AsyncOpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        self.supports_response_model = True

    @backoff.on_exception(
        backoff.expo,
        (Exception, openai.InternalServerError),
        max_tries=BACKOFF_TOLERANCE,
        giveup=lambda e: getattr(e, "status_code", None) != 429,
    )
    def sync_chat_completion_with_response_model(
        self, messages, model, temperature, max_tokens, response_model
    ):
        messages_with_json_formatting_instructions = add_json_instructions_to_messages(
            messages, response_model
        )
        raw_text_api_response = self.sync_chat_completion(
            messages_with_json_formatting_instructions, model, temperature, max_tokens
        )
        structured_api_response = extract_pydantic_model_from_response_sync(
            raw_text_api_response, response_model
        )
        return structured_api_response

    @backoff.on_exception(
        backoff.expo,
        (Exception, openai.InternalServerError),
        max_tries=BACKOFF_TOLERANCE,
        giveup=lambda e: getattr(e, "status_code", None) != 429,
    )
    async def async_chat_completion_with_response_model(
        self, messages, model, temperature, max_tokens, response_model
    ):
        messages_with_json_formatting_instructions = add_json_instructions_to_messages(
            messages, response_model
        )
        raw_text_api_response = await self.async_chat_completion(
            messages_with_json_formatting_instructions, model, temperature, max_tokens
        )
        structured_api_response = await extract_pydantic_model_from_response_async(
            raw_text_api_response, response_model
        )
        return structured_api_response
