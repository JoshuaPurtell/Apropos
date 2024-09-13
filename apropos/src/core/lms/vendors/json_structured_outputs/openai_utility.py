from apropos.src.core.lms.vendors.openai_like import OpenAIStandardProvider
from openai import OpenAI, AsyncOpenAI


class OpenAIUtility(OpenAIStandardProvider):
    def __init__(self):
        super().__init__(sync_client=OpenAI(), async_client=AsyncOpenAI())
