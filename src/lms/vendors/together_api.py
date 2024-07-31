from src.lms.vendors.openai_like import OpenAIStandardProvider
from together import AsyncTogether, Together
import instructor
import os

class TogetherAPIProvider(OpenAIStandardProvider):
    def __init__(self):
        self.sync_client = instructor.patch(Together(api_key=os.getenv("TOGETHER_API_KEY")))
        self.async_client = instructor.patch(AsyncTogether(api_key=os.getenv("TOGETHER_API_KEY")))
        self.supports_response_model = False
