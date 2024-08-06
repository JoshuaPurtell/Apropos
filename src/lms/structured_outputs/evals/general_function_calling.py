### Try mirascope
from typing import Literal, Type

from mirascope.openai import OpenAIExtractor
from pydantic import BaseModel


def get_mirascope_class(system_message, user_message, extract_object):
    mirascope_format = f"""
SYSTEM:
{system_message}
USER:
{user_message}
"""

    class Extractor(OpenAIExtractor[extract_object]):
        extract_schema: Type[extract_object] = extract_object
        prompt_template = mirascope_format

    structured_response = Extractor().extract()
    return structured_response


if __name__ == "__main__":

    class TaskDetails(BaseModel):
        description: str
        due_date: str
        priority: Literal["low", "normal", "high"]

    system_message = """
Here's the info: """
    user_message = """
# Task Details
- Description: Write a Python function that takes a string as input and returns the number of vowels in the string.
- Due Date: 2022-12-31
- Priority: high
"""
    task_details = get_mirascope_class(system_message, user_message, TaskDetails)
    print(task_details)
    print(task_details.description)
    print(task_details.due_date)
    print(task_details.priority)
