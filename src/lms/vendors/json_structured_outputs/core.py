from typing import Dict, List, Optional, Type, Tuple

import loguru

from pydantic import BaseModel

import json
import json
from src.lms.vendors.json_structured_outputs.groq_healing import groq_json_debugger
import ast
from typing import Dict, get_type_hints

logger = loguru.logger

def add_json_instructions_to_prompt(
    system_prompt, user_prompt, response_model: Optional[Type[BaseModel]] = None
) -> Tuple[str, str]:
    if response_model:
        dictified = response_model.schema()

        if "$defs" in dictified:
            raise ValueError("Nesting not supported in response model")
        type_hints = get_type_hints(response_model)

        type_map = {
            List[str]: "List[str]",
            List[int]: "List[int]",
            List[float]: "List[float]",
            List[bool]: "List[bool]",
            List[Dict]: "List[Dict]",
            int: "int",
            float: "float",
            bool: "bool",
            str: "str",
            Optional[str]: "str",
            Optional[int]: "int",
            Optional[float]: "float",
            Dict[str, str]: "Dict[str,str]",
            Dict[str, int]: "Dict[str,int]",
            Dict[str, float]: "Dict[str,float]",
            Dict[str, bool]: "Dict[str,bool]",
            Dict[int, str]: "Dict[int,str]",
            Dict[int, int]: "Dict[int,int]",
            Dict[int, float]: "Dict[int,float]",
            Dict[int, bool]: "Dict[int,bool]",
        }

        for k, v in type_hints.items():
            if v in type_map:
                type_hints[k] = type_map[v]
        stringified = ""
        example_dict = {
            "str": "<Your type-str response here>",
            "int": "<Your type-int response here>",
            "float": "<Your type-float response here>",
            "bool": "<Your type-bool response here>",
            "List[str]": ["<Your type-str response here>"],
            "List[int]": ["<Your type-int response here>"],
            "List[float]": ["<Your type-float response here>"],
            "List[bool]": ["<Your type-bool response here>"],
            "List[Dict]": [
                {"<Your type-str response here>": "<Your type-str response here>"}
            ],
            "Dict[str,str]": {
                "<Your type-str response here>": "<Your type-str response here>"
            },
            "Dict[str,int]": {
                "<Your type-str response here>": "<Your type-int response here>"
            },
            "Dict[str,float]": {
                "<Your type-str response here>": "<Your type-float response here>"
            },
            "Dict[str,bool]": {
                "<Your type-str response here>": "<Your type-bool response here>"
            },
            "Dict[int,str]": {
                "<Your type-int response here>": "<Your type-str response here>"
            },
            "Dict[int,int]": {
                "<Your type-int response here>": "<Your type-int response here>"
            },
            "Dict[int,float]": {
                "<Your type-int response here>": "<Your type-float response here>"
            },
            "Dict[int,bool]": {
                "<Your type-int response here>": "<Your type-bool response here>"
            },
        }

        for key in type_hints:
            if type_hints[key] not in example_dict.keys():
                raise ValueError(f"Type {type_hints[key]} not supported. key- {key}")
            stringified += f"{key}: {example_dict[type_hints[key]]}\n"
        system_prompt += f"""\n\n
Please deliver your response in the following json format:
```json
{{
{stringified}
}}
```
"""
    return system_prompt, user_prompt


def add_json_instructions_to_messages(messages: List[Dict[str,str]], response_model: Optional[Type[BaseModel]] = None) -> List[Dict[str,str]]:
    prev_system_message_content = messages[0]["content"]
    prev_user_message_content = messages[1]["content"]
    system_prompt, user_prompt = add_json_instructions_to_prompt(prev_system_message_content, prev_user_message_content, response_model)
    messages[0]["content"] = system_prompt
    messages[1]["content"] = user_prompt
    return messages



async def extract_pydantic_model_from_response(
    response_raw: str, response_model: Type[BaseModel]
) -> BaseModel:
    if "```" in response_raw and "json" in response_raw:
        response_prepared = response_raw.split("```json")[1].split("```")[0].strip()
    elif "```" in response_raw:
        response_prepared = response_raw.split("```")[1].strip()
    else:
        response_prepared = response_raw.strip()
    # TODO: review???? seems dangerous
    response_prepared = response_prepared.replace("null", '"None"')
    try:
        response = json.loads(response_prepared)
        final = response_model(**response)
    except Exception as e:
        try:
            response = ast.literal_eval(response_prepared)
            final = response_model(**response)
        except Exception as e:
            logger.warning(
                f"Groq reformatter Activated"
            )  # Failed to parse response: {response_prepared} - t
            from termcolor import colored

            print(colored("* invoking groq formatter *", "yellow"))
            final = await groq_json_debugger(response_prepared)
            final = response_model(**final)
    return final