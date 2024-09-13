from apropos.src.core.lms.vendors.openai_api import OpenAIAPIProvider
from pydantic import BaseModel, Field
from typing import Dict, Any

# from apropos.src.core.lms.llm import LLM
import asyncio


async def test_response_model_react():
    class ReactResponse(BaseModel):
        reasoning: str
        action: str
        action_args: Dict[str, Any]

    messages = [
        {
            "role": "system",
            "content": """# Premise
You are a software engineer
Here is some information about this setting
<Setting Information>
You are working to solve a computer science problem. You will need to submit a solution to the problem, which will be tested against a suite of hidden unit tests.
</Setting Information>
<Actions Available>
<edit_submission>
<action_context>
Edit the submission code. Use this when you want to make changes to the current solution.
</action_context>
<action_arg_spec>
{'first_line': <class 'int'>, 'last_line': <class 'int'>, 'new_code': <class 'str'>}
</action_arg_spec>
<action_description>
Edit the submission code
</action_description>

</edit_submission>
<add_submission>
<action_context>
Add the submission code. Use this when you want to start from scratch with a new solution.
</action_context>
<action_arg_spec>
{'submission': <class 'str'>}
</action_arg_spec>
<action_description>
Add the submission code
</action_description>

</add_submission>
<add_unit_test>
<action_context>
Add a unit test. The unit test information you submit must be in the format of a BCBUnitTest: 

class BCBUnitTest(BaseModel):
    test_description: str
    input_names: List[str]
    input_types: List[str]
    input_values: List[Any]
    assertion_condition: str
    assertion_type: Literal["assertTrue", "assertRaises"] = "assertTrue"


 It will be parsed via BCBUnitTest(**unit_test_dict)



# Some various notes:
1. If an input should be of a type defined by a specific package, add the package name/alias to the type. E.g. "np.ndarray" or "pd.DataFrame". You still should fully define the value for the input_value field e.g. "pd.DataFrame({'a': [1, 2, 3]})"

2. Unit tests will be compiled from the BCBUnitTest class as follows:
    A. For AssertTrue type tests, the test will be compiled as follows:
    ```python
    def test_case(self):
        # {{self.test_description}}

        {{defs}}
        result = {{function_name}}(**{{{{args}}}}})
        self.{{self.assertion_type}}({{self.assertion_condition}})
    ```
    B. For AssertRaises type tests, the test will be compiled as follows:

    ```python
    def test_case(self):
        # {{self.test_description}}
        {{defs}}
        with self.{{self.assertion_type}}({{self.assertion_condition}}):
            {{function_name}}(**{{{{args}}}}})
    ```

    Provide information accordingly.

</action_context>
<action_arg_spec>
{'unit_test_name': <class 'str'>, 'unit_test_dict': typing.Dict}
</action_arg_spec>
<action_description>
Add a unit test
</action_description>

</add_unit_test>
<remove_unit_test>
<action_context>
Remove a unit test
</action_context>
<action_arg_spec>
{'unit_test_name': <class 'str'>}
</action_arg_spec>
<action_description>
Remove a unit test
</action_description>

</remove_unit_test>
<test_submission>
<action_context>
Test the submission
</action_context>
<action_arg_spec>
{}
</action_arg_spec>
<action_description>
Test the submission
</action_description>

</test_submission>
<submit_solution>
<action_context>
Submit the solution
</action_context>
<action_arg_spec>
{}
</action_arg_spec>
<action_description>
Submit the solution
</action_description>

</submit_solution>

</Actions Available>
You'll be given your past actions/thoughts, along with recent raw observations from the environment
The environment one step in the past is your current environment.

# Objective
Please complete the problem by drafting a solution, creating unit tests, improving the solution, and submitting the solution.

# Constraints
You will be given a code_prompt_for_answer, which contains imports and the function signature. Your solution must comprise code that can be appended to code_prompt_for_answer and run as a single script.

    """,
        },
        {
            "role": "user",
            "content": '''# Recent Actions / Thoughts
<1 reasoning step(s) in the past>reasoning='I need to add more unit tests to cover additional scenarios such as invalid IP addresses and malformed JSON strings. This will ensure that the function behaves correctly under various conditions.' action='add_unit_test' action_args={'unit_test_name': 'test_invalid_ip', 'unit_test_dict': {'test_description': 'Test with an invalid IP address in JSON format.', 'input_names': ['ip_address'], 'input_types': ['str'], 'input_values': ['{"ip": "999.999.999.999"}'], 'assertion_condition': "result == 'Invalid IP address received'", 'assertion_type': 'assertTrue'}}</1 reasoning step(s) in the past>
<2 reasoning step(s) in the past>reasoning='I need to create unit tests to validate the functionality of the task_func. The tests should cover valid IP addresses, invalid IP addresses, and malformed JSON strings to ensure the function behaves as expected in different scenarios.' action='add_unit_test' action_args={'unit_test_name': 'test_valid_ip', 'unit_test_dict': {'test_description': 'Test with a valid IP address in JSON format.', 'input_names': ['ip_address'], 'input_types': ['str'], 'input_values': ['{"ip": "192.168.1.1"}'], 'assertion_condition': "result == '192.168.1.1'", 'assertion_type': 'assertTrue'}}</2 reasoning step(s) in the past>
<3 reasoning step(s) in the past>reasoning='The function needs to extract the IP address from a JSON string and validate it against a regex pattern. If the IP address is valid, it should return the IP address; otherwise, it should return an error message. I will implement this logic in the function.' action='add_submission' action_args={'submission': 'import re\nfrom urllib import request\nimport json\n# Constants\nIP_REGEX = r\'[0-9]+(?:\\.[0-9]+){3}\'\n\ndef task_func(ip_address):\n    """\n    Get the public IP address from a JSON response containing the IP address.\n    \n    Parameters:\n    ip_address (str): JSON-formatted string containing the IP address. \n\n    Returns:\n    str: The public IP address.\n    \n    Note:\n    - The function needs to check whether the provided IP address is valid.\n      If the IP address is not valid, the function will return \'Invalid IP address received\'.\n\n    Requirements:\n    - re\n    - urllib.request\n    - json\n    \n    Example:\n    >>> ip_address = \'{"ip": "192.168.1.1"}\'\n    >>> task_func(ip_address)\n    \'192.168.1.1\'\n    """\n    try:\n        data = json.loads(ip_address)\n        ip = data.get(\'ip\')\n        if re.match(IP_REGEX, ip):\n            return ip\n        else:\n            return \'Invalid IP address received\'\n    except (json.JSONDecodeError, TypeError):\n        return \'Invalid IP address received\''}</3 reasoning step(s) in the past>
# Recent Observations
<1 environment step(s) in the past>success=True result='Added unit test successfully'</1 environment step(s) in the past>
<2 environment step(s) in the past>success=True result='Added unit test successfully'</2 environment step(s) in the past>
<3 environment step(s) in the past>success=True result='Added submission successfully'</3 environment step(s) in the past>
<4 environment step(s) in the past>{'action_result': None, 'environment_state': {'question': 'import re\nfrom urllib import request\nimport json\n\n# Constants\nIP_REGEX = r\'[0-9]+(?:\\.[0-9]+){3}\'\n\ndef task_func(ip_address):\n    """\n    Get the public IP address from a JSON response containing the IP address.\n    \n    Parameters:\n    ip_address (str): JSON-formatted string containing the IP address. \n\n    Returns:\n    str: The public IP address.\n    \n    Note:\n    - The function needs to check whether the provided IP address is valid.\n      If the IP address is not valid, the function will return \'Invalid IP address received\'.\n\n    Requirements:\n    - re\n    - urllib.request\n    - json\n    \n    Example:\n    >>> ip_address = \'{"ip": "192.168.1.1"}\'\n    >>> task_func(ip_address)\n    \'192.168.1.1\'\n    """\n', 'code_prompt_for_answer': "import re\nfrom urllib import request\nimport json\n# Constants\nIP_REGEX = r'[0-9]+(?:\\.[0-9]+){3}'\ndef task_func(ip_address):\n", 'unit_tests_you_have_written': {'test_valid_ip': BCBUnitTest(test_description='Test with a valid IP address in JSON format.', input_names=['ip_address'], input_types=['str'], input_values=['{"ip": "192.168.1.1"}'], assertion_condition="result == '192.168.1.1'", assertion_type='assertTrue'), 'test_invalid_ip': BCBUnitTest(test_description='Test with an invalid IP address in JSON format.', input_names=['ip_address'], input_types=['str'], input_values=['{"ip": "999.999.999.999"}'], assertion_condition="result == 'Invalid IP address received'", assertion_type='assertTrue')}, 'current_solution': ''}}</4 environment step(s) in the past>

Your response:''',
        },
    ]
    react_step = await OpenAIAPIProvider().async_chat_completion_with_response_model(
        messages=messages,
        model="gpt-4o-mini-2024-07-18",
        temperature=0,
        max_tokens=1024,
        response_model=ReactResponse,
    )
    print(react_step)


async def test2_response_model_react():
    class ReactResponse(BaseModel):
        reasoning: str
        action: str
        action_args: Dict[str, Any]

    messages = [
        {
            "role": "system",
            "content": """# Premise
    You are a software engineer
    Here is some information about this setting
    <Setting Information>
    You are working to solve a computer science problem. You will need to submit a solution to the problem, which will be tested against a suite of hidden unit tests.
    </Setting Information>
    <Actions Available>
    <edit_submission>
    <action_context>
    Edit the submission code. Use this when you want to make changes to the current solution.
    </action_context>
    <action_arg_spec>
    {'first_line': <class 'int'>, 'last_line': <class 'int'>, 'new_code': <class 'str'>}
    </action_arg_spec>
    <action_description>
    Edit the submission code
    </action_description>

    </edit_submission>
    <add_submission>
    <action_context>
    Add the submission code. Use this when you want to start from scratch with a new solution.
    </action_context>
    <action_arg_spec>
    {'submission': <class 'str'>}
    </action_arg_spec>
    <action_description>
    Add the submission code
    </action_description>

    </add_submission>
    <add_unit_test>
    <action_context>
    Add a unit test. The unit test information you submit must be in the format of a BCBUnitTest: 

    class BCBUnitTest(BaseModel):
        test_description: str
        input_names: List[str]
        input_types: List[str]
        input_values: List[Any]
        assertion_condition: str
        assertion_type: Literal["assertTrue", "assertRaises"] = "assertTrue"


    It will be parsed via BCBUnitTest(**unit_test_dict)



    # Some various notes:
    1. If an input should be of a type defined by a specific package, add the package name/alias to the type. E.g. "np.ndarray" or "pd.DataFrame". You still should fully define the value for the input_value field e.g. "pd.DataFrame({'a': [1, 2, 3]})"

    2. Unit tests will be compiled from the BCBUnitTest class as follows:
        A. For AssertTrue type tests, the test will be compiled as follows:
        ```python
        def test_case(self):
            # {{self.test_description}}

            {{defs}}
            result = {{function_name}}(**{{{{args}}}}})
            self.{{self.assertion_type}}({{self.assertion_condition}})
        ```
        B. For AssertRaises type tests, the test will be compiled as follows:

        ```python
        def test_case(self):
            # {{self.test_description}}
            {{defs}}
            with self.{{self.assertion_type}}({{self.assertion_condition}}):
                {{function_name}}(**{{{{args}}}}})
        ```

        Provide information accordingly.

    </action_context>
    <action_arg_spec>
    {'unit_test_name': <class 'str'>, 'unit_test_dict': typing.Dict}
    </action_arg_spec>
    <action_description>
    Add a unit test
    </action_description>

    </add_unit_test>
    <remove_unit_test>
    <action_context>
    Remove a unit test
    </action_context>
    <action_arg_spec>
    {'unit_test_name': <class 'str'>}
    </action_arg_spec>
    <action_description>
    Remove a unit test
    </action_description>

    </remove_unit_test>
    <test_submission>
    <action_context>
    Test the submission
    </action_context>
    <action_arg_spec>
    {}
    </action_arg_spec>
    <action_description>
    Test the submission
    </action_description>

    </test_submission>
    <submit_solution>
    <action_context>
    Submit the solution
    </action_context>
    <action_arg_spec>
    {}
    </action_arg_spec>
    <action_description>
    Submit the solution
    </action_description>

    </submit_solution>

    </Actions Available>
    You'll be given your past actions/thoughts, along with recent raw observations from the environment
    The environment one step in the past is your current environment.

    # Objective
    Please complete the problem by drafting a solution, creating unit tests, improving the solution, and submitting the solution.

    # Constraints
    You will be given a code_prompt_for_answer, which contains imports and the function signature. Your solution must comprise code that can be appended to code_prompt_for_answer and run as a single script.
    """,
        },
        {
            "role": "user",
            "content": '''# Recent Actions / Thoughts
    <1 reasoning step(s) in the past>reasoning='I need to add more unit tests to cover additional scenarios such as invalid IP addresses and malformed JSON strings. This will ensure that the function behaves correctly under various conditions.' action='add_unit_test' action_args={'unit_test_name': 'test_invalid_ip', 'unit_test_dict': {'test_description': 'Test with an invalid IP address in JSON format.', 'input_names': ['ip_address'], 'input_types': ['str'], 'input_values': ['{"ip": "999.999.999.999"}'], 'assertion_condition': "result == 'Invalid IP address received'", 'assertion_type': 'assertTrue'}}</1 reasoning step(s) in the past>
    <2 reasoning step(s) in the past>reasoning='I need to create unit tests to validate the functionality of the task_func. The tests should cover valid IP addresses, invalid IP addresses, and malformed JSON strings to ensure the function behaves as expected in different scenarios.' action='add_unit_test' action_args={'unit_test_name': 'test_valid_ip', 'unit_test_dict': {'test_description': 'Test with a valid IP address in JSON format.', 'input_names': ['ip_address'], 'input_types': ['str'], 'input_values': ['{"ip": "192.168.1.1"}'], 'assertion_condition': "result == '192.168.1.1'", 'assertion_type': 'assertTrue'}}</2 reasoning step(s) in the past>
    <3 reasoning step(s) in the past>reasoning='The function needs to extract the IP address from a JSON string and validate it against a regex pattern. If the IP address is valid, it should return the IP address; otherwise, it should return an error message. I will implement this logic in the function.' action='add_submission' action_args={'submission': 'import re\nfrom urllib import request\nimport json\n# Constants\nIP_REGEX = r\'[0-9]+(?:\\.[0-9]+){3}\'\n\ndef task_func(ip_address):\n    """\n    Get the public IP address from a JSON response containing the IP address.\n    \n    Parameters:\n    ip_address (str): JSON-formatted string containing the IP address. \n\n    Returns:\n    str: The public IP address.\n    \n    Note:\n    - The function needs to check whether the provided IP address is valid.\n      If the IP address is not valid, the function will return \'Invalid IP address received\'.\n\n    Requirements:\n    - re\n    - urllib.request\n    - json\n    \n    Example:\n    >>> ip_address = \'{"ip": "192.168.1.1"}\'\n    >>> task_func(ip_address)\n    \'192.168.1.1\'\n    """\n    try:\n        data = json.loads(ip_address)\n        ip = data.get(\'ip\')\n        if re.match(IP_REGEX, ip):\n            return ip\n        else:\n            return \'Invalid IP address received\'\n    except (json.JSONDecodeError, TypeError):\n        return \'Invalid IP address received\''}</3 reasoning step(s) in the past>
    # Recent Observations
    <1 environment step(s) in the past>success=True result='Added unit test successfully'</1 environment step(s) in the past>
    <2 environment step(s) in the past>success=True result='Added unit test successfully'</2 environment step(s) in the past>
    <3 environment step(s) in the past>success=True result='Added submission successfully'</3 environment step(s) in the past>
    <4 environment step(s) in the past>{'action_result': None, 'environment_state': {'question': 'import re\nfrom urllib import request\nimport json\n\n# Constants\nIP_REGEX = r\'[0-9]+(?:\\.[0-9]+){3}\'\n\ndef task_func(ip_address):\n    """\n    Get the public IP address from a JSON response containing the IP address.\n    \n    Parameters:\n    ip_address (str): JSON-formatted string containing the IP address. \n\n    Returns:\n    str: The public IP address.\n    \n    Note:\n    - The function needs to check whether the provided IP address is valid.\n      If the IP address is not valid, the function will return \'Invalid IP address received\'.\n\n    Requirements:\n    - re\n    - urllib.request\n    - json\n    \n    Example:\n    >>> ip_address = \'{"ip": "192.168.1.1"}\'\n    >>> task_func(ip_address)\n    \'192.168.1.1\'\n    """\n', 'code_prompt_for_answer': "import re\nfrom urllib import request\nimport json\n# Constants\nIP_REGEX = r'[0-9]+(?:\\.[0-9]+){3}'\ndef task_func(ip_address):\n", 'unit_tests_you_have_written': {'test_valid_ip': BCBUnitTest(test_description='Test with a valid IP address in JSON format.', input_names=['ip_address'], input_types=['str'], input_values=['{"ip": "192.168.1.1"}'], assertion_condition="result == '192.168.1.1'", assertion_type='assertTrue'), 'test_invalid_ip': BCBUnitTest(test_description='Test with an invalid IP address in JSON format.', input_names=['ip_address'], input_types=['str'], input_values=['{"ip": "999.999.999.999"}'], assertion_condition="result == 'Invalid IP address received'", assertion_type='assertTrue')}, 'current_solution': ''}}</4 environment step(s) in the past>''',
        },
    ]
    react_step = await OpenAIAPIProvider().async_chat_completion_with_response_model(
        messages=messages,
        model="gpt-4o-mini-2024-07-18",
        temperature=0,
        max_tokens=1024,
        response_model=ReactResponse,
    )
    print(react_step)


async def run_tests():
    await test_response_model_react()
    await test2_response_model_react()


if __name__ == "__main__":
    asyncio.run(run_tests())
