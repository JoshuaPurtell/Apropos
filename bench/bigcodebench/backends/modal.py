import io

import modal
from modal import App, Sandbox
from modal.exception import SandboxTimeoutError

execution_image = modal.Image.from_registry("python:3.9-slim").pip_install(
    "pandas==1.3.3",
    "fuzzywuzzy==0.18.0",
    "openai==1.3.2",
    "ruptures==1.1.8",
    "tiktoken==0.4.0",
    "matplotlib==3.8.1",
    "tabulate==0.9.0",
    "PyYAML",
)
app = App("bigcodebench")

async def execute_code_remotely(question_dict, code_solution):
    solution = question_dict["eval_info"]['code_prompt'] + code_solution
    test = question_dict["eval_info"]['test']
    complete = solution + "\n" + test

    unit_test_script = """
import unittest
import io
def test_code():
    path = "script.py"
    loader = unittest.TestLoader()
    suite = loader.discover('.', pattern=path)
    runner = unittest.TextTestRunner()
    result = runner.run(suite)

    # Convert result to a serializable format
    result_dict = {
        "errors": len(result.errors),
        "failures": len(result.failures),
        "testsRun": result.testsRun,
        "wasSuccessful": result.wasSuccessful()
    }
    return result.wasSuccessful(), result_dict

if __name__ == "__main__":
    success,result = test_code()
    print("Success:", success)
    print(result)
"""
    try:
        with modal.NetworkFileSystem.ephemeral() as nfs:
            await nfs.write_file.aio("script.py", io.BytesIO(complete.encode()))
            await nfs.write_file.aio("unit_test_script.py", io.BytesIO(unit_test_script.encode()))

            sb = Sandbox.create(
                "bash",
                "-c",
                f"cd /vol && python -W ignore unit_test_script.py",
                image=execution_image,
                timeout=60,
                cloud="aws",
                network_file_systems={"/vol": nfs},
            )
            await sb.wait.aio()
            stdout = await sb.stdout.read.aio()
            stderr = await sb.stderr.read.aio()
            success = stdout.split("Success: ")[1].split("\n")[0] == "True"
            result_dict = eval(stdout.split("Success: ")[1].split("\n")[1])
            return success, result_dict
    except SandboxTimeoutError:
        return False, {
            "errors": 1,
            "failures": 0,
            "testsRun": 0,
            "wasSuccessful": False,
            "timeout": True
        }