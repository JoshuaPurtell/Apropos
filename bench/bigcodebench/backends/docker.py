import io
import asyncio
import docker
import tempfile
import os

async def execute_code_remotely_docker(question_dict, code_solution):
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

    result_dict = {
        "errors": len(result.errors),
        "failures": len(result.failures),
        "testsRun": result.testsRun,
        "wasSuccessful": result.wasSuccessful()
    }
    return result.wasSuccessful(), result_dict

if __name__ == "__main__":
    success, result = test_code()
    print("Success:", success)
    print(result)
"""

    client = docker.from_env()

    with tempfile.TemporaryDirectory() as temp_dir:
        script_path = os.path.join(temp_dir, "script.py")
        test_script_path = os.path.join(temp_dir, "unit_test_script.py")

        with open(script_path, "w") as f:
            f.write(complete)
        with open(test_script_path, "w") as f:
            f.write(unit_test_script)

        container = client.containers.run(
            "python:3.9-slim",
            command=f"python -W ignore /app/unit_test_script.py",
            volumes={temp_dir: {'bind': '/app', 'mode': 'ro'}},
            detach=True
        )

        try:
            container.wait(timeout=60)
            logs = container.logs().decode('utf-8')
        finally:
            container.remove()

    try:
        success = logs.split("Success: ")[1].split("\n")[0] == "True"
        result_dict = eval(logs.split("Success: ")[1].split("\n")[1])
        return success, result_dict
    except Exception as e:
        return False, {"errors": 1, "failures": 0, "testsRun": 0, "wasSuccessful": False}
