import asyncio
import os
import tempfile

import docker

from apropos.src.bench.bigcodebench.backends.shared import get_imports


async def execute_code_remotely_docker(
    question_dict, code_solution
):  # , packages: List[str] = []
    solution = question_dict["eval_info"]["code_prompt"] + code_solution
    packages, _ = get_imports(solution)
    test = question_dict["eval_info"]["test"]
    complete = solution + "\n" + test

    def get_unit_test_script(path_to_script: str):
        unit_test_script = f"""
import unittest
import io
import os
def test_code():
    path = "{path_to_script}"
    loader = unittest.TestLoader()
    suite = loader.discover('/app', pattern=path)
    runner = unittest.TextTestRunner()
    result = runner.run(suite)

    result_dict = {{
        "errors": len(result.errors),
        "failures": len(result.failures),
        "testsRun": result.testsRun,
        "wasSuccessful": result.wasSuccessful()
    }}

    return result.wasSuccessful(), result_dict
if __name__ == "__main__":
    success, result = test_code()
    print("Success:", success)
    print(result)
"""
        return unit_test_script

    client = docker.from_env()

    with tempfile.TemporaryDirectory() as temp_dir:
        script_path = os.path.join(temp_dir, "script.py")
        test_script_path = os.path.join(temp_dir, "unit_test_script.py")
        with open(script_path, "w") as f:
            f.write(complete)
        with open(test_script_path, "w") as f:
            f.write(get_unit_test_script("script.py"))

        install_command = "apt-get update && apt-get install -y gcc && "
        if packages:
            install_command += f"pip install uv && uv init && uv venv && uv pip install {' '.join(packages)} &&"
        else:
            install_command += "pip install uv && uv init && uv venv &&"
        container = client.containers.run(
            "python:3.9-slim",
            command=f"/bin/bash -c '{install_command} uv run /app/unit_test_script.py'",  # -W ignore
            volumes={temp_dir: {"bind": "/app", "mode": "rw"}},
            working_dir="/app",
            detach=True,
        )

        try:
            container.wait(timeout=60)
            logs = container.logs().decode("utf-8")
        finally:
            container.remove()
    try:
        success = logs.split("Success: ")[1].split("\n")[0] == "True"
        result_dict = eval(logs.split("Success: ")[1].split("\n")[1])
        return success, result_dict
    except Exception as e:
        print("Packages: ", packages)
        print("Please report this error to the developers of apropos-ai @ https://github.com/JoshuaPurtell/Apropos")
        print("Logs: ", logs)
        return False, {
            "errors": 1,
            "failures": 0,
            "testsRun": 0,
            "wasSuccessful": False,
        }


if __name__ == "__main__":
    import asyncio
    from apropos.src.bench.bigcodebench.main import BigCodeBenchComplete_Benchmark

    question = BigCodeBenchComplete_Benchmark().train[0]
    print(
        asyncio.run(
            execute_code_remotely_docker(
                question.information, question.information["answer"], ["pandas"]
            )
        )
    )
