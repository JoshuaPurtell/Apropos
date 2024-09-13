import asyncio
import os
import shutil
import tempfile
from typing import Any, Dict, Optional, Tuple

import aiodocker
import docker
from docker.models.containers import Container

from apropos.src.bench.bigcodebench.backends.shared import (
    get_imports,
    get_linux_import_snippets,
)


def get_unit_test_script(path_to_script: str) -> str:
    return f"""
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


def execute_code_remotely_docker_sync(
    question_dict: Dict[str, Any],
    code_solution: str,
    container: Optional[Container] = None,
) -> Tuple[bool, Dict[str, Any], Optional[Container]]:
    solution = question_dict["eval_info"]["code_prompt"] + code_solution

    test = question_dict["eval_info"]["test"]
    complete = solution + "\n" + test
    packages, _ = get_imports(complete)

    client = docker.from_env()
    temp_dir = tempfile.mkdtemp()

    try:
        script_path = os.path.join(temp_dir, "script.py")
        test_script_path = os.path.join(temp_dir, "unit_test_script.py")
        with open(script_path, "w") as f:
            f.write(complete)
        with open(test_script_path, "w") as f:
            f.write(get_unit_test_script("script.py"))

        if container is None:
            container = client.containers.run(
                "python:3.9-slim",
                command="/bin/bash",
                volumes={temp_dir: {"bind": "/app", "mode": "rw"}},
                working_dir="/app",
                detach=True,
                tty=True,
            )

            linux_imports = get_linux_import_snippets(complete, packages)
            setup_commands = [
                "pip install uv",
                "uv init",
                "uv venv",
            ]
            if packages:
                for package in packages:
                    setup_commands.append(f"uv pip install {package}")
            for linux_import in linux_imports:
                exit_code, output = container.exec_run(
                    f"sh -c '{linux_import}'", user="root"
                )  # .exec_run(cmd)
                if exit_code != 0:
                    raise Exception(f"Linux import command failed: {cmd}")
            for cmd in setup_commands:
                exit_code, output = container.exec_run(cmd)
                if exit_code != 0:
                    raise Exception(f"Setup command failed: {cmd}")

        exit_code, output = container.exec_run("uv run /app/unit_test_script.py")
        logs = output.decode("utf-8")

        try:
            success = logs.split("Success: ")[1].split("\n")[0] == "True"
            result_dict = eval(logs.split("Success: ")[1].split("\n")[1])
            return success, result_dict, container
        except Exception as e:
            print("Packages: ", packages)
            print(
                "Please report this error to the developers of apropos-ai @ https://github.com/JoshuaPurtell/Apropos"
            )
            print("Logs: ", logs)
            return (
                False,
                {
                    "errors": 1,
                    "failures": 0,
                    "testsRun": 0,
                    "wasSuccessful": False,
                },
                container,
            )
    finally:
        for file in [script_path, test_script_path]:
            try:
                os.remove(file)
            except OSError as e:
                print(f"Error removing {file}: {e}")

        try:
            shutil.rmtree(temp_dir)
        except OSError as e:
            print(f"Error removing directory {temp_dir}: {e}")


if __name__ == "__main__":
    import asyncio

    from apropos.src.bench.bigcodebench.main import BigCodeBenchComplete_Benchmark

    question = BigCodeBenchComplete_Benchmark().train[0]
    print(
        execute_code_remotely_docker_sync(
            question.information, question.information["answer"]
        )
    )
