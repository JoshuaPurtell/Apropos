import io
import os
import random
import unittest
from typing import Dict
## I really wouldn't do this! It deleted a bunch of my code


def execute_code_locally(question_dict: Dict, code_solution: str):
    random_integer = random.randint(0, 10000)

    solution = question_dict["eval_info"]["code_prompt"] + code_solution
    test = question_dict["eval_info"]["test"]
    complete = (
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        "import matplotlib.pyplot as plt\n"
        f"{solution}\n"
        f"{test}\n"
        "plt.close('all')"
    )

    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    path = f"{temp_dir}/temp_test_script_{random_integer}.py"
    with open(path, "w") as f:
        f.write(complete)

    loader = unittest.TestLoader()
    suite = loader.discover(temp_dir, pattern=f"temp_test_script_{random_integer}.py")
    runner = unittest.TextTestRunner(stream=io.StringIO())

    with open(os.devnull, "w") as devnull:
        with unittest.mock.patch("sys.stdout", devnull), unittest.mock.patch(
            "sys.stderr", devnull
        ):
            result = runner.run(suite)

    result_dict = {
        "errors": len(result.errors),
        "failures": len(result.failures),
        "testsRun": result.testsRun,
        "wasSuccessful": result.wasSuccessful(),
    }
    os.remove(path)
    return result.wasSuccessful() and result.testsRun > 0, result_dict
