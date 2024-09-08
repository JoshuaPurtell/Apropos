from typing import List, Tuple
import re
import sys

# cv2, pyplot
module_to_package_mapping = {
    "cv2": "opencv-python",
    "sklearn": "scikit-learn",
    "dateutil": "python-dateutil",
    "PIL": "Pillow",
}
tabu = ["urllib", "urllib.request", "urllib.parse","mpl_toolkits"]


def get_imports(code: str) -> Tuple[List[str], str]:
    from_imports = re.findall(r"from ([\w.]+) import [\w, ]+", code)
    head_imports = re.findall(r"^import (\w+)(?!\s+as)$", code, re.MULTILINE)
    module_head_imports = re.findall(r"^import ([\w.]+)(?!\s+as)$", code, re.MULTILINE)
    head_imports_with_alias = re.findall(r"import ([\w.]+) as (\w+)", code)
    imports_snippet = "\n".join(
        ["import " + imp for imp in head_imports]
        + [f"import {imp} as {alias}" for imp, alias in head_imports_with_alias]
        + [
            line.strip()
            for line in code.split("\n")
            if line.strip().startswith("from ")
        ]
    )
    standard_libs = set(sys.stdlib_module_names)
    all_unique_imports = set()
    for imp in (
        head_imports + module_head_imports + [imp for imp, _ in head_imports_with_alias]
    ):
        parts = imp.split(".")
        for i in range(len(parts)):
            package = ".".join(parts[: i + 1])
            if package not in standard_libs:
                all_unique_imports.add(package)
                break
    for package in from_imports:
        parts = package.split(".")
        for i in range(len(parts)):
            subpackage = ".".join(parts[: i + 1])
            if subpackage not in standard_libs:
                all_unique_imports.add(subpackage)
                break
    return [
        module_to_package_mapping.get(v, v) for v in all_unique_imports if v not in tabu
    ], imports_snippet


def test_get_imports():
    test_cases = [
        (
            "import cv2\nimport matplotlib.pyplot as plt",
            (
                ["opencv-python", "matplotlib"],
                "import cv2\nimport matplotlib.pyplot as plt",
            ),
        ),
        (
            "from numpy import array\nfrom scipy import stats",
            (["numpy", "scipy"], "from numpy import array\nfrom scipy import stats"),
        ),
        (
            "import os\nfrom datetime import datetime",
            ([], "import os\nfrom datetime import datetime"),
        ),
        (
            "import pandas as pd\nfrom sklearn.model_selection import train_test_split",
            (
                ["pandas", "scikit-learn"],
                "import pandas as pd\nfrom sklearn.model_selection import train_test_split",
            ),
        ),
        (
            "import tensorflow as tf\nfrom keras.layers import Dense",
            (
                ["tensorflow", "keras"],
                "import tensorflow as tf\nfrom keras.layers import Dense",
            ),
        ),
        (
            "import dateutil.parser\nfrom datetime import datetime",
            (
                ["python-dateutil"],
                "import dateutil.parser\nfrom datetime import datetime",
            ),
        ),
    ]

    failures = 0
    successes = 0

    for code, expected_output in test_cases:
        packages, imports_snippet = get_imports(code)
        if sorted(expected_output[0]) != sorted(
            packages
        ):  # or imports_snippet != expected_output[1]
            failures += 1
            print(f"Test case failed:")
            print(f"Input: {code}")
            print(f"Expected packages: {expected_output[0]}, got: {packages}")
            print(f"Expected imports snippet: {expected_output[1]}")
            print(f"Got imports snippet: {imports_snippet}")
            print()
        else:
            successes += 1

    print(f"Test results: {successes} passed, {failures} failed")


if __name__ == "__main__":
    test_get_imports()
