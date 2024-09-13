from typing import List, Tuple
import re
import sys

# cv2, pyplot
module_to_package_mapping = {
    "cv2": "opencv-python",
    "sklearn": "scikit-learn",
    "dateutil": "python-dateutil",
    "PIL": "Pillow",
    "faker": "Faker",
    "requests_mock": "requests-mock",
}
tabu = [
    "urllib",
    "urllib.request",
    "urllib.parse",
    "mpl_toolkits",
    "unittest.mock",
    "urllib.error",
    "pyplot",
    "http.server",
    "task_func",
]
# pyplot and task_func being here is suspicious


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


# pytz


def get_linux_import_snippets(full_code: str, imports: List[str]) -> List[str]:
    opengl = [
        "apt-get update && apt-get install -y libgl1-mesa-glx",
        "apt-get update && apt-get install -y libglib2.0-0",
    ]
    gcc = ["apt-get update && apt-get install -y gcc"]
    wordcloud = ["apt-get install -y python3-dev libfreetype6-dev libpng-dev"]
    fonts = [
        "apt-get update && apt-get install -y fontconfig",
        "mkdir -p ~/.fonts",
        "wget https://github.com/google/fonts/raw/main/apache/roboto/Roboto-Regular.ttf -O ~/.fonts/Roboto-Regular.ttf",
        "fc-cache -f -v",
    ]
    fonts = []  # might not be necessary?
    stopwords = [
        "pip install --no-cache-dir nltk",
        "python -c 'import nltk; nltk.download(\"stopwords\")'",
    ]
    linux_imports = []
    opengl_packages = ["Pillow", "opencv-python"]
    gcc_packages = [
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "cython",
        "torch",
        "tensorflow",
        "scikit-learn",
        "nltk",
        "Pillow",
        "opencv-python",
        "psycopg2",
        "mysqlclient",
        "lxml",
        "biopython",
    ]
    for import_ in imports:
        if import_ in opengl_packages:  # cv2, PIL
            linux_imports.extend(opengl)
        if import_ in gcc_packages:
            linux_imports.extend(gcc)
    if "stopwords.words" in full_code:
        linux_imports.extend(stopwords)
    if "='Arial'" in full_code:
        linux_imports.extend(fonts)
    if "wordcloud" in full_code:
        linux_imports.extend(wordcloud)
    return linux_imports


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
        if sorted(expected_output[0]) != sorted(packages):
            failures += 1
        else:
            successes += 1

    print(f"Test results: {successes} passed, {failures} failed")


if __name__ == "__main__":
    test_get_imports()
