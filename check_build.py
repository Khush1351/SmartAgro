import os
import py_compile

required_paths = [
    "app.py",
    "templates",
    "static",
    "utils",
    "requirements.txt"
]

for path in required_paths:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file or folder: {path}")

# Check Python syntax
py_compile.compile("app.py", doraise=True)

print("Build checks passed successfully.")