import sys
import os

print("--- Python Interpreter Check ---")
print(f"Executable Path: {sys.executable}")
print("-" * 30)

venv_path = os.path.abspath('.venv')

if sys.executable.startswith(venv_path):
    print("✅ SUCCESS: The interpreter is running from the correct virtual environment.")
else:
    print("❌ ERROR: The interpreter is NOT running from the virtual environment.")
    print(f"Your project's venv is at: {venv_path}")
    print("Please use 'Ctrl+Shift+P' -> 'Python: Select Interpreter' in VS Code to fix this.")

print("--------------------------------")