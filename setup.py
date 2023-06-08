import os
import subprocess
import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent

venv_dir = script_dir / "venv"
print(sys.executable)

if not os.path.exists(venv_dir):
    subprocess.check_call([sys.executable, "-m", "venv", venv_dir])

if os.name == "nt":  # Для Windows
    activate_script = venv_dir / "Scripts" / "activate.bat"
    pip_executable = venv_dir / "Scripts" / "pip.exe"
else:  # Для Linux/Mac
    activate_script = venv_dir / "bin" / "activate"
    pip_executable = venv_dir / "bin" / "pip"

subprocess.check_call(f"call {activate_script} && {pip_executable} install -r requirements.txt", shell=True)
