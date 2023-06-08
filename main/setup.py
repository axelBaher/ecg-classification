import os
import subprocess


venv_dir = "../venv-test"
python_executable = "python"

subprocess.check_call([python_executable, "-m", "venv", venv_dir])

if os.name == "nt":
    activate_script = os.path.join(venv_dir, "Scripts", "activate.bat")
    activate_cmd = f"call {activate_script}"
else:
    activate_script = os.path.join(venv_dir, "bin", "activate")
    activate_cmd = f"source {activate_script}"

subprocess.check_call(activate_cmd, shell=True)

requirements_file = "../requirements.txt"

subprocess.check_call([python_executable, "-m", "pip", "install", "-r", requirements_file])

print("Done!")
