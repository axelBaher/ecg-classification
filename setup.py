import os
import subprocess
import sys
from pathlib import Path
import argparse

script_dir = Path(__file__).resolve().parent
venv_dir = script_dir / "venv"
print(sys.executable)


def create_virtual_environment():
    if not os.path.exists(venv_dir):
        subprocess.check_call([sys.executable, "-m", "venv", venv_dir])

    if os.name == "nt":  # Win
        activate_script = venv_dir / "Scripts" / "activate.bat"
        pip_executable = venv_dir / "Scripts" / "pip.exe"
    else:  # Unix
        activate_script = venv_dir / "bin" / "activate"
        pip_executable = venv_dir / "bin" / "pip"

    subprocess.check_call(f"call {activate_script} && {pip_executable} install -r requirements.txt", shell=True)


def install_system_packages():
    pip_executable = sys.executable
    subprocess.check_call(f"{pip_executable} install -r requirements.txt", shell=True)


def main():
    parser = argparse.ArgumentParser(description="Packages installing")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-v", "--virtualenv", action="store_true",
                       help="Install packages into virtual environment")
    group.add_argument("-s", "--system", action="store_true", help="Install packages into default path")

    args = parser.parse_args()

    if args.virtualenv:
        create_virtual_environment()
    elif args.system:
        install_system_packages()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
