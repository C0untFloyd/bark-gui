import argparse
import glob
import os
import shutil
import site
import subprocess
import sys

script_dir = os.getcwd()


def run_cmd(cmd, capture_output=False, env=None):
    # Run shell commands
    return subprocess.run(cmd, shell=True, capture_output=capture_output, env=env)


def check_env():
    # If we have access to conda, we are probably in an environment
    conda_not_exist = run_cmd("conda", capture_output=True).returncode
    if conda_not_exist:
        print("Conda is not installed. Exiting...")
        sys.exit()
    
    # Ensure this is a new environment and not the base environment
    if os.environ["CONDA_DEFAULT_ENV"] == "base":
        print("Create an environment for this project and activate it. Exiting...")
        sys.exit()


def install_dependencies():
    # Select your GPU or, choose to run in CPU mode
    print("Do you have a GPU (Nvidia)?")
    print("Enter Y for Yes")
    print()
    gpuchoice = input("Input> ").lower()

    # Clone webui to our computer
    run_cmd("git clone https://github.com/C0untFloyd/bark-gui.git")
    if gpuchoice == "y":
        run_cmd("pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

    run_cmd("pip install IPython")
    run_cmd("pip install soundfile")
    run_cmd("pip install gradio")
    # Install the webui dependencies
    update_dependencies()


def update_dependencies():
    os.chdir("bark-gui")
    run_cmd("git pull")
    # Installs/Updates dependencies from all requirements.txt
    run_cmd("python -m pip install .")
    

def start_app():
    os.chdir("bark-gui")
    run_cmd('python webui.py -autolaunch')


if __name__ == "__main__":
    # Verifies we are in a conda environment
    check_env()

    parser = argparse.ArgumentParser()
    parser.add_argument('--update', action='store_true', help='Update the web UI.')
    args = parser.parse_args()

    if args.update:
        update_dependencies()
    else:
        # If webui has already been installed, skip and run
        if not os.path.exists("bark-gui/"):
            install_dependencies()
            os.chdir(script_dir)

        # Run the model with webui
        start_app()
