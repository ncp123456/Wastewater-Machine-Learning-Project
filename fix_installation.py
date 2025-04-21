import subprocess
import sys
import os

def run_command(command):
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    else:
        print(result.stdout)
    return result.returncode == 0

def main():
    # Upgrade pip and install/upgrade essential build tools
    commands = [
        "python -m pip install --upgrade pip",
        "pip install --upgrade setuptools wheel",
        "pip install ninja",
        "pip install --upgrade build"
    ]

    for cmd in commands:
        if not run_command(cmd):
            print(f"Failed to run: {cmd}")
            return

    # Install tabpfn with specific options
    tabpfn_cmd = "pip install --no-build-isolation tabpfn"
    if not run_command(tabpfn_cmd):
        print("Failed to install tabpfn with --no-build-isolation, trying alternative method...")
        # Try installing with pre-built wheels
        if not run_command("pip install --only-binary :all: tabpfn"):
            print("Failed to install tabpfn with pre-built wheels")
            return

    print("\nInstallation completed successfully!")
    print("\nPlease make sure you have Visual Studio Build Tools 2022 installed with C++ support.")
    print("You can download it from: https://visualstudio.microsoft.com/visual-cpp-build-tools/")
    print("During installation, select 'Desktop development with C++'")

if __name__ == "__main__":
    main() 