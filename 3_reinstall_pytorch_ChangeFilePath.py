import subprocess

# Step 1: Uninstall existing torch
print("Uninstalling existing torch...")
result = subprocess.run(
    [r"C:\Program Files\MACSiQView\Python3\python.exe", "-m", "pip", "uninstall", "torch", "-y", "--user"],
    capture_output=True, text=True
)
print("STDOUT:", result.stdout)
print("STDERR:", result.stderr)

# Step 2: Delete leftover torch files manually
import shutil
import os
torch_path = r"C:\Users\username\AppData\Roaming\Python\Python39\site-packages\torch"
if os.path.exists(torch_path):
    print("Removing leftover torch folder...")
    shutil.rmtree(torch_path, ignore_errors=True)
    print("Done.")
else:
    print("No leftover torch folder found.")

# Step 3: Install CUDA torch
print("Installing CUDA torch...")
result = subprocess.run(
    [r"C:\Program Files\MACSiQView\Python3\python.exe", "-m", "pip", "install", "--user", "torch", "--index-url", "https://download.pytorch.org/whl/cu121"],
    capture_output=True, text=True
)
print("STDOUT:", result.stdout)
print("STDERR:", result.stderr)
print("Return code:", result.returncode)