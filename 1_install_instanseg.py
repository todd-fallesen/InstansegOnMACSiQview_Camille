import subprocess
result = subprocess.run([r"C:\Program Files\MACSiQView\Python3\python.exe", "-m", "pip", "install", "--user", "instanseg-torch"], capture_output=True, text=True)
print("STDOUT:", result.stdout)
print("STDERR:", result.stderr)
print("Return code:", result.returncode)