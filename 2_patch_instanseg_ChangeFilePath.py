import sys
sys.path.insert(0, r"C:\Users\charoyc\AppData\Roaming\Python\Python39\site-packages")

tiling_path = r"C:\Users\charoyc\AppData\Roaming\Python\Python39\site-packages\instanseg\utils\tiling.py"

with open(tiling_path, "r") as f:
    content = f.read()

# Fix Python 3.10+ type hint syntax that breaks on Python 3.9
fixed = content.replace(
    "map_list: torch.Tensor | None = None",
    "map_list = None"
)

with open(tiling_path, "w") as f:
    f.write(fixed)

print("Patch applied successfully!")