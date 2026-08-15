from pathlib import Path


ROOT = Path(__file__).resolve().parent

EXCLUDED_DIRS = {
    ".venv",
    ".git",
    "__pycache__",
    ".idea",
}

for path in sorted(ROOT.rglob("*")):

    relative = path.relative_to(ROOT)

    # Skip excluded directories.
    if any(
        part in EXCLUDED_DIRS
        for part in relative.parts
    ):
        continue

    depth = len(relative.parts) - 1
    indent = "    " * depth

    if path.is_dir():
        print(f"{indent}[DIR]  {path.name}")
    else:
        print(f"{indent}[FILE] {path.name}")