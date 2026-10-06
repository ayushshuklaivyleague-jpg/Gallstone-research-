import sys
import re

sys.stdout.reconfigure(encoding='utf-8')

with open("PAPER.md", "r", encoding="utf-8") as f:
    text = f.read()

# Find occurrences with line numbers
lines = text.splitlines()
for i, line in enumerate(lines):
    if "ground truth" in line.lower():
        print(f"Line {i+1}: {line}")
