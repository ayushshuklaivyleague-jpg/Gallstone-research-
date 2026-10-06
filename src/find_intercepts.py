import sys
import re

sys.stdout.reconfigure(encoding='utf-8')

with open("PAPER.md", "r", encoding="utf-8") as f:
    text = f.read()

# Find all occurrences of 2.15, -2.15, -0.15, 1.05, intercept
matches = re.findall(r".*(?:2\.147|2\.15|-0\.15|intercept|slope).*?$", text, flags=re.MULTILINE | re.IGNORECASE)
print(f"Total matching lines: {len(matches)}")
for m in matches[:30]:
    print(m.strip())
