import re

with open("PAPER.md", "r", encoding="utf-8", errors="replace") as f:
    text = f.read()

print("File length:", len(text))
print("Number of lines:", len(text.splitlines()))

# Check for sections
sections = re.findall(r"^##\s+.*$", text, flags=re.MULTILINE)
print("Sections found:")
for s in sections:
    print(" ", s)
