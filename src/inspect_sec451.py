import sys

sys.stdout.reconfigure(encoding='utf-8')

with open("PAPER.md", "r", encoding="utf-8") as f:
    text = f.read()

pos451 = text.find("#### 4.5.1 Rationale for the 20-Feature Schema")
print(text[pos451:pos451+1500])
