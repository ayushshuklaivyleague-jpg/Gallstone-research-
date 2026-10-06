import sys

sys.stdout.reconfigure(encoding='utf-8')

with open("PAPER.md", "r", encoding="utf-8") as f:
    text = f.read()

pos331 = text.find("#### 3.3.1")
pos4 = text.find("## 4. Generalization & Validation Experiments")
print(text[pos331:pos4])
