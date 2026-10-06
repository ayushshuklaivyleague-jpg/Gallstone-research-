import sys

sys.stdout.reconfigure(encoding='utf-8')

with open("PAPER.md", "r", encoding="utf-8", errors="replace") as f:
    text = f.read()

pos33 = text.find("### 3.3")
pos4 = text.find("## 4. Generalization & Validation Experiments")

print("=== Section 3.3 ===")
print(text[pos33:pos4])
