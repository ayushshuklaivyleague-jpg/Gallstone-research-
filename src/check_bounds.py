import sys

sys.stdout.reconfigure(encoding='utf-8')

with open("PAPER.md", "r", encoding="utf-8") as f:
    text = f.read()

pos331 = text.find("#### 3.3.1 Explicit Calibration")
pos4 = text.find("## 4. Generalization & Validation Experiments")

print("Length of section:", pos4 - pos331)
print(text[pos331:pos331+300])
print("...")
print(text[pos4-300:pos4])
