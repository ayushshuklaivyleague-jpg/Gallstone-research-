import sys

sys.stdout.reconfigure(encoding='utf-8')

with open("PAPER.md", "r", encoding="utf-8") as f:
    text = f.read()

pos61 = text.find("### 6.1 Model Response & Feature Perturbation Analysis")
print(text[pos61:pos61+1500])
