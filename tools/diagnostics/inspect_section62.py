import sys

sys.stdout.reconfigure(encoding='utf-8')

with open("PAPER.md", "r", encoding="utf-8", errors="replace") as f:
    text = f.read()

pos62 = text.find("### 6.2 Model Failure Analysis")
pos7 = text.find("## 7. Computational Environment & Reproducibility")

print(text[pos62:pos62+4000])
