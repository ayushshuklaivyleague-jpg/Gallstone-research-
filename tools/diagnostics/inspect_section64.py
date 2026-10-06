import sys

sys.stdout.reconfigure(encoding='utf-8')

with open("PAPER.md", "r", encoding="utf-8", errors="replace") as f:
    text = f.read()

pos64 = text.find("### 6.4")
pos7 = text.find("## 7. Computational Environment & Reproducibility")

print(text[pos64:pos7])
