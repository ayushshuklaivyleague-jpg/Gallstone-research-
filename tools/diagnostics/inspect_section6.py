import sys

sys.stdout.reconfigure(encoding='utf-8')

with open("PAPER.md", "r", encoding="utf-8", errors="replace") as f:
    text = f.read()

pos6 = text.find("## 6. Model Response Analysis, Error Characterization & Discussion")
pos7 = text.find("## 7. Computational Environment & Reproducibility")

print(text[pos6:pos6+3500])
