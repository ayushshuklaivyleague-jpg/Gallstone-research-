import sys

sys.stdout.reconfigure(encoding='utf-8')

with open("PAPER.md", "r", encoding="utf-8") as f:
    text = f.read()

pos63 = text.find("### 6.3 Illustrative Decision-Analytic Framework")
print(text[pos63:pos63+1000])
