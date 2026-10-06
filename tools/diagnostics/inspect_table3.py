import sys

sys.stdout.reconfigure(encoding='utf-8')

with open("PAPER.md", "r", encoding="utf-8") as f:
    text = f.read()

pos_t3 = text.find("TABLE 3:")
print(text[pos_t3-50:pos_t3+1500])
