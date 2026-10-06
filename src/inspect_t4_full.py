import sys

sys.stdout.reconfigure(encoding='utf-8')

with open("PAPER.md", "r", encoding="utf-8") as f:
    text = f.read()

pos_t4 = text.find("TABLE 4:")
print(text[pos_t4:pos_t4+2000])
