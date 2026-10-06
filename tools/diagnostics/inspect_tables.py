import sys

sys.stdout.reconfigure(encoding='utf-8')

with open("PAPER.md", "r", encoding="utf-8", errors="replace") as f:
    text = f.read()

pos_t4 = text.find("TABLE 4:")
print("=== Table 4 snippet ===")
print(text[pos_t4-100:pos_t4+1200])

pos_t8 = text.find("TABLE 8:")
print("\n=== Table 8 snippet ===")
print(text[pos_t8-100:pos_t8+1500])
