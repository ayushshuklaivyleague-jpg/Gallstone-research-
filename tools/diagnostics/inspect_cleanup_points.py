import re

with open("PAPER.md", "r", encoding="utf-8") as f:
    text = f.read()

# 1. Search for Table 2
matches_t2 = re.findall(r".*Table 2.*", text)
print("=== Matches for Table 2 ===")
for m in matches_t2:
    print(m)

# 2. Search for counterbalances
matches_cb = re.findall(r".*counterbalance.*", text, re.IGNORECASE)
print("\n=== Matches for counterbalance ===")
for m in matches_cb:
    print(m)

# 3. Search for Section 6.3 triage text
pos63 = text.find("### 6.3 Illustrative Decision-Analytic Framework")
print("\n=== Section 6.3 snippet ===")
print(text[pos63:pos63+1200])

# 4. Search for "ground truth"
matches_gt = re.findall(r".*ground truth.*", text, re.IGNORECASE)
print(f"\n=== Total 'ground truth' matches: {len(matches_gt)} ===")
for m in matches_gt[:15]:
    print(m)
