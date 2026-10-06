import sys

sys.stdout.reconfigure(encoding='utf-8')

with open("PAPER.md", "r", encoding="utf-8") as f:
    text = f.read()

pos61 = text.find("These perturbation shifts illustrate the mathematical behavior")
print(text[pos61:pos61+1000])
