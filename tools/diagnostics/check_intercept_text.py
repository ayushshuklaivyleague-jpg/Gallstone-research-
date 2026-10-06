import sys

sys.stdout.reconfigure(encoding='utf-8')

with open("PAPER.md", "r", encoding="utf-8") as f:
    text = f.read()

pos = text.find("Mathematical Interpretation of Test Calibration Parameters")
print(text[pos:pos+700])
