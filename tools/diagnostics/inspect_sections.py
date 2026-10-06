import sys
import re

sys.stdout.reconfigure(encoding='utf-8')

with open("PAPER.md", "r", encoding="utf-8", errors="replace") as f:
    text = f.read()

pos3 = text.find("## 3. Learning Framework & Statistical Methodology")
pos4 = text.find("## 4. Generalization & Validation Experiments")
pos5 = text.find("## 5. Subgroup, Sensitivity, Ablation, and Decision Curve Analyses")
pos6 = text.find("## 6. Model Response Analysis, Error Characterization & Discussion")
pos7 = text.find("## 7. Computational Environment & Reproducibility")

print("=== Section 3 snippet ===")
print(text[pos3:pos3+2500])

print("\n=== Section 4 Exp 5 snippet ===")
pos_exp5 = text.find("### 4.5 Experiment 5")
print(text[pos_exp5:pos_exp5+1500])
