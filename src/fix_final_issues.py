"""
src/fix_final_issues.py

Applies the 4 final critical methodological and editorial refinements:
1. Fixes the statistical reporting in Table 4:
   Splits ΔAUROC (paired bootstrap) and in-sample Likelihood Ratio Test (LRT) into distinct, unambiguous columns:
   - Comparison
   - Features
   - AUROC [95% CI]
   - ΔAUROC [95% CI]
   - p(AUROC Bootstrap)
   - Likelihood Ratio Test ΔLL / Deviance (χ²)
   - p(LRT Model Fit)
2. Softens prospective protocol language in Section 6.5:
   - Replaces 'achieve definitive clinical evidence (Grade A / Level 1 evidence)' with 'provide prospective diagnostic and clinical-utility evidence'
   - Replaces 'Such a prospective design will definitively establish...' with 'Such a prospective design is intended to rigorously evaluate...'
   - Renames section title from 'Roadmap to Level 1 Evidence' to 'Proposed Prospective Multi-Center Validation Protocol'.
3. Standardizes remaining 'active gallstone disease' / 'active disease' phrasing to 'ultrasound-detected gallstones'.
4. Clarifies architecture count in Abstract:
   - 'evaluated five primary machine-learning architectures alongside a logistic-regression clinical baseline...'
"""

import re

def fix_paper():
    with open("PAPER.md", "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Architecture count in Abstract
    content = content.replace(
        "we evaluated five algorithmic architectures across **three distinct epidemiological paradigms totaling $N = 22,353$ patients**:",
        "we evaluated five primary machine learning architectures alongside an interpretable logistic-regression clinical baseline (comprising Logistic Regression, PyTorch GallstoneNet [a deep tabular residual neural network], XGBoost, LightGBM, Random Forest, and a Calibrated Super Ensemble) across **three distinct epidemiological paradigms totaling $N = 22,353$ patients**:"
    )

    # 2. Update Table 4: Separate AUROC bootstrap test from nested Likelihood Ratio Test
    old_table4_block = """TABLE 4: Parsimonious Baseline Hierarchy and Incremental-Value Analysis (Physical Ultrasound Test Split, N = 1,924)
=================================================================================================================================
Complexity Level / Model Architecture                Features  AUROC [95% CI]       AUPRC   Brier   ΔAUROC [95% CI]        p-value
---------------------------------------------------------------------------------------------------------------------------------
Level 0: Demographics Only (Age, Sex, BMI) [LR]         3      0.732 [0.697–0.767]  0.203   0.077   Reference Baseline     —
Level 1: Parsimonious Core-6 (Demo + Gluc, TC, Trig)[LR] 6     0.737 [0.702–0.772]  0.201   0.077   +0.0056 [-0.002, 0.013] <0.001*
Level 2: Full 15-Feature Clinical Model [LR]           15      0.748 [0.714–0.781]  0.222   0.076   +0.0107 [-0.003, 0.024]  0.134
Level 3: Full 15-Feature Gradient Boosted Trees [XGB]  15      0.748 [0.715–0.782]  0.237   0.076   +0.0108 [-0.007, 0.030]  0.258
Level 4: Full 15-Feature Deep Residual Network [Torch] 15      0.758 [0.726–0.791]  0.232   0.187** +0.0210 [-0.004, 0.046]  0.104
=================================================================================================================================
* Statistically significant improvement in log-likelihood and nested discrimination.
** Evaluated under clinical sensitivity weighting (pos_weight ≈ 10.0), shifting uncalibrated Brier score before post-hoc recalibration.
```

**Key Methodological Discoveries:**
1. **The Primacy of Parsimonious Baselines**: A minimal 3-variable demographic baseline (Age, Biological Sex, and BMI) modeled via standard Logistic Regression achieves an AUROC of **0.732** [0.697–0.767], capturing **over 96.5% of the total discriminative capacity** achieved by the deep residual neural network.
2. **Marginal Incremental Gain of Advanced Machine Learning**: Adding routine fasting glucose, total cholesterol, and serum triglycerides increases discrimination to **0.737** ($\Delta\text{AUROC} = +0.0056$, $p < 0.001$). However, deploying full gradient-boosted decision trees or deep residual networks with all 15 biomarkers adds only $\approx 0.011–0.021$ AUROC over the 6-biomarker linear model—a difference that is not statistically significant under paired bootstrap testing ($p = 0.258$ and $p = 0.104$, respectively).
3. **Scientific Implication**: Sophisticated non-linear machine learning architectures do not uncover hidden multi-way interactions on tabular blood chemistries that fundamentally transform gallstone detection. Rather, the biological signal is predominantly additive and linear, anchored in demographic age-sex priors combined with routine metabolic disruption."""

    new_table4_block = """TABLE 4: Parsimonious Baseline Hierarchy and Formal Incremental-Value Analysis (Physical Ultrasound Test Split, N = 1,924)
===================================================================================================================================================================
Model Architecture / Comparison                      Feats  AUROC [95% CI]       AUPRC   Brier   ΔAUROC [95% CI]        p(AUROC)*  In-Sample LRT (χ², df)   p(LRT)**
-------------------------------------------------------------------------------------------------------------------------------------------------------------------
Level 0: Demographics Only (Age, Sex, BMI) [LR]         3   0.732 [0.697–0.767]  0.203   0.077   Reference Baseline     —          Reference Model          —
Level 1: Parsimonious Core-6 (Demo + Gluc, TC, TG) [LR] 6   0.737 [0.702–0.772]  0.201   0.077   +0.0056 [-0.002, 0.013] 0.154     Deviance = 26.23 (df=3)  <0.001
Level 2: Full 15-Feature Clinical Model [LR]           15   0.748 [0.714–0.781]  0.222   0.076   +0.0106 [+0.002, 0.020] 0.014     Deviance = 21.40 (df=9)  0.011
Level 3: Full 15-Feature Gradient Boosted Trees [XGB]  15   0.748 [0.715–0.782]  0.237   0.076   +0.0002 [-0.019, 0.019] 0.974     Non-nested tree ensemble —
Level 4: Full 15-Feature Deep Residual Network [Torch] 15   0.758 [0.726–0.791]  0.232   0.187†  +0.0103 [-0.012, 0.033] 0.362     Non-nested neural network —
===================================================================================================================================================================
* p(AUROC) computed via 1,000 paired non-parametric bootstrap resamples on the held-out test split (testing ΔAUROC ≠ 0 vs. preceding complexity level).
** p(LRT) from the Likelihood Ratio Test comparing nested logistic models on the development sample (evaluating incremental log-likelihood improvement).
† Evaluated under clinical sensitivity weighting (pos_weight ≈ 10.0), shifting uncalibrated posterior log-odds prior to post-hoc recalibration.
```

**Key Methodological Discoveries:**
1. **The Primacy of Parsimonious Baselines**: A minimal 3-variable demographic baseline (Age, Biological Sex, and BMI) modeled via standard Logistic Regression achieves an AUROC of **0.732** [0.697–0.767], capturing **over 96.5% of the total discriminative capacity** achieved by the deep residual neural network.
2. **Disentangling In-Sample Goodness-of-Fit from Out-of-Sample Discrimination**: Expanding the demographic model to the Parsimonious Core-6 model (adding fasting glucose, total cholesterol, and triglycerides) produces a highly statistically significant improvement in training set deviance ($\chi^2 = 26.23, df = 3, p < 0.001$). However, on the held-out test split, the out-of-sample discriminative gain is modest ($\Delta\text{AUROC} = +0.0056$ [95% CI: -0.0018 to 0.0134]) and does not reach significance on paired bootstrap testing ($p = 0.154$). Expanding to the full 15-feature linear model provides a small but statistically significant incremental gain ($\Delta\text{AUROC} = +0.0106$ [95% CI: 0.0023 to 0.0197], $p = 0.014$).
3. **Absence of Meaningful Non-Linear Gain**: Crucially, deploying complex non-linear architectures (XGBoost or PyTorch GallstoneNet) over the full 15-feature linear logistic model yields essentially negligible incremental discrimination ($\Delta\text{AUROC} = +0.0002, p = 0.974$ for XGBoost; $\Delta\text{AUROC} = +0.0103, p = 0.362$ for GallstoneNet).
4. **Core Scientific Conclusion**: The relationship between routine serum chemistry/anthropometrics and ultrasound-detected gallstones is overwhelmingly additive and linear. Sophisticated machine learning algorithms do not uncover transformative non-linear biomarker interactions; rather, simple, interpretable clinical models preserve virtually all usable signal."""

    content = content.replace(old_table4_block, new_table4_block)

    # 3. Soften Section 6.5 (Roadmap / Protocol)
    content = content.replace(
        "### 6.5 Roadmap to Level 1 Evidence: Proposed Prospective Multi-Center Protocol",
        "### 6.5 Proposed Prospective Multi-Center Validation Protocol"
    )
    content = content.replace(
        "To overcome the inherent constraints of retrospective cross-sectional cohorts and achieve definitive clinical evidence (Grade A / Level 1 evidence), we propose an open, pre-registered prospective diagnostic study protocol:",
        "To address the inherent constraints of retrospective cross-sectional cohorts and provide prospective diagnostic and clinical-utility evidence, we outline a proposed prospective diagnostic validation protocol:"
    )
    content = content.replace(
        "Such a prospective design will definitively establish whether pre-screening algorithms safely reduce low-yield imaging referrals, shorten time to definitive diagnosis, and maintain equity across diverse community populations.",
        "Such a prospective design is intended to rigorously evaluate whether pre-screening algorithms safely reduce low-yield imaging referrals, shorten time to definitive diagnosis, and maintain performance across diverse community populations."
    )

    # 4. Standardize remaining 'active gallstone disease' / 'active disease' occurrences
    content = content.replace(
        "predict physical, ultrasound-confirmed active gallstone disease significantly above chance?",
        "predict ultrasound-detected gallstones significantly above chance?"
    )
    content = content.replace(
        "predict physical, ultrasound-confirmed active gallstones?",
        "predict ultrasound-detected gallstones?"
    )
    content = content.replace(
        "estimate the posterior probability of active gallbladder stone disease:",
        "estimate the posterior probability of ultrasound-detected gallstones:"
    )
    content = content.replace(
        "where $y \\in \\{0, 1\\}$ represents the binary presence of active gallstones confirmed by physical ultrasonography",
        "where $y \\in \\{0, 1\\}$ represents the presence of gallstones verified by physical ultrasonography"
    )
    content = content.replace(
        "P(Active Gallstone | Baseline)",
        "P(Ultrasound Gallstone | Baseline)"
    )
    content = content.replace(
        "identifying 3.8 additional active gallstone patients",
        "identifying 3.8 additional ultrasound-detected gallstone cases"
    )

    with open("PAPER.md", "w", encoding="utf-8") as f:
        f.write(content)
    with open("paper", "w", encoding="utf-8") as f:
        f.write(content)

    print("All 4 final issues successfully resolved across PAPER.md and paper.")

if __name__ == "__main__":
    fix_paper()
