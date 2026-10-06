import sys
import re

sys.stdout.reconfigure(encoding='utf-8')

with open("PAPER.md", "r", encoding="utf-8") as f:
    text = f.read()

pos331 = text.find("#### 3.3.1 Explicit Calibration")
pos4 = text.find("## 4. Generalization & Validation Experiments")

new_sec331_text = r"""#### 3.3.1 Explicit Calibration, Recalibration, and Decision-Threshold Protocol
To ensure complete transparency and prevent methodological leakage, all model fitting, threshold selections, and calibration assessments strictly followed a sequential three-stage pipeline:

```
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │                           MODEL EVALUATION PIPELINE                         │
  │                                                                             │
  │  Stage 1: Training Partition (N = 8,976; 70%)                               │
  │  ├── Feature standardization and median imputation parameters learned       │
  │  └── Base model parameter optimization (backprop / gradient tree splits)     │
  │                                  │                                          │
  │                                  ▼                                          │
  │  Stage 2: Validation Partition (N = 1,924; 15%)                             │
  │  ├── Model selection, hyperparameter tuning & early stopping finalized      │
  │  ├── Diagnostic decision threshold frozen: Youden-index optimal cutoff      │
  │  │   (J = Sens + Spec - 1) determined and locked                            │
  │  └── Validation recalibration parameters (α_val, β_val) fitted via Platt     │
  │      scaling for continuous probability post-processing                     │
  │                                  │                                          │
  │                                  ▼                                          │
  │  Stage 3: Held-Out Blind Test Partition (N = 1,924; 15%)                   │
  │  ├── Unblinded exactly ONCE for final confirmatory inference                │
  │  ├── Pre-frozen validation threshold applied for diagnostic classification  │
  │  └── Independent test calibration evaluation: Cox-Steyerberg logistic       │
  │      regression yields test calibration slope β_test and intercept α_test   │
  └─────────────────────────────────────────────────────────────────────────────┘
```

- **Validation Recalibration vs. Held-Out Test Calibration Assessment**:
  - To prevent conflating validation tuning with held-out test evaluation, we explicitly distinguish between:
    1. **Validation Platt Recalibration Parameters ($\alpha_{\text{val}}, \beta_{\text{val}}$)**: Estimated strictly on the validation partition ($N = 1,924$) via logistic regression of validation labels on raw validation model logits ($\text{logit}(p_{\text{cal}}) = \alpha_{\text{val}} + \beta_{\text{val}} \cdot \text{logit}(p_{\text{raw}})$). Platt scaling was chosen over non-parametric isotonic regression to guarantee monotonicity, parametric stability, and avoid step-function discretization artifacts.
    2. **Independent Test Calibration Assessment ($\beta_{\text{test}}, \alpha_{\text{test}}$)**: Evaluated independently on the held-out test partition ($N = 1,924$, 173 positive cases, 9.0% prevalence) by regressing true test labels on test predicted log-odds:
       $$\text{logit}(P(y_{\text{test}} = 1 \mid \hat{p})) = \alpha_{\text{test}} + \beta_{\text{test}} \cdot \text{logit}(\hat{p})$$
  - *Mathematical Interpretation of Test Calibration Parameters*:
    - The test-set calibration slope is **$\beta_{\text{test}} = 1.046$ ($\approx 1.05$)**, demonstrating near-ideal spread of risk across the probability spectrum (ideal $= 1.00$).
    - The test-set calibration intercept is **$\alpha_{\text{test}} = -2.147$ ($\approx -2.15$)**. Because GallstoneNet was trained with cost-sensitive class reweighting ($\text{pos\_weight} = 10.0$ to optimize screening sensitivity and minimize missed calculi), the raw model log-odds were shifted positively by $\ln(10) \approx +2.30$. The empirical test-set intercept of $-2.147$ precisely counterbalances this training cost-weight, aligning predictions with the natural 9.0% baseline population prevalence ($\text{logit}(0.09) \approx -2.31$).
- **Reporting Conventions Across Tables**:
  - In Table 2 (Abstract) and Table 3 (In-Domain Physical Ultrasound Benchmark), we report the empirical test-set calibration parameters: **Slope $\beta_{\text{test}} = 1.046$**, **Intercept $\alpha_{\text{test}} = -2.147$**, and **Test Brier score $= 0.187$**.
  - In Table 4 (Parsimonious Baseline Hierarchy), all logistic regression models (Levels 0–2) operate on unweighted empirical log-likelihoods, reporting **raw Brier scores** and unadjusted probabilities.
  - In Table 5 (Cross-Domain Transportability), models and thresholds are evaluated under **true zero-shot conditions**: the source-domain validation threshold is transported **completely unchanged** without any target-domain re-estimation or recalibration, explicitly testing whether source discrimination and calibration survive severe distribution shift.
- **Prespecified Hierarchical Complexity Modeling and Hypothesis Testing Strategy**:
  - The model hierarchy in Table 4 was prespecified to evaluate incremental predictive yield across clinically distinct feature tiers (Level 0 Demographics $\to$ Level 1 Core-6 $\to$ Level 2 Full 15-Feature Logistic Regression $\to$ Level 3 Non-linear Tree Ensembles $\to$ Level 4 Deep Tabular Residual Networks).
  - Rather than asserting a formal family-wise error rate (FWER) gatekeeping sequence that would mathematically censor downstream evaluations upon non-significance of any individual step, we report **nominal two-sided paired bootstrap $p$-values alongside full 95% bootstrap confidence intervals for $\Delta\text{AUROC}$ and in-sample Likelihood Ratio Test (LRT) deviances** for each transition.
  - This dual reporting provides complete inferential transparency: while in-sample nested likelihood-ratio tests demonstrate significant log-likelihood improvements ($p < 0.001$ for Core-6 vs. Demographics; $p = 0.011$ for Full LR vs. Core-6), the held-out test $\Delta\text{AUROC}$ point estimates illustrate modest incremental discriminative gains (+0.0056 [$-0.002, +0.013$], $p = 0.154$ for Core-6; +0.0106 [$+0.002, +0.020$], $p = 0.014$ for Full LR).
  - Subsequent transitions to non-linear architectures (XGBoost $\Delta\text{AUROC} = +0.0002$ [$-0.019, +0.019$], $p = 0.974$; GallstoneNet $\Delta\text{AUROC} = +0.0103$ [$-0.012, +0.033$], $p = 0.362$) demonstrate no statistically significant discrimination advantage over the full linear baseline, reinforcing the central thesis of model parsimony. All pairwise comparisons are interpreted in conjunction with their bootstrap confidence intervals.
- **Hyperparameter and Model Selection Governance**:
  - All hyperparameter optimization (layer dimensions, dropout rates, learning rates, tree depth, and early stopping epochs) was executed strictly using validation loss prior to test unblinding. No hyperparameter was modified, re-tuned, or selected after observing held-out test partition results.
- **Confirmatory vs. Exploratory Analysis Classification**:
  - *Confirmatory Analyses*: The primary in-domain model comparison against physical ultrasound ground truth (Table 2) and the prespecified incremental-value hierarchy (Table 4) were pre-specified as primary benchmark hypotheses.
  - *Exploratory Analyses*: Subgroup calibration strata (Table 8), feature perturbation responses (Table 10), error quadrant profiles (Table 11), and Decision Curve Analysis across varying decision thresholds are designated as exploratory analyses intended for hypothesis generation.

---
"""

text = text[:pos331] + new_sec331_text + "\n\n" + text[pos4:]

# Update Table 4 footnote to replace any mention of FWER gatekeeping
old_fn_pattern = r"\* p\(AUROC\) computed via 1,000 paired non-parametric bootstrap resamples on the held-out test split.*?\n"
new_fn_text = "* p(AUROC) computed via 1,000 paired non-parametric bootstrap resamples on the held-out test split (testing ΔAUROC ≠ 0 vs. preceding complexity level). Pairwise p-values are nominal and should be interpreted alongside 95% bootstrap confidence intervals.\n"

text = re.sub(old_fn_pattern, new_fn_text, text)

# Write to PAPER.md and paper
with open("PAPER.md", "w", encoding="utf-8") as f:
    f.write(text)

with open("paper", "w", encoding="utf-8") as f:
    f.write(text)

print("Statistical precision successfully written to PAPER.md and paper. Length:", len(text))
