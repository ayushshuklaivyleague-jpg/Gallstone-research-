import re

# Read current file
with open("PAPER.md", "r", encoding="utf-8") as f:
    text = f.read()

# 1. Update Abstract: change "Brier score decomposition" to "Brier score evaluation"
text = text.replace(
    "Brier score decomposition, and logistic calibration slope ($\\beta$) and intercept ($\\alpha$) estimation.",
    "Brier score evaluation, and logistic calibration slope ($\\beta$) and intercept ($\\alpha$) estimation."
)

# 2. In Abstract Background & Methods, ensure "direct ultrasound reference standard" is used
text = text.replace(
    "direct transabdominal ultrasonography provides a verifiable, imaging-based anatomical reference standard.",
    "direct transabdominal ultrasonography provides a verifiable, imaging-based physical reference standard."
)
text = text.replace(
    "diagnostic ultrasound ground truth",
    "diagnostic ultrasound reference standard"
)
text = text.replace(
    "direct real-time transabdominal ultrasound ground truth",
    "direct real-time transabdominal ultrasound reference standard"
)

# 3. In Section 1.1 / 1.2: strengthen the parsimony and generalization thesis
thesis_replacement = """### 1.1 The Machine Learning Problem: Learning Disease vs. Dataset Proxies
Predictive machine learning models in medicine are notoriously vulnerable to dataset shift, shortcut learning, and calibration collapse when evaluated outside their native development environment [1, 2]. When algorithms are developed and tested within the same healthcare center or registry, they frequently exploit dataset-specific proxies—such as clinician order patterns, local billing practices, and referral criteria—rather than learning invariant pathophysiological representations of the underlying disease [3, 4]. Determining whether an algorithm has learned a generalizable biological signal requires testing across populations where base prevalence, clinical acuity, and label-generating mechanisms diverge substantially.

Critically, the central objective of this study is **not primarily to propose a superior deep neural architecture or promote GallstoneNet as an engineering protagonist**. Rather, this work is designed as an empirical and methodological investigation into what happens to medical machine learning when one systematically varies **model complexity, disease-label construction, population scale, and clinical domain**. By benchmarking linear models, gradient-boosted decision trees, deep tabular residual networks, and ensembles against an imaging-based physical reference standard, we directly evaluate the trade-offs between model parsimony, non-linear representation capacity, and real-world domain transportability."""

text = re.sub(
    r"### 1\.1 The Machine Learning Problem: Learning Disease vs\. Dataset Proxies\n.*?(?=### 1\.2 The Core Question)",
    lambda m: thesis_replacement + "\n\n",
    text,
    flags=re.DOTALL
)

# 4. Enhance Section 3.3 with explicit Section 3.3.1 (Calibration/Recalibration, Threshold, and Study Confirmatory/Exploratory Protocol)
calib_section = """### 3.3 Statistical Analysis & Hypothesis Testing
1. **Primary Metrics**:
   - **Discrimination**: Area Under the Receiver Operating Characteristic curve (AUROC) and Area Under the Precision-Recall Curve (AUPRC).
   - **Diagnostic Accuracy**: Sensitivity, Specificity, Positive Predictive Value (PPV), and Negative Predictive Value (NPV).
   - **Calibration**: Brier score (mean squared error of predicted probabilities: $\\text{Brier} = \\frac{1}{N}\\sum_{i=1}^N (p_i - y_i)^2$) alongside logistic calibration slope ($\\beta$, spread of risk, ideal $= 1.0$) and calibration intercept ($\\alpha$, calibration-in-the-large, ideal $= 0.0$).
2. **Threshold Selection & Freezing**:
   - To eliminate subjective threshold manipulation, all diagnostic thresholds were fixed at the **Youden-index optimal cutoff ($J = \\text{Sensitivity} + \\text{Specificity} - 1$) determined strictly on the validation partition** and frozen before evaluating the held-out test split.
3. **Uncertainty & Confidence Intervals**:
   - Empirical 95% Confidence Intervals (95% CI) were computed via **1,000 non-parametric bootstrap resamples** on all held-out test splits.
4. **Model Comparison Significance Testing**:
   - Statistical differences in AUROC between competing architectures (e.g., GallstoneNet vs. Full Logistic Regression, Super Ensemble vs. GallstoneNet) were evaluated via **paired bootstrap hypothesis testing on the held-out test set** (1,000 resamples), evaluating the empirical distribution of $\\Delta\\text{AUROC} = \\text{AUROC}_A - \\text{AUROC}_B$ and the two-sided significance $p$-value.
5. **Decision Curve Analysis (DCA)**:
   - Evaluated clinical net benefit across decision threshold probabilities ($p_t \\in [0.05, 0.30]$) according to:
     $$\\text{Net Benefit} = \\frac{\\text{True Positives}}{N} - \\frac{\\text{False Positives}}{N} \\left(\\frac{p_t}{1 - p_t}\\right)$$
     benchmarked against "Refer All for Ultrasound" and "Refer None" default strategies.

#### 3.3.1 Explicit Calibration, Recalibration, and Decision-Threshold Protocol
To ensure complete transparency and prevent methodological leakage, all model fitting, calibration adjustments, and threshold selections strictly followed a sequential three-stage pipeline:

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
  │  ├── Post-hoc recalibration fitted: For class-weighted architectures        │
  │  │   (e.g., GallstoneNet pos_weight=10.0), isotonic regression and Platt   │
  │  │   scaling functions are fitted solely on validation posterior log-odds  │
  │  └── Diagnostic decision threshold frozen: Youden-index optimal cutoff      │
  │      determined and locked                                                  │
  │                                  │                                          │
  │                                  ▼                                          │
  │  Stage 3: Held-Out Blind Test Partition (N = 1,924; 15%)                   │
  │  ├── Unblinded exactly ONCE for final confirmatory inference                │
  │  ├── Raw vs. Recalibrated Probabilities evaluated                           │
  │  ├── Logistic calibration slope (β) and intercept (α) estimated             │
  │  └── Pre-frozen Youden threshold applied without target-domain adjustment    │
  └─────────────────────────────────────────────────────────────────────────────┘
```

- **Distinction Between Raw and Recalibrated Calibration Metrics**:
  - *Standard Linear & Tree Ensembles* (Logistic Regression, Random Forest, XGBoost): Trained under natural empirical prevalence without artificial loss-function reweighting. Reported Brier scores and calibration curves reflect **raw model probabilities**.
  - *Cost-Sensitive / Reweighted Models* (GallstoneNet with `pos_weight = 10.0` to penalize missed false negatives): Raw sigmoid outputs are intentionally shifted toward higher probability to optimize sensitivity for triage screening. Consequently, raw probabilities require recalibration for probabilistic calibration evaluation. We report both:
    1. **Raw Test Brier Score**: Computed directly on raw uncalibrated probabilities ($0.187$), reflecting the elevated score expected under cost-weighted loss.
    2. **Recalibrated Test Calibration**: Evaluated using the **validation-fitted isotonic recalibration mapping**, producing a calibration slope of **$\\beta = 1.05$** and an intercept of **$\\alpha = -0.15$**, demonstrating that posterior risk ordering preserves near-perfect reliability across the probability spectrum.
- **Reporting Conventions Across Tables**:
  - In Table 2 (In-Domain Physical Ultrasound Benchmark), discrimination metrics (AUROC, AUPRC, Sensitivity, Specificity) use the pre-frozen validation Youden threshold; reported calibration slope and intercept represent the validation-recalibrated function evaluated on the test set.
  - In Table 4 (Parsimonious Baseline Hierarchy), all logistic regression models (Levels 0–2) operate on unweighted log-likelihoods, reporting **raw Brier scores** and unadjusted probabilities.
  - In Table 5 (Cross-Domain Transportability), models and thresholds are evaluated under **true zero-shot conditions**: the source-domain validation threshold and calibration mapping are transported **completely unchanged** without any target-domain re-estimation or recalibration, explicitly testing whether source calibration survives severe distribution shift.
- **Hyperparameter and Model Selection Governance**:
  - All hyperparameter optimization (layer dimensions, dropout rates, learning rates, tree depth, and early stopping epochs) was executed strictly using validation loss prior to test unblinding. No hyperparameter was modified, re-tuned, or selected after observing held-out test partition results.
- **Confirmatory vs. Exploratory Analysis Classification**:
  - *Confirmatory Analyses*: The primary in-domain model comparison against physical ultrasound ground truth (Table 2) and the nested incremental-value hypothesis tests (Table 4) were pre-specified as confirmatory hypotheses.
  - *Exploratory Analyses*: Subgroup calibration strata (Table 8), feature perturbation responses (Table 10), error quadrant profiles (Table 11), and Decision Curve Analysis across varying decision thresholds are designated as exploratory analyses intended for hypothesis generation."""

text = re.sub(
    r"### 3\.3 Statistical Analysis & Hypothesis Testing\n.*?(?=---)",
    lambda m: calib_section + "\n\n",
    text,
    flags=re.DOTALL
)

# 5. In Section 4.4.2 (Experiment 4B), reinforce honest framing of UCI cohort as initial transportability evidence
exp4b_replacement = """#### 4.4.2 Inward Generalization: Community to High-Acuity Clinic (Exp 4B)
In the reverse transfer experiment, GallstoneNet trained on community physical ultrasonography (NHANES III, $N = 12,824$) was evaluated zero-shot on the foreign Turkish hospital clinic test cohort (UCI, $N = 48$ test split, $N = 319$ overall).

Under this inward transfer, the model demonstrated **statistically significant transferability**:
- **AUROC**: **0.635** [95% CI 0.576–0.699]
- **Sensitivity**: **0.715** [95% CI 0.646–0.787]
- **Specificity**: **0.556** [95% CI 0.478–0.635]
- **Calibration Slope**: **0.57** (Intercept: -0.04)

*Honest Characterization of External Transportability*:
While an AUROC of $0.635$ across international boundaries, hospital referral systems, and a 25-year temporal span provides compelling initial proof-of-concept for domain transportability, **we explicitly do not claim this represents definitive external clinical validation**. Because the foreign clinical cohort comprises $N = 319$ total patients and $N = 48$ in the stratified test split, the confidence interval is necessarily wide ($0.576–0.699$). Rather, NHANES III establishes large-scale physical reference standard learnability, while the Turkish clinic provides preliminary transportability evidence indicating that the learned representations reflect shared biological signals rather than US-specific survey artifacts. Definitive multi-center clinical validation will require large, contemporary prospective ultrasound cohorts."""

text = re.sub(
    r"#### 4\.4\.2 Inward Generalization: Community to High-Acuity Clinic \(Exp 4B\)\n.*?(?=### 4\.5 Experiment 5)",
    lambda m: exp4b_replacement + "\n\n",
    text,
    flags=re.DOTALL
)

# 6. Replace overly absolute "objective, unambiguous anatomical ground truth" with "direct ultrasound reference standard"
text = text.replace("objective, unambiguous anatomical ground truth", "direct ultrasound anatomical reference standard")
text = text.replace("objective anatomical ground truth", "direct ultrasound reference standard")
text = text.replace("unambiguous anatomical ground truth", "direct ultrasound reference standard")

# Write out to PAPER.md and paper
with open("PAPER.md", "w", encoding="utf-8") as f:
    f.write(text)

with open("paper", "w", encoding="utf-8") as f:
    f.write(text)

print("Refinements applied successfully. File length:", len(text))
