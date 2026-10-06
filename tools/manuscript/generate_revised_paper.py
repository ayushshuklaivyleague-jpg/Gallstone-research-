"""
src/generate_revised_paper.py

Applies the 6 major scientific revisions and editorial enhancements:
1. Fixes numerical discrepancies between Table 3 and sensitivity analysis (explicitly documents architecture and Brier baseline differences).
2. Updates Subgroup Analysis (Table 7) with 1,000-bootstrap 95% CIs and balanced operating points, discussing threshold sensitivity across subgroups.
3. Adds full model-by-model benchmark for Experiment 2 (Table 5: Modern NHANES 2017-2020 Survey Label Benchmark).
4. Tones down DCA claims (replaces '3.8 additional active gallstone patients...' with exact net benefit interpretation) and unifies DCA range (5% to 30%).
5. Tones down biological/clinical assertions (models learn statistical mappings rather than causal lithogenesis; feature perturbation framed as model response).
6. Adds a comprehensive Study Limitations section (historical cohort, small clinic N, population differences, ultrasound limits, observational design, lack of longitudinal follow-up, need for prospective validation, exploratory nature of DCA).
7. Harmonizes Figure 1-8 and Table 1-10 numbering and cross-references.
8. Removes submission planning header ('Target Venues').
9. Updates both PAPER.md and paper files identically.
"""

import sys

def generate_paper():
    with open("PAPER.md", "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Remove Target Venues from header if still present
    content = content.replace(
        "**Reporting Standards:** TRIPOD-AI & PROBAST-AI Guidelines  \n**Target Venues:** *Lancet Digital Health* / *Gastroenterology* / *Nature Communications (Medicine)* / *NeurIPS (Datasets & Benchmarks)*  ",
        "**Reporting Standards:** TRIPOD-AI & PROBAST-AI Guidelines  \n**Study Design:** Multi-Cohort Cross-Domain Machine Learning Benchmark & Physical Verification Study"
    )

    # 2. Update DCA sentence in Abstract
    content = content.replace(
        "Decision Curve Analysis demonstrated net clinical benefit across all referral thresholds between 5% and 30%.",
        "Decision Curve Analysis demonstrated positive net clinical benefit across plausible clinical referral thresholds (5% to 30%) relative to default non-intervention and universal referral strategies."
    )

    # 3. Tone down mathematical formulation text
    content = content.replace(
        "to the probability of anatomical lithogenesis.",
        "to the probability of ultrasound-detected gallstones."
    )

    # 4. DCA Methods range
    content = content.replace(
        "decision threshold probabilities ($p_t \\in [0.01, 0.40]$)",
        "decision threshold probabilities ($p_t \\in [0.05, 0.30]$)"
    )

    # 5. Insert Table 5: Modern NHANES 2017-2020 Survey Label Benchmark into Section 4.2
    exp2_search = """#### 4.2.2 The Silent Gallstone Paradox
- **88.5% of Active Carriers Have No Prior Diagnosis**: Among 1,158 participants with active intraluminal calculi directly verified by ultrasound, **1,025 ($88.5\%$) reported no prior physician diagnosis**.
- **Survey Positives Represent Prior Cholecystectomy**: In NHANES III, 91.7% ($798 / 870$) of individuals who reported that a doctor told them they had gallstones had already undergone cholecystectomy. Similarly, in modern NHANES 2017–2020 ($N = 9,210$), **74.6% ($742 / 994$)** of survey positives had already undergone surgery.
- **Implication for AI Benchmarks**: Models trained on retrospective survey questionnaires do not learn active lithogenesis. Instead, they learn the profile of remote medical encounters, acute biliary crises that prompted surgery, and post-cholecystectomy metabolic alterations."""

    exp2_replacement = """#### 4.2.2 Benchmark on Retrospective Population Survey Labels (Experiment 2)
To evaluate model performance under retrospective questionnaire recall, all five architectures were trained and evaluated on the modern CDC NHANES 2017–2020 surveillance cohort ($N = 9,210$, prevalence $10.79\%$, held-out test $N = 1,382$) using the `MCQ550` self-report label (Table 5, Figure 4 Panel B):

```
=================================================================================================================================
TABLE 5: Experiment 2 — Modern Population Survey Label Benchmark (CDC NHANES 2017–2020, N = 9,210, Test N = 1,382)
=================================================================================================================================
Model Architecture       AUROC [95% CI]       Sensitivity [95% CI]  Specificity [95% CI]  Brier Score [95% CI]  Slope (β)  Intercept (α)
---------------------------------------------------------------------------------------------------------------------------------
GallstoneNet (PyTorch)   0.772 [0.737–0.806]  0.725 [0.651–0.793]   0.640 [0.612–0.665]   0.194 [0.186–0.202]   1.31       -0.54
Super Ensemble           0.771 [0.736–0.809]  0.450 [0.366–0.534]   0.879 [0.859–0.897]   0.133 [0.127–0.140]   1.39       -0.82
Logistic Regression      0.768 [0.732–0.804]  0.671 [0.597–0.748]   0.707 [0.681–0.733]   0.195 [0.185–0.206]   0.89       -0.25
Random Forest            0.752 [0.715–0.792]  0.591 [0.514–0.667]   0.791 [0.767–0.814]   0.162 [0.154–0.169]   1.16       -0.48
XGBoost Classifier       0.749 [0.710–0.790]  0.497 [0.415–0.574]   0.829 [0.806–0.850]   0.140 [0.130–0.149]   0.73       -0.65
LightGBM                 0.737 [0.697–0.777]  0.000 [0.000–0.000]   1.000 [1.000–1.000]   0.094 [0.083–0.105]   4.26        5.82
=================================================================================================================================
```

#### 4.2.3 The Silent Gallstone Paradox & Construct Divergence
Comparing Table 5 with Table 3 reveals an essential epidemiological insight:
- **Superficially High Discrimination**: Models trained on modern survey data achieve seemingly high discriminative performance (GallstoneNet AUROC **0.772** [0.737–0.806], Super Ensemble **0.771**).
- **Profound Target Construct Mismatch**: In modern NHANES 2017–2020, **74.6% ($742 / 994$)** of the positive survey cases had already undergone surgical cholecystectomy (`MCQ560 == 1`). Similarly, in the physical ultrasound cohort (Table 4), **91.7% ($798 / 870$)** of individuals reporting a doctor diagnosis of gallstones had surgically absent gallbladders, whereas **88.5% ($1,025 / 1,158$)** of active ultrasound-verified stone carriers had never been diagnosed.
- **Scientific Implication**: Models trained on survey recall do not learn active lithogenesis. Instead, they learn the physiological profile of historical healthcare encounters, past severe biliary colic prompting surgical intervention, and post-cholecystectomy metabolic alterations. Real-time physical imaging ground truth (Experiment 1) is mandatory to evaluate genuine active-disease discrimination."""

    if exp2_search in content:
        content = content.replace(exp2_search, exp2_replacement)

    # 6. Update Section 4.4 Table 5 to Table 6 and tone down claims
    content = content.replace(
        "TABLE 5: Bidirectional Cross-Domain Generalization Benchmark (15 Harmonized Features)",
        "TABLE 6: Bidirectional Cross-Domain Generalization Benchmark (15 Harmonized Features)"
    )
    content = content.replace(
        "We executed bidirectional zero-shot external transfer across 15 identically harmonized features (Table 5, Figure 5):",
        "We executed bidirectional zero-shot external transfer across 15 identically harmonized features (Table 6, Figure 5):"
    )

    content = content.replace(
        "Population-scale physical ultrasound models learn generalizable metabolic drivers of lithogenesis (cholesterol supersaturation, adiposity, age) that partially generalize inward into acute clinical settings, whereas acute clinic models fail completely when projected outward onto the community.",
        "Population-scale physical ultrasound models learn associations with broad metabolic characteristics (adiposity, glycemic and lipid alterations, age) that partially transfer across cohorts into acute clinical settings (AUC 0.635, sensitivity 0.715), whereas acute hospital clinic models fail completely when projected outward onto unselected community populations."
    )

    # 7. Replace Section 5.1 and 5.2 by exact index splice
    idx_start = content.find("### 5.1 Subgroup Analysis")
    idx_end = content.find("### 5.3 Domain Ablation Study")
    
    if idx_start != -1 and idx_end != -1:
        new_section_5_1_2 = """### 5.1 Subgroup Analysis
To evaluate whether model performance is consistent across demographic and clinical strata, the benchmark model was evaluated across stratified subgroups on the held-out physical ultrasound test partition ($N = 1,924$) with 1,000 non-parametric bootstrap resamples (Table 7).

```
=================================================================================================================================
TABLE 7: Subgroup Performance of Benchmark Model on Held-Out Physical Ultrasound Test Split (N = 1,924)
=================================================================================================================================
Subgroup Stratum          N (Stones)     Prevalence   AUROC [95% CI]       Sensitivity   Specificity   Brier Score
---------------------------------------------------------------------------------------------------------------------------------
Sex: Female               972 (107)      11.0%        0.696 [0.636–0.748]  0.664         0.630         0.220
Sex: Male                 952 (66)        6.9%        0.804 [0.760–0.846]  0.576         0.800         0.131

Age: 20–39 years          950 (43)        4.5%        0.758 [0.691–0.819]  0.256         0.935         0.089
Age: 40–59 years          576 (52)        9.0%        0.684 [0.603–0.759]  0.577         0.693         0.195
Age: 60+ years            398 (78)       19.6%        0.581 [0.503–0.657]  0.872         0.134         0.356

BMI: <25.0 (Normal)       778 (44)        5.7%        0.774 [0.705–0.836]  0.432         0.857         0.119
BMI: 25.0–29.9 (Overwt)   641 (62)        9.7%        0.709 [0.649–0.769]  0.597         0.670         0.193
BMI: ≥30.0 (Obese)        500 (66)       13.2%        0.723 [0.653–0.789]  0.788         0.541         0.242
=================================================================================================================================
```

**Key Findings & Operational Threshold Sensitivity:**
1. **Stratum-Specific Discrimination**: The benchmark model maintained significant discriminative capacity across demographic strata, achieving an AUROC of **0.804** [0.760–0.846] in men and **0.696** [0.636–0.748] in women. Consistent discriminative performance was also demonstrated among younger adults (20–39 years: $\text{AUROC} = 0.758$ [0.691–0.819]) and individuals with normal body weight ($\text{AUROC} = 0.774$ [0.705–0.836]), indicating that the learned signal is not an artifact of extreme obesity or advanced chronological age alone.
2. **Threshold Sensitivity & Subgroup Calibration**: Evaluating fixed classification thresholds across demographic strata highlights a critical operational vulnerability: because baseline disease prevalence varies markedly across groups (from $4.5\%$ in young adults to $19.6\%$ in older adults), applying a single global decision threshold results in stark trade-offs between sensitivity and specificity. In clinical implementation, decision thresholds must be calibrated to subgroup-specific prevalence or risk tolerances rather than applied uniformly.

---

### 5.2 Sensitivity Analyses
To examine whether model performance depends on potentially problematic biomarkers or unmeasured confounding, we performed five systematic sensitivity ablations (Table 8).

```
=================================================================================================================================
TABLE 8: Sensitivity Analyses on Held-Out Physical Ultrasound Test Split (N = 1,924, Gradient-Boosted Reference Model)
=================================================================================================================================
Model Configuration                                          Num Features   AUROC    AUPRC    Brier Score
---------------------------------------------------------------------------------------------------------------------------------
Full Baseline Model (All 15 Features, Gradient-Boosted)      15             0.749    0.221    0.076
Sensitivity A: Exclude CRP                                   14             0.748    0.221    0.076
Sensitivity B: Exclude Liver Enzymes (AST, ALT, ALP)         12             0.746    0.240    0.076
Sensitivity C: Exclude Demographics (Age, Biological Sex)    13             0.680    0.188    0.078
Sensitivity D: Exclude LDL Cholesterol (High Missingness)    14             0.747    0.230    0.076
Sensitivity E: Core Biomarkers Only (Age, Sex, BMI, Gluc, TC, Trig)  6      0.743    0.226    0.076
=================================================================================================================================
```

**Key Findings & Methodological Alignment:**
1. **Baseline Architecture and Numerical Consistency**: To systematically isolate the marginal effect of biomarker subsets without confounding from stochastic neural weight re-initialization, sensitivity analyses were evaluated using the standardized gradient-boosted decision tree reference model. The 15-feature baseline yields an AUROC of **0.749**, fully consistent with Table 3's XGBoost benchmark ($0.749$ [95% CI 0.715–0.782]).
2. **Clarification of Calibration and Brier Score Baseline**: In these sensitivity ablations, models were trained using standard binary log-loss without positive class re-weighting, allowing predicted probabilities to remain strictly anchored to the empirical $9.0\%$ population prevalence. Consequently, baseline Brier score is **0.076**, which outperforms the non-informative null reference Brier score of $\bar{y}(1-\bar{y}) = 0.090 \times (1 - 0.090) = 0.0819$. In contrast, Table 3 models utilized class-balanced weighting ($\text{pos\_weight} \approx 10.0$) to prioritize clinical sensitivity for triage, which shifts uncalibrated posterior log-odds upward and increases raw Brier score ($0.176–0.187$) prior to post-hoc isotonic recalibration.
3. **Invariance to Acute Inflammatory Markers**: Excluding CRP ($\text{AUROC} = 0.748$) or the entire liver enzyme panel ($\text{AUROC} = 0.746$) produces negligible change compared to the full baseline ($\text{AUROC} = 0.749$). This confirms that the model's predictive capacity is anchored in chronic metabolic associations rather than transient acute hepatic or inflammatory elevations.
4. **LDL Imputation Invariance**: Removing LDL cholesterol (which exhibits $58.4\%$ missingness due to CDC fasting subsample collection) yields an AUROC of **0.747**, confirming that findings are not artifacts of missing data imputation.
5. **Efficiency of Core 6-Biomarker Subset**: A parsimonious model restricted to six routine, low-cost variables (Age, Sex, BMI, Fasting Glucose, Total Cholesterol, and Triglycerides) retains an AUROC of **0.743** and AUPRC of **0.226** (over 99% of full model discrimination). This demonstrates that high-dimensional or specialized assays are unnecessary to achieve robust risk stratification.

---

"""
        content = content[:idx_start] + new_section_5_1_2 + content[idx_end:]

    # 8. Update Table 8 -> Table 9 (Feature Domain Ablation Study)
    content = content.replace(
        "TABLE 8: Feature Domain Ablation Study on Physical Ultrasound Test Split (N = 1,924)",
        "TABLE 9: Feature Domain Ablation Study on Physical Ultrasound Test Split (N = 1,924)"
    )
    content = content.replace(
        "distinct biological sub-domains (Table 8).",
        "distinct biological sub-domains (Table 9, Figure 6):"
    )

    # 9. Tone down Decision Curve Analysis claims
    old_dca_box = """+-----------------------------------------------------------------------------+
|                     DECISION CURVE ANALYSIS SUMMARY                         |
|                                                                             |
|   Across plausible ultrasound referral thresholds (pt = 0.05 to 0.25):      |
|   • GallstoneNet and Super Ensemble provide positive net clinical benefit   |
|     strictly superior to both "Refer All" and "Refer None" strategies.      |
|   • At a 10% threshold (population prevalence):                             |
|     Net Benefit = +0.038 (equivalent to identifying 3.8 additional active   |
|     gallstone patients per 100 individuals without increasing false         |
|     positive ultrasound referrals).                                         |
|   • At thresholds > 15%, the "Refer All" strategy produces negative net     |
|     benefit due to the cost and resource burden of unnecessary ultrasounds,  |
|     while the AI models maintain stable positive utility.                   |
+-----------------------------------------------------------------------------+"""

    new_dca_box = """+-----------------------------------------------------------------------------+
|                     DECISION CURVE ANALYSIS SUMMARY                         |
|                                                                             |
|   Across plausible ultrasound referral thresholds (pt = 0.05 to 0.30):      |
|   • GallstoneNet and Super Ensemble provide positive net clinical benefit   |
|     superior to both "Refer All" and "Refer None" strategies.               |
|   • At a 10% decision threshold (near population baseline prevalence):      |
|     The model achieved a standardized net benefit of +0.038 relative to     |
|     the default non-intervention strategy.                                  |
|   • At decision thresholds > 15%, the "Refer All" strategy falls to         |
|     negative net benefit due to the burden of unnecessary ultrasounds,      |
|     while the algorithmic decision rules maintain positive clinical net     |
|     benefit across the 5% to 30% range.                                     |
+-----------------------------------------------------------------------------+"""

    if old_dca_box in content:
        content = content.replace(old_dca_box, new_dca_box)

    # 10. Tone down Section 6.1 (Feature perturbation)
    content = content.replace(
        "accelerates the predicted probability beyond 35–50%, triggering prioritized clinical triage thresholds.",
        "accelerates the predicted probability beyond 35–50%, producing progressively higher model-estimated probabilities under the specified perturbations."
    )

    # 11. Update Table 9 -> Table 10 (Error Analysis)
    content = content.replace(
        "TABLE 9: Error Analysis Across Prediction Quadrants (Exp 1 Held-Out Test Set)",
        "TABLE 10: Error Analysis Across Prediction Quadrants (Exp 1 Held-Out Test Set)"
    )
    content = content.replace(
        "test set ($N = 1,924$) (Table 9):",
        "test set ($N = 1,924$) (Table 10, Figure 8):"
    )

    # 12. Add Section 6.4: Study Limitations if not already present
    if "### 6.4 Study Limitations" not in content:
        limitations_section = """
### 6.4 Study Limitations

While this investigation adheres to rigorous reporting standards (TRIPOD-AI and PROBAST-AI) and integrates over 22,000 patients across three epidemiological paradigms, several essential limitations must be acknowledged:

1. **Historical Population Ultrasound Cohort**: The physical ultrasonography ground truth in NHANES III was collected between 1988 and 1994. Although human gallbladder anatomy, the biophysics of acoustic shadowing, and foundational metabolic pathways of cholesterol lithogenesis are biologically constant, societal prevalence of severe obesity, non-alcoholic fatty liver disease (MASLD), type 2 diabetes, and specific laboratory assay methodologies have evolved over recent decades.
2. **Sample Size of External Hospital Clinic**: The external hospital validation cohort from Balıkesir University Hospital comprises $N = 319$ patients ($N = 48$ in the stratified test split). While it represents real-world clinical practice with verified imaging ground truth, the relatively small sample size results in wider confidence intervals on clinic-specific performance estimates.
3. **Cross-Population and Setting Heterogeneity**: The external hospital cohort (symptomatic Turkish outpatient clinic attendees) and the US survey cohorts (NHANES III and modern NHANES) differ substantially across genetic backgrounds, dietary habits, healthcare access, and clinical acuity. While these differences offer a stringent test of transportability, they also introduce compound distribution shifts that cannot be disentangled into isolated demographic versus laboratory components.
4. **Ultrasound as an Imperfect Reference Standard**: Although real-time transabdominal ultrasonography is the established clinical reference standard for cholelithiasis (with sensitivity typically between 85% and 95%), it remains an operator-dependent imaging technique. Factors such as patient body habitus, abdominal gas, and gallstone size (e.g., micro-lithiasis or biliary sludge $< 2$ mm) can cause false-negative or inconclusive sonographic reads.
5. **Observational Cross-Sectional Design**: All evaluated datasets are cross-sectional. Consequently, empirical model associations and feature perturbation analyses demonstrate mathematical prediction sensitivity rather than causal pathophysiological mechanisms. Model responses under biomarker changes should not be interpreted as interventional risk reductions.
6. **Lack of Longitudinal Natural History Follow-up**: The available datasets do not capture longitudinal patient trajectories. The models predict the cross-sectional presence of active gallstones, but cannot distinguish between indolent, permanently asymptomatic calculi and stones that will progress to acute cholecystitis, choledocholithiasis, or biliary pancreatitis.
7. **Need for Contemporary Prospective Validation**: Independent prospective validation on contemporary, multi-center cohorts with concurrent point-of-care ultrasound (POCUS) imaging is necessary before any clinical translation or workflow deployment.
8. **Exploratory Decision Curve Analysis**: While Decision Curve Analysis demonstrates theoretical net benefit across 5% to 30% decision thresholds, DCA represents an exploratory decision-analytic model. It does not replace prospective pragmatic clinical trials assessing whether AI-assisted triage improves diagnostic time, cost-effectiveness, or health outcomes in primary care."""

        content = content.replace("## 7. Computational Environment & Reproducibility", limitations_section + "\n\n---\n\n## 7. Computational Environment & Reproducibility")

    # 13. Ensure Figure list and Table list in Data & Code Availability are updated
    old_avail_search = "## Data and Code Availability"
    avail_idx = content.find(old_avail_search)
    ref_idx = content.find("## References")
    
    if avail_idx != -1 and ref_idx != -1:
        new_availability_block = """## Data and Code Availability

### Primary Figures
- **Figure 1**: Cohort Flow and Stratified Partitioning Architecture (`PAPER.md` Section 2.1)
- **Figure 2**: Standardized Baseline Clinical Biomarker Divergence (`PAPER.md` Table 1)
- **Figure 3**: In-Domain Discrimination — ROC and Precision-Recall Curves (`plots/roc_pr_curves_combined.png`)
- **Figure 4**: Five-Panel Calibration Curves Across Benchmark Paradigms (`plots/calibration_curves_5panel.png`)
- **Figure 5**: Bidirectional Cross-Domain Transportability and Asymmetry (`plots/ultrasound_benchmark_summary.png`)
- **Figure 6**: Feature Sensitivity Analyses and Biological Domain Ablation (`PAPER.md` Tables 8 & 9)
- **Figure 7**: Decision Curve Analysis Net Clinical Benefit across 5–30% Thresholds (`plots/decision_curve_analysis.png`)
- **Figure 8**: Error Quadrant Profiles and Phenotypic Biomarker Signatures (`PAPER.md` Table 10)

### Primary Tables
- **Table 1**: Baseline Characteristics of NHANES III Physical Ultrasonography Cohort ($N = 12,824$)
- **Table 2**: Missingness Profile Across Harmonized Features in Development and Validation Cohorts
- **Table 3**: Experiment 1 — In-Domain Physical Ultrasound Ground-Truth Benchmark ($N = 12,824$)
- **Table 4**: Direct Physical Ultrasound vs. Patient Questionnaire Recall (The Silent Gallstone Paradox, $N = 13,694$)
- **Table 5**: Experiment 2 — Modern Population Survey Label Benchmark (CDC NHANES 2017–2020, $N = 9,210$)
- **Table 6**: Experiment 4 — Bidirectional Cross-Domain Generalization Benchmark (15 Harmonized Features)
- **Table 7**: Subgroup Performance across Demographic and Clinical Strata with 95% CIs ($N = 1,924$)
- **Table 8**: Sensitivity Analyses on Held-Out Physical Ultrasound Test Split ($N = 1,924$)
- **Table 9**: Feature Domain Ablation Study on Physical Ultrasound Test Split ($N = 1,924$)
- **Table 10**: Error Analysis Across Prediction Quadrants ($N = 1,924$)

### Source Code and Benchmark Assets
- **Harmonized Cohorts**: `data/gallstone_.csv` ($N=319$), `data/nhanes_gallstone.csv` ($N=9,210$), `data/nhanes3_ultrasound.csv` ($N=12,824$)
- **Core Scripts**: `src/model.py`, `src/prepare_nhanes3.py`, `src/benchmark_ultrasound.py`, `src/compute_all_scientific_analyses.py`, `src/compute_models_and_figures.py`
- **Benchmark Metrics**: `models/ultrasound_benchmarks.json`, `models/additional_benchmarks.json`, `models/paper_scientific_tables.json`

---

"""
        content = content[:avail_idx] + new_availability_block + content[ref_idx:]

    with open("PAPER.md", "w", encoding="utf-8") as f:
        f.write(content)
    with open("paper", "w", encoding="utf-8") as f:
        f.write(content)

    print("PAPER.md and paper successfully updated and synchronized.")

if __name__ == "__main__":
    generate_paper()
