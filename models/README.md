# Trained Models & Benchmark Artifacts

This directory contains saved model checkpoints, benchmark evaluation logs, and pre-computed statistical tables for the Gallstone AI research benchmark.

---

## 📦 Checkpoint Files (`.pt`)

| File Name | Architecture | Input Dim | Training Regime | Validation Purpose |
|:---|:---|:---:|:---|:---|
| `best_model.pt` | PyTorch GallstoneNet | 38 | UCI Hospital Clinic | Best overall validation loss checkpoint |
| `exp_a_uci_clinic_gallstonenet.pt` | PyTorch GallstoneNet | 38 | Exp A: UCI Hospital Clinic ($N=319$) | In-domain high-acuity clinical referral benchmark |
| `exp_b_nhanes_full_gallstonenet.pt` | PyTorch GallstoneNet | 28 | Exp B: Modern NHANES ($N=9,210$) | Surveillance survey recall benchmark (`MCQ550`) |
| `exp_c_cross_external_gallstonenet.pt` | PyTorch GallstoneNet | 20 | Exp C: NHANES $\to$ UCI Transfer | Cross-cohort domain transfer checkpoint |
| `exp_d_joint_harmonized_gallstonenet.pt` | PyTorch GallstoneNet | 20 | Exp D: Joint Multi-Cohort ($N=9,529$) | Harmonized multi-cohort pooled model |

*Note: All checkpoints store the PyTorch `state_dict`, input dimension, feature names, and pre-computed in-split scaler and imputer statistics for leak-free downstream inference.*

---

## 📊 Pre-Computed Scientific JSON Artifacts

- **`ultrasound_benchmarks.json`**: Primary experimental results on the Physical Ultrasound Ground-Truth Cohort ($N = 12,824$, Exp E1, E2, E3). Contains point estimates, 1,000-sample bootstrap 95% confidence intervals, Brier scores, calibration slopes, intercepts, and confusion matrices across GallstoneNet, XGBoost, LightGBM, Random Forest, and Super Ensemble.
- **`multi_cohort_benchmarks.json`**: Benchmark metrics across Experiments A, B, C, and D.
- **`additional_benchmarks.json`**: Subgroup performance stratified by Age, Sex, BMI; Sensitivity analyses (withholding CRP, liver enzymes, demographics, LDL); and Feature domain ablations.
- **`paper_scientific_tables.json`**: Formatted Table 1 (Baseline Clinical Characteristics with SMD and Welch $t$-test / Chi-square $p$-values) and Table 2 (Feature Missingness Profile).
