import sys
import re

sys.stdout.reconfigure(encoding='utf-8')

with open("PAPER.md", "r", encoding="utf-8") as f:
    text = f.read()

# 1. Update Title to Reference-Standard Validation
old_title = "# Predicting Ultrasound-Detected Gallstones from Non-Imaging Clinical Data: Cross-Cohort Generalization and Ground-Truth Validation"
new_title = "# Predicting Ultrasound-Detected Gallstones from Non-Imaging Clinical Data: Cross-Cohort Generalization and Reference-Standard Validation"
assert old_title in text, "Title not found"
text = text.replace(old_title, new_title)

# 2. Fix Table 2 reference in Section 3.3.1 (Reporting conventions and Confirmatory Analyses)
old_t2_rep = "In Table 2 (Abstract) and Table 3 (In-Domain Physical Ultrasound Benchmark), we report the empirical test-set calibration parameters:"
new_t2_rep = "In the Abstract and Table 3 (In-Domain Physical Ultrasound Benchmark), we report the empirical test-set calibration parameters:"
assert old_t2_rep in text, "Reporting conventions text not found"
text = text.replace(old_t2_rep, new_t2_rep)

old_t2_conf = "The primary in-domain model comparison against physical ultrasound ground truth (Table 2) and the prespecified incremental-value hierarchy (Table 4) were pre-specified as primary benchmark hypotheses."
new_t2_conf = "The primary in-domain model comparison against the physical ultrasound reference standard (Table 3) and the prespecified incremental-value hierarchy (Table 4) were pre-specified as primary benchmark hypotheses."
assert old_t2_conf in text, "Confirmatory analyses text not found"
text = text.replace(old_t2_conf, new_t2_conf)

# 3. Soften intercept statement ("precisely counterbalances" -> "directionally consistent")
old_cb = r"The empirical test-set intercept of $-2.147$ precisely counterbalances this training cost-weight, aligning predictions with the natural 9.0% baseline population prevalence"
new_cb = r"The negative calibration intercept is directionally consistent with the upward log-odds shift induced by cost-sensitive training and is broadly compatible with the 9.0% test-set prevalence"
assert old_cb in text, "Intercept text not found"
text = text.replace(old_cb, new_cb)

# 4. Feature accounting clarity in Section 4.5.1
old_exp5_feats = "This pooled experiment expanded from the core 15-variable schema to an **extended 20-variable panel** because the UCI and modern NHANES protocols uniquely shared 5 additional clinically documented variables:"
new_exp5_feats = "This pooled experiment evaluated an **extended 20-variable panel** across both cohorts: the 20-variable panel retained the shared core clinical and biochemical features (excluding CRP, which was not assayed in modern NHANES), incorporated serum hemoglobin, and added five binary comorbidity indicators (comorbidity presence, CAD, hypothyroidism, hyperlipidemia, and diabetes mellitus alongside biological sex):"
assert old_exp5_feats in text, "Exp 5 feature text not found"
text = text.replace(old_exp5_feats, new_exp5_feats)

# 5. Tone down Section 6.3 decision-analytic framework from commands to policy scenarios
old_sec63 = """In rural primary care, community clinics, and low-resource global health settings where diagnostic ultrasound consoles are unavailable or backlogged, this non-imaging model provides an accessible pre-test risk stratification mechanism:
- **Low Risk ($< 10\%$)**: Reassure patient, avoid unnecessary imaging referrals, evaluate alternative dyspeptic causes.
- **Intermediate Risk ($10–35\%$)**: Order elective transabdominal ultrasound, initiate lifestyle and metabolic counseling.
- **High Risk ($> 35\%$)**: Expedite prioritized diagnostic ultrasound or bedside Point-of-Care Ultrasound (POCUS)."""

new_sec63 = """In rural primary care, community clinics, and low-resource global health settings where diagnostic ultrasound consoles are unavailable or backlogged, this non-imaging model provides an accessible pre-test risk stratification mechanism. To illustrate how such a tool could interface with clinical workflows under an exploratory decision-policy framework:
- **Hypothetical Low-Risk Scenario ($< 10\%$)**: In a triage policy, low predicted probability would correspond to standard clinical follow-up without urgent sonographic referral, while evaluating alternative non-biliary causes of dyspepsia.
- **Hypothetical Intermediate-Risk Scenario ($10–35\%$)**: In an elective triage workflow, intermediate probability would correspond to standard outpatient transabdominal ultrasound referral alongside metabolic risk-factor counseling.
- **Hypothetical High-Risk Scenario ($> 35\%$)**: In a prioritization workflow, elevated risk would correspond to prioritized scheduling for diagnostic ultrasound or bedside Point-of-Care Ultrasound (POCUS)."""

assert old_sec63 in text, "Section 6.3 text not found"
text = text.replace(old_sec63, new_sec63)

# 6. Standardize remaining prominent "ground truth" occurrences to "reference standard"
text = text.replace(
    "When trained and evaluated against physical ultrasound ground truth ($N = 12,824$, Experiment 1)",
    "When trained and evaluated against the physical ultrasound reference standard ($N = 12,824$, Experiment 1)"
)
text = text.replace(
    "3. **Population Ultrasonography Ground Truth (CDC NHANES III, $N = 12,824$)**: Representative nationwide cohort with physical, real-time transabdominal ultrasound ground truth (`GUPFDX1R`, active gallstone prevalence: $9.0\%$).",
    "3. **Population Ultrasonography Reference Standard (CDC NHANES III, $N = 12,824$)**: Representative nationwide cohort with direct, real-time transabdominal ultrasound reference standard (`GUPFDX1R`, active gallstone prevalence: $9.0\%$).",
)
text = text.replace(
    'subgraph Regime3["3. Population Ultrasonography Ground Truth (NHANES III)"]',
    'subgraph Regime3["3. Population Ultrasonography Reference Standard (NHANES III)"]'
)
text = text.replace(
    "To establish an uncompromised physical ground truth, we retrieved and harmonized",
    "To establish a direct physical reference standard, we retrieved and harmonized"
)
text = text.replace(
    "Real-time physical imaging ground truth (Experiment 1) is mandatory to evaluate genuine active-disease discrimination.",
    "Real-time physical imaging reference standards (Experiment 1) are mandatory to evaluate genuine active-disease discrimination."
)
text = text.replace(
    "The physical ultrasonography ground truth in NHANES III was collected between 1988 and 1994.",
    "The physical ultrasonography reference standard in NHANES III was collected between 1988 and 1994."
)
text = text.replace(
    "While it represents real-world clinical practice with verified imaging ground truth,",
    "While it represents real-world clinical practice with verified diagnostic ultrasound reference standards,"
)
text = text.replace(
    "Objective imaging ground truth is essential for reliable benchmark formulation.",
    "Objective imaging reference standards are essential for reliable benchmark formulation."
)

# Write to PAPER.md and paper
with open("PAPER.md", "w", encoding="utf-8") as f:
    f.write(text)

with open("paper", "w", encoding="utf-8") as f:
    f.write(text)

print("All 5 cleanups successfully applied to PAPER.md and paper. Length:", len(text))
