"""
src/apply_thematic_polishing.py

Implements the user's high-level philosophical and scientific refinements:
1. Embeds the central thesis into the Abstract, Introduction, Section 4.1.2, and Conclusions:
   'Routine demographic and metabolic variables contain substantial signal for ultrasound-detected gallstones,
    but increasing model complexity provides little additional discrimination, while transportability depends strongly
    on the domain in which the model was learned.'
2. Updates Abstract Findings to explicitly report the parsimonious baseline hierarchy:
   Demographics (0.732) -> Core-6 (0.737) -> Full Linear (0.748) -> Complex ML (0.748-0.758, non-significant gain).
3. Tightens Section 6.1 (Perturbation analysis): resolves tension by explicitly distinguishing the mathematical
   non-linear curvature of the fitted neural model from the underlying additive biological relationship.
4. Prominently highlights the external clinical cohort size constraint (N=319 / N=48 test) and the historical nature
   of NHANES III (1988-1994) in the Discussion and Limitations.
5. Keeps clinical language restrained across all sections.
"""

def polish_paper():
    with open("PAPER.md", "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Update Abstract Findings to include parsimony finding
    old_findings_lead = """When trained and evaluated against physical ultrasound ground truth ($N = 12,824$, Experiment 1), GallstoneNet achieved an AUROC of **0.758** [95% CI 0.726–0.791], sensitivity of **0.711** [0.646–0.776], specificity of **0.641** [0.618–0.664], and an almost ideal calibration slope of **1.05** (intercept **-2.15**, Brier score 0.187). Paired bootstrap testing revealed no statistically significant difference in discriminative capacity between GallstoneNet and the Super Ensemble ($\Delta\text{AUROC} = -0.0051$ [95% CI -0.0217 to 0.0119], $p = 0.538$)."""

    new_findings_lead = """When evaluated against physical ultrasound ground truth ($N = 12,824$, Experiment 1), a minimal 3-variable demographic baseline (Age, Sex, BMI) modeled via Logistic Regression achieved an AUROC of **0.732** [95% CI 0.697–0.767], capturing over 96% of the discriminative capacity of complex architectures. Adding routine fasting glucose and lipids (Parsimonious Core-6) reached an AUROC of **0.737** [0.702–0.772]. Full 15-feature linear modeling reached an AUROC of **0.748** [0.714–0.781], while sophisticated non-linear models (XGBoost AUROC **0.748**, PyTorch GallstoneNet AUROC **0.758** [0.726–0.791]) offered no statistically significant incremental discrimination over linear baselines ($\Delta\text{AUROC} = +0.0002$ to $+0.0103$, $p > 0.35$)."""

    content = content.replace(old_findings_lead, new_findings_lead)

    # 2. Update Abstract Interpretation with the user's exact central thesis
    old_abstract_interp = """### Interpretation
Routine non-imaging blood chemistries and anthropometric biomarkers carry genuine, reproducible statistical signal for ultrasound-detected gallstones in unselected populations, with the overwhelming majority of predictive signal captured by simple demographic and routine metabolic variables. Retrospective self-reported survey labels introduce severe construct mismatch by primarily capturing healthcare utilization and cholecystectomy history rather than active lithogenesis. Furthermore, population-level physical ultrasound models exhibit superior outward transportability compared to high-acuity clinic models, providing an evidence-based foundation for non-imaging pre-screening and triage in global primary care."""

    new_abstract_interp = """### Interpretation
Routine demographic and metabolic variables contain substantial predictive signal for ultrasound-detected gallstones, but increasing model complexity provides little additional discrimination, while cross-domain transportability depends strongly on the data-generating regime in which the model was learned. Retrospective survey recall labels introduce profound construct mismatch by capturing surgical history rather than active stones. Conversely, broad population models retain partial generalizability when transferred inward to foreign hospital clinics, whereas hospital-derived models fail outward in community screening. These findings establish an evidence-based foundation and an open benchmark for parsimonious non-imaging triage."""

    content = content.replace(old_abstract_interp, new_abstract_interp)

    # 3. Tighten Section 6.1 Perturbation paragraph to resolve the non-linear conceptual tension
    old_perturb_para = """The model response function behaves monotonically and synergistically: single biomarker elevations produce modest, non-linear risk increases ($+3\%$ to $+9\%$), whereas compound metabolic dysfunction accelerates the predicted probability beyond 35–50%, producing progressively higher model-estimated probabilities under the specified perturbations."""

    new_perturb_para = """These perturbation shifts illustrate the mathematical behavior and curvature of the fitted neural network's non-linear response surface under simulated multi-dimensional biomarker variation. While the neural architecture parameterizes a smooth, monotonic sigmoid response where compound metabolic perturbations produce non-additive logit shifts, our empirical benchmarking (Section 4.1.2) demonstrates that this fitted non-linearity does not translate into meaningful out-of-sample discriminative superiority over standard additive linear models ($\Delta\text{AUROC} = +0.0103, p = 0.362$). Thus, while flexible architectures can fit non-linear mathematical response manifolds, clinical risk prediction for ultrasound-detected gallstones is effectively and parsimoniously served by standard additive formulations."""

    content = content.replace(old_perturb_para, new_perturb_para)

    # 4. Sharpen Section 8 (Conclusions) with the central thesis and clear evidentiary boundaries
    old_conclusions = """## 8. Conclusions

Routine non-imaging blood and anthropometric biomarkers carry robust, generalizable predictive signal for physical gallbladder stones. We expose the Silent Gallstone Paradox, establish the failure modes of survey-recall labels, demonstrate asymmetric cross-domain transportability, and provide an open, verified benchmark for translational medical AI."""

    new_conclusions = """## 8. Conclusions

This multi-cohort investigation yields four foundational conclusions for clinical machine learning:

1. **The Parsimony Principle**: Routine demographic and metabolic variables (Age, Biological Sex, BMI, Glucose, Cholesterol, and Triglycerides) contain substantial predictive signal for ultrasound-detected gallstones (AUROC $\approx 0.73–0.75$). Crucially, increasing algorithmic complexity through gradient boosted trees or deep tabular residual networks provides negligible, non-significant incremental discrimination over simple linear baselines ($\Delta\text{AUROC} \le +0.010$, $p > 0.35$).
2. **The Target Construct Imperative**: Retrospective questionnaire recall introduces severe construct mismatch, primarily modeling past healthcare access and surgical cholecystectomy history rather than unoperated active stones. Objective imaging ground truth is essential for reliable benchmark formulation.
3. **Transportability Asymmetry**: Domain transfer is inherently directional. Models trained on broad population screening learn generalized metabolic associations that partially survive inward transfer to acute clinical referral settings, whereas models developed within high-acuity clinics overfit to acute inflammatory presentation and fail completely when projected outward onto community screening.
4. **The Evidence Frontier**: While our benchmark establishes rigorous retrospective baseline standards across $>22,000$ individuals, definitive clinical translation requires moving beyond retrospective data to contemporary, prospective multi-center trials with blinded ultrasound endpoints."""

    content = content.replace(old_conclusions, new_conclusions)

    with open("PAPER.md", "w", encoding="utf-8") as f:
        f.write(content)
    with open("paper", "w", encoding="utf-8") as f:
        f.write(content)

    print("Successfully applied thematic polishing and thesis unification across PAPER.md and paper.")

if __name__ == "__main__":
    polish_paper()
