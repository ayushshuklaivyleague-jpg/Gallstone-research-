"""
prepare_nhanes.py — Extract, clean, and harmonize NHANES 2017-2020 data for Gallstone research.

Features extracted:
  • Demographics: Age, Gender
  • Target: Gallstone Status (MCQ550: 1=Yes, 2=No -> 1/0)
  • Symptoms: Upper right quadrant abdominal pain (MCQ520: 1=Yes, 2=No -> 1/0)
  • Comorbidities: CAD (MCQ160C), Thyroid problem (MCQ160M), Diabetes (fasting glucose >= 126 or HbA1c >= 6.5)
  • Body Measures: Height, Weight, BMI, Waist Circumference
  • Blood Chemistry / Liver & Kidney:
      ALT, AST, ALP, Total Bilirubin, Albumin, Creatinine, Uric Acid, Fasting Glucose
  • Lipid Panel: Total Cholesterol, HDL, LDL, Triglycerides
  • Hematology: Hemoglobin, White Blood Cells, Platelets
  • Glycemic: HbA1c (Glycohemoglobin)
"""

import os
import sys
import numpy as np
import pandas as pd

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
NHANES_DIR = os.path.join(DATA_DIR, "nhanes")
OUT_CSV = os.path.join(DATA_DIR, "nhanes_gallstone.csv")


def recode_binary_survey(series: pd.Series) -> pd.Series:
    """
    Recode NHANES questionnaire binary responses:
      1.0 -> 1.0 (Yes)
      2.0 -> 0.0 (No)
      7.0 (Refused), 9.0 (Don't Know), NaN -> np.nan
    """
    mapping = {1.0: 1.0, 2.0: 0.0}
    return series.map(mapping)


def process_nhanes():
    print("Loading NHANES XPT files...")
    
    # 1. Questionnaire: Gallstone target & symptoms
    mcq = pd.read_sas(os.path.join(NHANES_DIR, "P_MCQ.XPT"))
    demo = pd.read_sas(os.path.join(NHANES_DIR, "P_DEMO.XPT"))
    bmx = pd.read_sas(os.path.join(NHANES_DIR, "P_BMX.XPT"))
    biopro = pd.read_sas(os.path.join(NHANES_DIR, "P_BIOPRO.XPT"))
    tchol = pd.read_sas(os.path.join(NHANES_DIR, "P_TCHOL.XPT"))
    hdl = pd.read_sas(os.path.join(NHANES_DIR, "P_HDL.XPT"))
    trig = pd.read_sas(os.path.join(NHANES_DIR, "P_TRIGLY.XPT"))
    ghb = pd.read_sas(os.path.join(NHANES_DIR, "P_GHB.XPT"))
    cbc = pd.read_sas(os.path.join(NHANES_DIR, "P_CBC.XPT"))

    # Subset columns
    mcq_sub = mcq[["SEQN", "MCQ550", "MCQ520", "MCQ160C", "MCQ160M"]].copy()
    demo_sub = demo[["SEQN", "RIAGENDR", "RIDAGEYR"]].copy()
    bmx_sub = bmx[["SEQN", "BMXWT", "BMXHT", "BMXBMI", "BMXWAIST"]].copy()
    biopro_sub = biopro[["SEQN", "LBXSATSI", "LBXSASSI", "LBXSAPSI", "LBXSCR", "LBXSTB", "LBXSAL", "LBXSUA", "LBXSGL"]].copy()
    tchol_sub = tchol[["SEQN", "LBXTC"]].copy()
    hdl_sub = hdl[["SEQN", "LBDHDD"]].copy()
    trig_sub = trig[["SEQN", "LBXTR", "LBDLDL"]].copy()
    ghb_sub = ghb[["SEQN", "LBXGH"]].copy()
    cbc_sub = cbc[["SEQN", "LBXHGB", "LBXWBCSI", "LBXPLTSI"]].copy()

    # Merge on SEQN
    df = mcq_sub.merge(demo_sub, on="SEQN", how="inner")
    df = df.merge(bmx_sub, on="SEQN", how="left")
    df = df.merge(biopro_sub, on="SEQN", how="left")
    df = df.merge(tchol_sub, on="SEQN", how="left")
    df = df.merge(hdl_sub, on="SEQN", how="left")
    df = df.merge(trig_sub, on="SEQN", how="left")
    df = df.merge(ghb_sub, on="SEQN", how="left")
    df = df.merge(cbc_sub, on="SEQN", how="left")

    print(f"Total merged records: {len(df)}")

    # Filter for known gallstone status (MCQ550: 1 = Yes, 2 = No)
    df = df[df["MCQ550"].isin([1.0, 2.0])].copy()
    print(f"Records with verified gallstone status: {len(df)}")

    # Standardize column names & encodings
    # Target: 1 = Gallstone present, 0 = Gallstone absent
    df["Gallstone Status"] = (df["MCQ550"] == 1.0).astype(int)

    # Gender: 0 = Male (RIAGENDR=1), 1 = Female (RIAGENDR=2) — matching UCI
    df["Gender"] = (df["RIAGENDR"] == 2.0).astype(int)

    # Demographics
    df["Age"] = df["RIDAGEYR"]

    # Symptoms: Abdominal pain in RUQ (1.0 = Yes, 0.0 = No, NaN = Refused/Unknown/Missing)
    df["Abdominal Pain RUQ"] = recode_binary_survey(df["MCQ520"])

    # Comorbidities
    df["Coronary Artery Disease (CAD)"] = recode_binary_survey(df["MCQ160C"])
    df["Hypothyroidism"] = recode_binary_survey(df["MCQ160M"])
    
    # Diabetes: Positive if fasting glucose >= 126 or HbA1c >= 6.5; np.nan if both missing; else 0.0
    has_dm = (df["LBXSGL"] >= 126.0) | (df["LBXGH"] >= 6.5)
    dm_missing = df["LBXSGL"].isna() & df["LBXGH"].isna()
    df["Diabetes Mellitus (DM)"] = np.where(dm_missing, np.nan, np.where(has_dm, 1.0, 0.0))
    
    # Hyperlipidemia: Positive if TC >= 200 or Trig >= 150 or LDL >= 130; np.nan if all missing; else 0.0
    has_hl = (df["LBXTC"] >= 200.0) | (df["LBXTR"] >= 150.0) | (df["LBDLDL"] >= 130.0)
    hl_missing = df["LBXTC"].isna() & df["LBXTR"].isna() & df["LBDLDL"].isna()
    df["Hyperlipidemia"] = np.where(hl_missing, np.nan, np.where(has_hl, 1.0, 0.0))

    # General Comorbidity flag: 1.0 if any comorbidity is 1.0; np.nan if all comorb fields are missing; else 0.0
    comorb_mat = pd.concat([
        df["Coronary Artery Disease (CAD)"],
        df["Hypothyroidism"],
        df["Diabetes Mellitus (DM)"],
        df["Hyperlipidemia"],
    ], axis=1)
    any_comorb = (comorb_mat == 1.0).any(axis=1)
    all_comorb_nan = comorb_mat.isna().all(axis=1)
    df["Comorbidity"] = np.where(all_comorb_nan, np.nan, np.where(any_comorb, 1.0, 0.0))

    # Anthropometrics
    df["Height"] = df["BMXHT"]
    df["Weight"] = df["BMXWT"]
    df["Body Mass Index (BMI)"] = df["BMXBMI"]
    df["Waist Circumference"] = df["BMXWAIST"]

    # Clinical Labs (Harmonized with UCI)
    df["Glucose"] = df["LBXSGL"]
    df["Total Cholesterol (TC)"] = df["LBXTC"]
    df["High Density Lipoprotein (HDL)"] = df["LBDHDD"]
    df["Low Density Lipoprotein (LDL)"] = df["LBDLDL"]
    df["Triglyceride"] = df["LBXTR"]
    df["Aspartat Aminotransferaz (AST)"] = df["LBXSASSI"]
    df["Alanin Aminotransferaz (ALT)"] = df["LBXSATSI"]
    df["Alkaline Phosphatase (ALP)"] = df["LBXSAPSI"]
    df["Creatinine"] = df["LBXSCR"]
    df["Hemoglobin (HGB)"] = df["LBXHGB"]

    # Additional NHANES markers
    df["Total Bilirubin"] = df["LBXSTB"]
    df["Albumin"] = df["LBXSAL"]
    df["Uric Acid"] = df["LBXSUA"]
    df["HbA1c"] = df["LBXGH"]
    df["White Blood Cells (WBC)"] = df["LBXWBCSI"]
    df["Platelets (PLT)"] = df["LBXPLTSI"]

    # Select final curated columns
    final_cols = [
        "SEQN",
        "Gallstone Status",
        "Gender",
        "Age",
        "Abdominal Pain RUQ",
        "Comorbidity",
        "Coronary Artery Disease (CAD)",
        "Hypothyroidism",
        "Hyperlipidemia",
        "Diabetes Mellitus (DM)",
        "Height",
        "Weight",
        "Body Mass Index (BMI)",
        "Waist Circumference",
        "Glucose",
        "Total Cholesterol (TC)",
        "High Density Lipoprotein (HDL)",
        "Low Density Lipoprotein (LDL)",
        "Triglyceride",
        "Aspartat Aminotransferaz (AST)",
        "Alanin Aminotransferaz (ALT)",
        "Alkaline Phosphatase (ALP)",
        "Creatinine",
        "Hemoglobin (HGB)",
        "Total Bilirubin",
        "Albumin",
        "Uric Acid",
        "HbA1c",
        "White Blood Cells (WBC)",
        "Platelets (PLT)",
    ]

    out_df = df[final_cols].copy()
    out_df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(out_df)} records to {OUT_CSV}")
    print(f"Target breakdown:\n{out_df['Gallstone Status'].value_counts()}")
    print(f"\nMissing value count in final CSV:\n{out_df.isna().sum()}")


if __name__ == "__main__":
    process_nhanes()
