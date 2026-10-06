"""
src/prepare_nhanes3.py — Ingestion, Extraction, and Harmonization of NHANES III Gallbladder Ultrasound Cohort.

Extracts ground-truth transabdominal ultrasound findings (sonographer + radiologist adjudication)
and merges them with matching laboratory chemistry, demographic, and physical examination data.

Targets extracted:
  • target_active_us: 1 if physical active gallstones confirmed on ultrasound (FDX 02, 03, 04, 05),
                      0 if normal gallbladder (FDX 01). Excludes absent/inconclusive.
  • target_cholecystectomy_us: 1 if surgically absent gallbladder confirmed on ultrasound (FDX 07, 08),
                               0 if normal gallbladder (FDX 01).
  • target_total_us: 1 if active stones OR cholecystectomy, 0 if normal.
  • target_self_report: 1 if doctor ever told participant had gallstones (HAJ9 == 1), 0 if (HAJ9 == 2).

Harmonized features:
  • Age, Gender, Height, Weight, BMI
  • Total Cholesterol, Triglycerides, HDL, LDL, Glucose
  • AST, ALT, ALP, Creatinine, Total Bilirubin, CRP
  • Gallbladder Wall Thickness (mm)
"""

import os
import sys
import numpy as np
import pandas as pd

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
NH3_DIR = os.path.join(DATA_DIR, "nhanes3")
OUT_CSV = os.path.join(DATA_DIR, "nhanes3_ultrasound.csv")


def parse_clean_float(text):
    val = text.strip()
    if not val or val in [".", "-", "888", "999", "8888", "9999", "88888", "99999", "888888", "999999"]:
        return np.nan
    try:
        return float(val)
    except:
        return np.nan


def run_extraction():
    exam_path = os.path.join(NH3_DIR, "exam.dat")
    lab_path = os.path.join(NH3_DIR, "lab.dat")
    adult_path = os.path.join(NH3_DIR, "adult.dat")

    for p in [exam_path, lab_path, adult_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required file: {p}")

    print("Phase 1: Parsing Adult Interview questionnaire (HAJ9, HAJ12)...")
    adult_records = {}
    with open(adult_path, "r", encoding="latin1") as f:
        for line in f:
            if len(line) < 1850:
                continue
            seqn = int(line[0:5])
            h9 = line[1839:1840].strip()
            h12 = line[1843:1844].strip()
            
            sr_gallstone = 1 if h9 == "1" else (0 if h9 == "2" else np.nan)
            sr_surgery = 1 if h12 == "1" else (0 if h12 == "2" else np.nan)
            adult_records[seqn] = (sr_gallstone, sr_surgery)

    print(f"  Loaded {len(adult_records):,} adult interview records.")

    print("Phase 2: Parsing Examination file (Demographics, Body Measures, Ultrasound)...")
    exam_records = {}
    with open(exam_path, "r", encoding="latin1") as f:
        for line in f:
            if len(line) < 5500:
                continue
            seqn = int(line[0:5])
            sex_raw = line[14:15].strip()
            age_raw = line[15:17].strip()
            wt_raw = line[1507:1513].strip()
            bmi_raw = line[1523:1527].strip()
            ht_raw = line[1527:1532].strip()
            wall_raw = line[5459:5461].strip()
            tdx = line[5478:5480].strip()
            fdx = line[5490:5492].strip()

            # Target filtering: keep evaluate-able ultrasound outcomes
            if fdx not in ["01", "02", "03", "04", "05", "07", "08"]:
                continue

            gender = 0 if sex_raw == "1" else (1 if sex_raw == "2" else np.nan)
            age = int(age_raw) if age_raw else np.nan
            weight = parse_clean_float(wt_raw)
            height = parse_clean_float(ht_raw)
            
            # Direct BMI computation or parse fallback
            if pd.notna(weight) and pd.notna(height) and height > 40:
                bmi = weight / ((height / 100.0) ** 2)
            else:
                bmi = parse_clean_float(bmi_raw)

            gb_wall = parse_clean_float(wall_raw)

            # Define clinical outcomes
            is_active = 1 if fdx in ["02", "03", "04", "05"] else 0
            is_chole = 1 if fdx in ["07", "08"] else 0
            is_total = 1 if (is_active or is_chole) else 0

            exam_records[seqn] = {
                "SEQN": seqn,
                "Gender": gender,
                "Age": age,
                "Weight": weight,
                "Height": height,
                "BMI": bmi,
                "GB_Wall_Thickness": gb_wall,
                "Ultrasound_Code": fdx,
                "target_active_us": is_active,
                "target_cholecystectomy_us": is_chole,
                "target_total_us": is_total,
            }

    print(f"  Loaded {len(exam_records):,} evaluate-able ultrasound examinees.")

    print("Phase 3: Parsing Laboratory file (Chemistry, Liver enzymes, Lipids)...")
    lab_records = {}
    with open(lab_path, "r", encoding="latin1") as f:
        for line in f:
            if len(line) < 1850:
                continue
            seqn = int(line[0:5])
            if seqn not in exam_records:
                continue

            tc = parse_clean_float(line[1597:1600])
            tg = parse_clean_float(line[1605:1609])
            ldl = parse_clean_float(line[1614:1617])
            hdl = parse_clean_float(line[1621:1624])
            crp = parse_clean_float(line[1666:1671])
            glu = parse_clean_float(line[1757:1760])
            tbp = parse_clean_float(line[1773:1777])
            cep = parse_clean_float(line[1783:1787])
            ast = parse_clean_float(line[1820:1823])
            alt = parse_clean_float(line[1823:1826])
            alp = parse_clean_float(line[1834:1838])

            lab_records[seqn] = {
                "Total Cholesterol": tc,
                "Triglyceride": tg,
                "LDL": ldl,
                "HDL": hdl,
                "CRP": crp,
                "Glucose": glu,
                "Total Bilirubin": tbp,
                "Creatinine": cep,
                "AST": ast,
                "ALT": alt,
                "ALP": alp,
            }

    print(f"  Matched {len(lab_records):,} examinees with laboratory profiles.")

    print("Phase 4: Merging into consolidated analytical cohort...")
    combined = []
    for seqn, edata in exam_records.items():
        ldata = lab_records.get(seqn, {})
        sr_gallstone, sr_surgery = adult_records.get(seqn, (np.nan, np.nan))
        
        row = dict(edata)
        row.update(ldata)
        row["target_self_report"] = sr_gallstone
        row["self_report_surgery"] = sr_surgery
        combined.append(row)

    df = pd.DataFrame(combined)

    # Reorder columns logically
    meta_cols = ["SEQN", "Ultrasound_Code", "target_active_us", "target_cholecystectomy_us", "target_total_us", "target_self_report", "self_report_surgery"]
    feature_cols = [
        "Age", "Gender", "Height", "Weight", "BMI",
        "Glucose", "Total Cholesterol", "LDL", "HDL", "Triglyceride",
        "AST", "ALT", "ALP", "Creatinine", "Total Bilirubin", "CRP", "GB_Wall_Thickness"
    ]
    df = df[meta_cols + feature_cols]

    print("\nDataset Construction Summary:")
    print(f"  Total Valid Ultrasound Cohort: {len(df):,}")
    print(f"  Active Ultrasound Gallstones:  {df['target_active_us'].sum():,} ({df['target_active_us'].mean()*100:.2f}%)")
    print(f"  Ultrasound Cholecystectomy:   {df['target_cholecystectomy_us'].sum():,} ({df['target_cholecystectomy_us'].mean()*100:.2f}%)")
    print(f"  Total Gallstone Disease (US): {df['target_total_us'].sum():,} ({df['target_total_us'].mean()*100:.2f}%)")
    print(f"  Self-Report Gallstone Recall: {df['target_self_report'].sum():,} ({df['target_self_report'].mean()*100:.2f}%)")

    # Clean active-only comparison cohort: Exclude cholecystectomy so comparison is active stones (1) vs normal (0)
    active_df = df[df["target_cholecystectomy_us"] == 0].copy()
    print(f"\nActive Ultrasound vs Normal Gallbladder Cohort:")
    print(f"  Total N: {len(active_df):,}")
    print(f"  Active Stones: {active_df['target_active_us'].sum():,} ({active_df['target_active_us'].mean()*100:.2f}%)")
    print(f"  Normal Controls: {(active_df['target_active_us']==0).sum():,} ({(active_df['target_active_us']==0).mean()*100:.2f}%)")

    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved full ultrasound cohort to: {OUT_CSV} ({os.path.getsize(OUT_CSV):,} bytes)")

    return df


if __name__ == "__main__":
    run_extraction()
