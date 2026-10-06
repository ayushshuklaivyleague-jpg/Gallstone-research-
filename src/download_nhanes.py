"""Download NHANES 2017-2020 data files for gallstone analysis."""
import os
import sys
import urllib.request

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles"
FILES = {
    "P_MCQ.XPT":    "Medical Conditions Questionnaire (gallstone Q)",
    "P_DEMO.XPT":   "Demographics (age, gender, ethnicity)",
    "P_BMX.XPT":    "Body Measures (BMI, waist, height, weight)",
    "P_BIOPRO.XPT": "Biochemistry Profile (liver enzymes, creatinine, glucose)",
    "P_TCHOL.XPT":  "Total Cholesterol",
    "P_HDL.XPT":    "HDL Cholesterol",
    "P_TRIGLY.XPT": "Triglycerides & LDL",
    "P_GHB.XPT":    "Glycohemoglobin (HbA1c)",
    "P_CBC.XPT":    "Complete Blood Count (hemoglobin)",
    "P_GLU.XPT":    "Plasma Fasting Glucose",
}

outdir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "nhanes")
os.makedirs(outdir, exist_ok=True)

for filename, desc in FILES.items():
    url = f"{BASE}/{filename}"
    outpath = os.path.join(outdir, filename)
    if os.path.exists(outpath):
        print(f"  [skip] {filename} already exists")
        continue
    print(f"  Downloading {filename:15s} -- {desc}")
    try:
        urllib.request.urlretrieve(url, outpath)
        size_kb = os.path.getsize(outpath) / 1024
        print(f"           -> {size_kb:.0f} KB")
    except Exception as e:
        print(f"           -> FAILED: {e}")

print("\nDone!")
