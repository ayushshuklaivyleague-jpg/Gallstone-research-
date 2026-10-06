"""
Download NHANES 2017–March 2020 Pre-Pandemic Data Files for Gallstone Analysis.

Note on CDC directory conventions:
  The CDC publishes the continuous NHANES 2017–March 2020 pre-pandemic cycle files
  using the "P_" prefix (e.g., P_DEMO.XPT, P_BIOPRO.XPT). In the CDC archive hierarchy,
  these files are hosted under the 2017 data files directory:
  https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/
"""
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

failed_files = []

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
        failed_files.append((filename, str(e)))

if failed_files:
    print(f"\n❌ ERROR: {len(failed_files)} files failed to download:")
    for f, err in failed_files:
        print(f"   • {f}: {err}")
    sys.exit(1)

print("\n✅ All required NHANES 2017–2020 files are verified and ready.")
