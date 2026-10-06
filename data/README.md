# Data Provenance, Acquisition, and Governance Policy

This research benchmark evaluates predictive machine learning models across **three distinct epidemiological paradigms** encompassing $N = 22,353$ participants. In accordance with clinical data governance and ethics standards, **raw patient-level clinical data files are not redistributed directly within this repository**.

All analyses are secondary research on publicly accessible, legally obtainable, and fully de-identified data sources. Researchers can reproduce all datasets and preprocessing pipelines by following the step-by-step instructions below.

---

## 📁 Expected Directory Structure

After downloading the raw sources and executing the extraction scripts, your local `data/` directory should have the following structure:

```text
data/
├── README.md                          # This provenance guide
├── .gitkeep
│
├── gallstone_.csv                     # UCI Turkish Clinic cohort (N = 319)
├── nhanes3_ultrasound.csv             # Processed NHANES III physical ultrasound cohort (N = 12,824)
├── nhanes_gallstone.csv               # Processed NHANES 2017–2020 survey cohort (N = 9,210)
│
├── nhanes3/                           # Raw CDC NHANES III survey files
│   ├── adult.dat                      # Adult questionnaire raw ASCII fixed-width data
│   ├── exam.dat                       # Examination raw ASCII fixed-width data (ultrasound)
│   ├── lab.dat                        # Laboratory raw ASCII fixed-width data (chemistry)
│   └── HGUHS.xpt                      # NHANES III Gallbladder Ultrasound Examination file
│
└── nhanes/                            # Raw CDC NHANES 2017–2020 pre-pandemic SAS XPT files
    ├── P_DEMO.XPT                     # Demographics
    ├── P_BMX.XPT                      # Body measures (height, weight, BMI)
    ├── P_GLU.XPT                      # Fasting glucose
    ├── P_TCHOL.XPT                    # Total cholesterol
    ├── P_HDL.XPT                      # HDL cholesterol
    ├── P_TRIGLY.XPT                   # Triglycerides & LDL
    ├── P_BIOPRO.XPT                   # Standard biochemistry profile (AST, ALT, ALP, etc.)
    ├── P_CBC.XPT                      # Complete blood count
    ├── P_GHB.XPT                      # Glycohemoglobin
    └── P_MCQ.XPT                      # Medical conditions questionnaire (MCQ550, MCQ560)
```

---

## 1. Hospital Diagnostic Clinic Cohort (Balıkesir University Hospital / UCI)

- **Dataset Name**: Cholelithiasis Prediction Dataset from Bioelectrical Impedance and Clinical Laboratory Biomarkers
- **Original Source**: Outpatient internal medicine and gastroenterology clinics of Balıkesir University Hospital, Balıkesir, Turkey.
- **Repository**: UCI Machine Learning Repository (Donated 2023).
- **Accession DOI**: [10.24432/C57H08](https://doi.org/10.24432/C57H08)
- **URL**: [https://archive.ics.uci.edu/dataset/892/cholelithiasis+prediction+dataset](https://archive.ics.uci.edu/dataset/892/cholelithiasis+prediction+dataset)
- **Licensing & Terms**: Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Sample Size**: $N = 319$ adult outpatients (158 active gallstone cases, 161 stone-free controls; prevalence 49.5%).
- **Reference Standard**: Real-time diagnostic transabdominal ultrasonography (Siemens Sonoline G50, 3.5 MHz curved array transducer).
- **How to Obtain**:
  1. Download the data archive from the UCI Machine Learning Repository link above.
  2. Extract the CSV file and place it at: `data/gallstone_.csv`.

---

## 2. Population Ultrasonography Ground-Truth Cohort (CDC NHANES III, 1988–1994)

- **Dataset Name**: Third National Health and Nutrition Examination Survey (NHANES III) — Gallbladder Ultrasonography & Laboratory Examination
- **Original Source**: Centers for Disease Control and Prevention (CDC) / National Center for Health Statistics (NCHS), Hyattsville, Maryland, USA.
- **URL**: [https://wwwn.cdc.gov/nchs/nhanes/nhanes3/default.aspx](https://wwwn.cdc.gov/nchs/nhanes/nhanes3/default.aspx)
- **Licensing & Terms**: Public domain (U.S. Government work). De-identified public-use data file.
- **Sample Size**: $N = 12,824$ adult participants with verified gallbladder acoustic status after excluding prior surgical cholecystectomy ($N = 870$) and non-visualized gallbladders ($N = 612$). Active gallstone prevalence: 9.0% (1,158 active cases / 11,666 normal controls).
- **Reference Standard**: Standardized real-time transabdominal ultrasonography (Toshiba SSA-90A console, 3.75 MHz convex sector transducer; primary diagnostic code `GUPFDX1R`).
- **How to Obtain**:
  1. Download the NHANES III Examination (`exam.dat`), Laboratory (`lab.dat`), and Adult Interview (`adult.dat`) data files from the CDC NCHS data portal:
     - Examination data: `https://ftp.cdc.gov/pub/Health_Statistics/NCHS/nhanes/nhanes3/1A/exam.dat`
     - Laboratory data: `https://ftp.cdc.gov/pub/Health_Statistics/NCHS/nhanes/nhanes3/1A/lab.dat`
     - Adult Interview data: `https://ftp.cdc.gov/pub/Health_Statistics/NCHS/nhanes/nhanes3/1A/adult.dat`
  2. Place these files inside `data/nhanes3/`.
  3. Run the automated fixed-width parser and feature extraction pipeline:
     ```bash
     python -m src.prepare_nhanes3
     ```
     This generates `data/nhanes3_ultrasound.csv`.

---

## 3. Modern Population Surveillance Survey Cohort (CDC continuous NHANES 2017–2020 Pre-Pandemic)

- **Dataset Name**: Continuous National Health and Nutrition Examination Survey (NHANES) 2017–2020 Pre-Pandemic Examination & Laboratory Data
- **Original Source**: CDC / NCHS, Hyattsville, Maryland, USA.
- **URL**: [https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?Cycle=2017-2020](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?Cycle=2017-2020)
- **Licensing & Terms**: Public domain (U.S. Government work). Fully de-identified public-use data file.
- **Sample Size**: $N = 9,210$ civilian non-institutionalized adults aged $\ge 20$ who completed the MEC examination and health questionnaires. Recall prevalence: 10.8% (994 survey-positive / 8,216 negative).
- **Target Label**: Self-reported questionnaire recall (`MCQ550`: "Has a doctor or other health professional ever told you that you had gallstones?"). Crucially, query `MCQ560` reveals that 74.6% (742 / 994) of survey-positive respondents had undergone prior surgical cholecystectomy.
- **How to Obtain**:
  1. Download the pre-pandemic 2017–2020 SAS transport (`.XPT`) files from the CDC website into `data/nhanes/`:
     - Demographics: `P_DEMO.XPT`
     - Body Measures: `P_BMX.XPT`
     - Medical Conditions: `P_MCQ.XPT`
     - Fasting Glucose: `P_GLU.XPT`
     - Total Cholesterol: `P_TCHOL.XPT`
     - HDL Cholesterol: `P_HDL.XPT`
     - Triglycerides: `P_TRIGLY.XPT`
     - Biochemistry Profile: `P_BIOPRO.XPT`
     - Complete Blood Count: `P_CBC.XPT`
     - Glycohemoglobin: `P_GHB.XPT`
  2. Run the automated merger script:
     ```bash
     python -m src.prepare_nhanes
     ```
     This produces `data/nhanes_gallstone.csv`.

---

## ⚖️ Ethics and Data Governance Statement

- **Secondary Research**: This study is an observational secondary analysis of previously collected, fully anonymized, and publicly accessible data.
- **No Primary Patient Contact**: The researchers had no direct interaction with study participants and did not collect primary biospecimens or perform sonographic examinations.
- **IRB Exemption**: Under U.S. Federal Policy for the Protection of Human Subjects (45 CFR 46.104(d)(4)), secondary analysis of publicly available, de-identified datasets does not constitute human subjects research requiring independent institutional review board (IRB) approval.
- **Patient Privacy**: No individual identifiers (e.g., names, dates of birth, social security numbers, medical record numbers, or precise geographic identifiers) are contained in the processed analytical files.
