import pandas as pd

uci = pd.read_csv('data/gallstone_.csv')
print('=== UCI Lab Medians ===')
for col in ['Glucose', 'Total Cholesterol (TC)', 'High Density Lipoprotein (HDL)', 'Low Density Lipoprotein (LDL)', 'Triglyceride', 'Aspartat Aminotransferaz (AST)', 'Alanin Aminotransferaz (ALT)', 'Alkaline Phosphatase (ALP)', 'Creatinine', 'Hemoglobin (HGB)']:
    print(f'{col:35s}: median = {uci[col].median():.1f}, min = {uci[col].min():.1f}, max = {uci[col].max():.1f}')

biopro = pd.read_sas('data/nhanes/P_BIOPRO.XPT')
tchol = pd.read_sas('data/nhanes/P_TCHOL.XPT')
hdl = pd.read_sas('data/nhanes/P_HDL.XPT')
trig = pd.read_sas('data/nhanes/P_TRIGLY.XPT')
cbc = pd.read_sas('data/nhanes/P_CBC.XPT')
print('\n=== NHANES Lab Medians ===')
print('Glucose (LBXSGL)                   : median =', biopro['LBXSGL'].median())
print('Total Cholesterol (LBXTC)          : median =', tchol['LBXTC'].median())
print('HDL (LBDHDD)                       : median =', hdl['LBDHDD'].median())
print('LDL (LBDLDL)                       : median =', trig['LBDLDL'].median())
print('Triglycerides (LBXTR)              : median =', trig['LBXTR'].median())
print('AST (LBXSASSI)                     : median =', biopro['LBXSASSI'].median())
print('ALT (LBXSATSI)                     : median =', biopro['LBXSATSI'].median())
print('ALP (LBXSAPSI)                     : median =', biopro['LBXSAPSI'].median())
print('Creatinine (LBXSCR)                : median =', biopro['LBXSCR'].median())
print('Hemoglobin (LBXHGB)                : median =', cbc['LBXHGB'].median())
