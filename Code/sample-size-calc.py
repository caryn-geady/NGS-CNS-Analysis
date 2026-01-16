# load in register.csv and take the 22 ids from the second column
import pandas as pd
def load_patient_ids(path,col_idx=1):
    df = pd.read_csv(path)
    patient_ids = df.iloc[:, col_idx].dropna().drop_duplicates().tolist()
    return patient_ids

cohort2_ids = load_patient_ids('register.csv')

# load in results.csv and take the unique ids from the first column (USUBJID)
cohort1_ids = load_patient_ids('results.csv', col_idx=0)