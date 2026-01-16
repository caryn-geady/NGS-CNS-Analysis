import pandas as pd

radFile = 'ngs-composite-radiomics.csv'
cnsKey = 'mrn-cns-key.csv'
ngsKey = 'mrn-ngs-key.csv'

# load the data
radiomics = pd.read_csv(radFile) # from Caryn
cns_mapping = pd.read_csv(cnsKey) # from Kevin
ngs_mapping = pd.read_csv(ngsKey) # from Tony

# in cns_mapping, replace string pattern 'OC' with 'C' in the AnonMrn column
cns_mapping['AnonMrn'] = cns_mapping['AnonMrn'].str.replace('OC', 'C', regex=False)

# run the unify_keys logic
cns_mapping.to_csv('cns_mapping_processed.csv', index=False)  

# %%

uniMapping = pd.read_csv('unified_mapping.csv')

eligiblePatients = radiomics['USUBJID'].isin(uniMapping['NGS_ID'])