import pandas as pd
import os

# Load the partial key CSV file and rename columns
partial_key = pd.read_csv('NGS CNS - Glioma-anon-key.csv')
partial_key.rename(columns={'Mrn': 'MRN', 'AnonMrn': 'AnonMRN'}, inplace=True)
partial_key = partial_key[['MRN', 'AnonMRN']]  # Keep only relevant columns

# Load the clinical data from the Excel file
clinical_data = pd.read_excel('NGS_data_CNS_cohort_20250521.xlsx')

# Merge the partial key with the clinical data on MRN
merged_data = pd.merge(clinical_data, partial_key, on='MRN', how='left')

# Get folder names in the '../NGS Data/' directory
ngs_data_path = '../NGS Data/'
ngs_folders = [f for f in os.listdir(ngs_data_path) if os.path.isdir(os.path.join(ngs_data_path, f))]

# Create a DataFrame for folder names
ngs_folders_df = pd.DataFrame(ngs_folders, columns=['AnonMRN'])

# Filter merged_data to include only rows with AnonMRN matching folder names
filtered_data = merged_data[merged_data['AnonMRN'].isin(ngs_folders_df['AnonMRN'])]

# Load additional clinical data from another CSV file
clinical2 = pd.read_csv('CNSTumors_DATA_LABELS_2025-05-05_1826.csv')

# Merge the filtered data with clinical2 on Database_ID and Study #
final_data = pd.merge(filtered_data, clinical2, left_on='Database_ID', right_on='Study #', how='left')


# %% CLINICAL SURVIVAL MODEL

# final_data was saved to csv and cleaned manually - now we can load the simpler versio
df_clinical = pd.read_csv('merged-clinical.csv')

vars_to_keep = ['AGE', 'HISTOL', 'IDH', 'MGMT',
                'Extent of surgical resection', 
                'BECOG', 'EVENT', 'TIME']


from lifelines import CoxPHFitter

# Drop rows with missing values
df_clinical_cleaned = df_clinical[vars_to_keep].dropna()

# drop last row 
df_clinical_cleaned = df_clinical_cleaned[:-1]

# variable encoding where necessary
df_clinical_cleaned['HISTOL'] = df_clinical_cleaned['HISTOL'] == 'GBM'
df_clinical_cleaned['IDH'] = df_clinical_cleaned['IDH'] == 'IDHmut'
df_clinical_cleaned['MGMT'] = df_clinical_cleaned['MGMT'] == 'Methylated (M)'
df_clinical_cleaned['Extent of surgical resection'] = df_clinical_cleaned['Extent of surgical resection'] == 'Total'
df_clinical_cleaned['BECOG'] = df_clinical_cleaned['BECOG'] != '0: Asymptomatic'
# Convert EVENT to a binary variable (1 for event, 0 for censored)
df_clinical_cleaned['EVENT'] = df_clinical_cleaned['EVENT'] == 'Dead'
# Ensure TIME is numeric
df_clinical_cleaned['TIME'] = pd.to_numeric(df_clinical_cleaned['TIME'], errors='coerce')


# Fit the Cox proportional hazards model
cox_model = CoxPHFitter()
cox_model.fit(df_clinical_cleaned, duration_col='TIME', event_col='EVENT')

# Print the summary of the Cox model
print("Cox Proportional Hazards Model Summary:")
cox_model.print_summary()

# Optionally, save the model summary to a file
cox_model.summary.to_csv('cox_model_summary.csv')

# Save the final merged data to a CSV file
# %% IMAGE PROCESSING

'''
Image processing pipeline for the NGS CNS Glioma dataset.
Pipeline includes:
1. Loading the MRI from each folder and performing bias field correction and save the new image.
2. Using the GTV and brain masks, we will isolate a peritumoral region around the GTV -- effectively we will expand the GTV by a certain number of pixels in each direction to create a peritumoral region and then omit any pixels in the peritumoral region that are NOT in the brain.
3. Save the peritumoral region as a new NIfTI file in the same folder as the original MRI.
4. Create a CSV that creates a mapping to the various image/mask files we have in each folder for PyRadiomics batch processing.
'''

# load the MRI and masks from each folder
import nibabel as nib
import numpy as np
import SimpleITK as sitk

def bias_field_correction(image_path):
    """Perform bias field correction on the given image."""
    image = sitk.ReadImage(image_path)
    corrected_image = sitk.N4BiasFieldCorrectionImageFilter().Execute(image)
    return corrected_image

# mri_path = '../NGS Data/NGS0015/MR-REG.nii.gz'
# corrected_mri = bias_field_correction(mri_path)
# # Save the corrected MRI
# corrected_mri_path = '../NGS Data/NGS0015/MR-REG-corrected.nii.gz'
# sitk.WriteImage(corrected_mri, corrected_mri_path)
# # Load the GTV and brain masks
# gtv_path = '../NGS Data/NGS0015/ROI__[GTVp].nii.gz'
# brain_mask_path = '../NGS Data/NGS0015/ROI__[Brain].nii.gz'
# gtv_mask = sitk.ReadImage(gtv_path)
# brain_mask = sitk.ReadImage(brain_mask_path)

# %%
def create_peritumoral_region(gtv_mask, brain_mask, expansion_radius=5):
    """Create a peritumoral region around the GTV mask."""
    # Convert masks to binary
    gtv_binary = sitk.BinaryThreshold(gtv_mask, lowerThreshold=1, upperThreshold=1)
    brain_binary = sitk.BinaryThreshold(brain_mask, lowerThreshold=1, upperThreshold=1)

    # Convert expansion_radius to a list for 3D dilation
    expansion_radius_vector = [expansion_radius] * 3  # [5, 5, 5] for uniform dilation in x, y, z

    # Dilate the GTV mask to create a peritumoral region
    dilated_gtv = sitk.BinaryDilate(gtv_binary, expansion_radius_vector)

    # Combine with the brain mask to ensure we only keep pixels in the brain
    peritumoral_region = sitk.And(dilated_gtv, brain_binary)
    peritumoral_region = sitk.And(peritumoral_region, sitk.Not(gtv_binary))

    return peritumoral_region
# # Create the peritumoral region
# peritumoral_region = create_peritumoral_region(gtv_mask, brain_mask, expansion_radius=5)
# # Save the peritumoral region as a new NIfTI file
# peritumoral_region_path = '../NGS Data/NGS0015/peritumoral_region.nii.gz'
# sitk.WriteImage(peritumoral_region, peritumoral_region_path)



# %%

# okay, this looks like it works, now we need to run this for each folder in the NGS Data directory
def process_folder(folder_name):
    """Process a single folder to create peritumoral regions."""
    folder_path = os.path.join(ngs_data_path, folder_name)
    
    try:
        # Load the MRI and masks
        mri_path = os.path.join(folder_path, 'MR-REG.nii.gz')
        # print(mri_path)
        
        # Possible GTV mask filenames
        gtv_filenames = [
            'ROI__[GTV].nii.gz',
            'ROI__[GTVp].nii.gz',
            'ROI__[GTVp1].nii.gz',
            'ROI__[GTVp_flair].nii.gz',
            'ROI__[ref_GTVp_flair].nii.gz',
            'ROI__[GTVm].nii.gz',
            'ROI__[GTVm1].nii.gz'
        ]
        
        # Find the first existing GTV mask file
        gtv_path = None
        for filename in gtv_filenames:
            potential_path = os.path.join(folder_path, filename)
            # print(f"Checking for GTV mask: {potential_path}")
            if os.path.exists(potential_path):
                gtv_path = potential_path
                print(f"Found GTV mask: {gtv_path}")
                break
        
        if gtv_path is None:
            raise FileNotFoundError(f"No GTV mask file found in folder: {folder_path}")
        
        brain_mask_path = os.path.join(folder_path, 'ROI__[Brain].nii.gz')

        # Check if files exist
        if not os.path.exists(mri_path):
            raise FileNotFoundError(f"Missing MRI file: {mri_path}")
        if not os.path.exists(brain_mask_path):
            raise FileNotFoundError(f"Missing brain mask file: {brain_mask_path}")

        # Perform bias field correction on the MRI
        corrected_mri = bias_field_correction(mri_path)
        corrected_mri_path = os.path.join(folder_path, 'MR-REG-corrected.nii.gz')
        sitk.WriteImage(corrected_mri, corrected_mri_path)

        # Load GTV and brain masks
        gtv_mask = sitk.ReadImage(gtv_path)
        brain_mask = sitk.ReadImage(brain_mask_path)

        # Create peritumoral region
        peritumoral_region = create_peritumoral_region(gtv_mask, brain_mask, expansion_radius=5)
        
        # Save the peritumoral region
        peritumoral_region_path = os.path.join(folder_path, 'peritumoral_region.nii.gz')
        sitk.WriteImage(peritumoral_region, peritumoral_region_path)

    except FileNotFoundError as e:
        print(f"Error processing folder {folder_name}: {e}")
    except Exception as e:
        print(f"Unexpected error processing folder {folder_name}: {e}")

ngs_folders = ['NGS0501','NGS0427']

# Process each folder in the NGS Data directory
# for folder in ngs_folders:
#     process_folder(folder)
    
# Create a CSV mapping for image/mask files

# %%
import pandas as pd
radiomics = pd.read_csv('results.csv')
radiomics_cohort2 = pd.read_csv('radiomics-cohort2-1.csv')





# %%
def process_radiomics_data(radiomics):
    """Process radiomics DataFrame into concatenated features based on modality and ROI."""
    # Separate rows into six DataFrames based on modality and ROI
    ct_brain = radiomics[(radiomics['ROI'] == 'Brain') & (radiomics['Image'].str.contains('CT.nii.gz'))].reset_index()
    ct_tumor = radiomics[(radiomics['ROI'] == 'Tumor') & (radiomics['Image'].str.contains('CT.nii.gz'))].reset_index()
    ct_peritumor = radiomics[(radiomics['ROI'] == 'Peritumoral') & (radiomics['Image'].str.contains('CT.nii.gz'))].reset_index()
    mr_brain = radiomics[(radiomics['ROI'] == 'Brain') & (radiomics['Image'].str.contains('MR-REG-corrected.nii.gz'))].reset_index()
    mr_tumor = radiomics[(radiomics['ROI'] == 'Tumor') & (radiomics['Image'].str.contains('MR-REG-corrected.nii.gz'))].reset_index()
    mr_peritumor = radiomics[(radiomics['ROI'] == 'Peritumoral') & (radiomics['Image'].str.contains('MR-REG-corrected.nii.gz'))].reset_index()

    # Function to process each DataFrame
    def process_dataframe(df, suffix):
        # Drop columns up to 'original_shape_Elongation'
        start_col = 'original_shape_Elongation'
        df = df.loc[:, df.columns[df.columns.get_loc(start_col):]]
        
        # Append suffix to column names
        df.columns = [col + suffix for col in df.columns]
        
        # Add USUBJID column back
        df['USUBJID'] = radiomics['USUBJID'].drop_duplicates().reset_index(drop=True)
        
        return df

    # Process each DataFrame
    ct_brain = process_dataframe(ct_brain, '_CT_Brain')
    ct_tumor = process_dataframe(ct_tumor, '_CT_GTV')
    ct_peritumor = process_dataframe(ct_peritumor, '_CT_Peritumor')
    mr_brain = process_dataframe(mr_brain, '_MR_Brain')
    mr_tumor = process_dataframe(mr_tumor, '_MR_GTV')
    mr_peritumor = process_dataframe(mr_peritumor, '_MR_Peritumor')

    # Concatenate all DataFrames
    final_df = pd.concat([ct_brain, ct_tumor, ct_peritumor, mr_brain, mr_tumor, mr_peritumor], axis=1)

    # Ensure USUBJID is the first column
    final_df = final_df.loc[:, ['USUBJID'] + [col for col in final_df.columns if col != 'USUBJID']]

    return final_df

# %%
# Process the radiomics DataFrame
# Process both cohorts and concatenate results
try:
    radiomics_df1 = process_radiomics_data(radiomics)
except Exception as e:
    print(f"Error processing cohort1 radiomics: {e}")
    radiomics_df1 = pd.DataFrame()

try:
    radiomics_df2 = process_radiomics_data(radiomics_cohort2)
except Exception as e:
    print(f"Error processing cohort2 radiomics: {e}")
    radiomics_df2 = pd.DataFrame()

# Concatenate the two processed cohorts (stack rows; align columns)
final_radiomics_df = pd.concat([radiomics_df1, radiomics_df2], ignore_index=True, sort=False)

# keep only one column named USUBJID
final_radiomics_df = final_radiomics_df.loc[:, ~final_radiomics_df.columns.duplicated()]

# Save the final DataFrame to a CSV file
# final_radiomics_df.to_csv('processed_radiomics.csv', index=False)

# Print the final DataFrame
print(final_radiomics_df)

# %% FEATURE REDUCTION

'''
We will do a stepwise feature reduction using the following methods:
1. Apply a variance threshold - any feature that has a variance less than the median variance across all features will be removed.
2. Apply a correlation threshold - any feature that has a correlation greater than 0.1 with 'original_shape_VoxelVolume_CT_GTV' will be removed.
3. Apply a second correlation threshold - any feature that has a correlation greater than 0.7 with any remaining feature will be removed.
'''
from sklearn.feature_selection import VarianceThreshold

def reduce_features(df, target_feature, volume_threshold=0.1, correlation_threshold=0.7):
    """Reduce features in the DataFrame based on variance and correlation thresholds."""
    # Preserve the USUBJID column
    usubjid_column = df['USUBJID'].drop_duplicates().reset_index(drop=True)
    df_features = df.drop(columns=['USUBJID'], errors='ignore')  # Exclude USUBJID for feature reduction

    # Step 1: Variance Threshold
    # Calculate variance for each feature
    variances = df_features.var()
    median_variance = variances.median()
    
    # Select features with variance above the threshold
    high_variance_features = variances[variances > median_variance].index.tolist()
    
    # Filter DataFrame to keep only high variance features
    df_high_variance = df_features[high_variance_features]
    
    # Step 2: Correlation Threshold with target feature
    correlations = df_high_variance.corr()[target_feature]
    correlated_features = correlations[abs(correlations) < volume_threshold].index.tolist()
    
    # Filter DataFrame to keep only features correlated with the target feature
    df_correlated = df_high_variance[correlated_features]
    
    # Step 3: Second Correlation Threshold (0.7)
    corr_matrix = df_correlated.corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find features with correlation greater than 0.7
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > correlation_threshold)]
    
    # Drop highly correlated features
    df_final = df_correlated.drop(columns=to_drop, errors='ignore')
    
    # Re-add the USUBJID column
    df_final['USUBJID'] = usubjid_column
    
    # Ensure USUBJID is the first column
    df_final = df_final.loc[:, ['USUBJID'] + [col for col in df_final.columns if col != 'USUBJID']]
    
    return df_final

# Define the target feature for correlation checks
target_feature = 'original_shape_VoxelVolume_CT_GTV'

# Reduce features in the final radiomics DataFrame
reduced_radiomics_df = reduce_features(final_radiomics_df, target_feature)

# Save the reduced DataFrame to a CSV file
reduced_radiomics_df.to_csv('reduced_radiomics.csv', index=False)

# Print the reduced DataFrame
print(reduced_radiomics_df)

# %%

''' 
We still need to minimize the number of features we have, so we will use a Random Forest classifier to rank the features and then select the top 10 features based on their importance.
First, we need to look at the overlap in patients between the clinical data and the radiomics data.
'''

# clinical is df_clinical 
# radiomics is reduced_radiomics_df
df_clinic = df_clinical.copy()
# Ensure USUBJID is in the clinical DataFrame
if 'USUBJID' not in df_clinic.columns:
    df_clinic['USUBJID'] = df_clinic['AnonMRN']

# drop unnecessary columns from clinical data
df_clinic = df_clinic[['USUBJID', 'AGE', 'HISTOL', 'IDH', 'MGMT',
                       'Extent of surgical resection', 
                       'BECOG', 'EVENT', 'TIME']]

# let's not merge them - instead, let's look at the overlap in USUBJID
# Find common USUBJID values in both DataFrames
common_usubjid = set(df_clinic['USUBJID']).intersection(set(reduced_radiomics_df['USUBJID']))

# now filter both DataFrames to keep only the common USUBJID values
df_clinic_filtered = df_clinic[df_clinic['USUBJID'].isin(common_usubjid)].reset_index(drop=True)
reduced_radiomics_filtered = reduced_radiomics_df[reduced_radiomics_df['USUBJID'].isin(common_usubjid)].reset_index(drop=True)
# Print the number of common USUBJID values
print(f"Number of common USUBJID values: {len(common_usubjid)}")

# %%

def fit_cox_model(df_clinical):
    """
    Preprocess clinical data and fit a Cox proportional hazards model.

    Parameters:
    df_clinical (pd.DataFrame): Clinical data DataFrame.

    Returns:
    CoxPHFitter: Fitted Cox proportional hazards model.
    """
    vars_to_keep = ['AGE', 'HISTOL', 'IDH', 'MGMT',
                    'Extent of surgical resection', 
                    'BECOG', 'EVENT', 'TIME']

    # Drop rows with missing values
    df_clinical_cleaned = df_clinical[vars_to_keep].dropna()

    # Drop last row
    df_clinical_cleaned = df_clinical_cleaned[:-1]

    # Variable encoding where necessary
    df_clinical_cleaned['HISTOL'] = df_clinical_cleaned['HISTOL'] == 'GBM'
    df_clinical_cleaned['IDH'] = df_clinical_cleaned['IDH'] == 'IDHmut'
    df_clinical_cleaned['MGMT'] = df_clinical_cleaned['MGMT'] == 'Methylated (M)'
    df_clinical_cleaned['Extent of surgical resection'] = df_clinical_cleaned['Extent of surgical resection'] == 'Total'
    df_clinical_cleaned['BECOG'] = df_clinical_cleaned['BECOG'] != '0: Asymptomatic'
    # Convert EVENT to a binary variable (1 for event, 0 for censored)
    df_clinical_cleaned['EVENT'] = df_clinical_cleaned['EVENT'] == 'Dead'
    # Ensure TIME is numeric
    df_clinical_cleaned['TIME'] = pd.to_numeric(df_clinical_cleaned['TIME'], errors='coerce')

    # Fit the Cox proportional hazards model
    cox_model = CoxPHFitter()
    cox_model.fit(df_clinical_cleaned, duration_col='TIME', event_col='EVENT')

    # Print the summary of the Cox model
    print("Cox Proportional Hazards Model Summary:")
    cox_model.print_summary()

    # Optionally, save the model summary to a file
    cox_model.summary.to_csv('cox_model_summary.csv')

    return cox_model

# Fit the Cox model using the filtered clinical data
cox_model = fit_cox_model(df_clinic_filtered)
# %%

'''
Now we will use the Random Forest classifier to rank the features in the radiomics data and select the top 10 features based on their importance.
'''

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
def rank_features_with_rf(df_radiomics, df_clinical, target_col='EVENT'):
    """
    Rank features using a Random Forest classifier and select the top 10 features.

    Parameters:
    df_radiomics (pd.DataFrame): Radiomics data DataFrame.
    df_clinical (pd.DataFrame): Clinical data DataFrame.
    target_col (str): The target column for classification.

    Returns:
    pd.DataFrame: DataFrame with top 10 ranked features.
    """
    
    vars_to_keep = ['USUBJID', 'AGE', 'HISTOL', 'IDH', 'MGMT',
                    'Extent of surgical resection',
                    'BECOG', 'EVENT', 'TIME']
    
    # Drop rows with missing values
    df_clinical_cleaned = df_clinical[vars_to_keep].dropna()

    # Drop last row
    df_clinical_cleaned = df_clinical_cleaned[:-1]

    # Variable encoding where necessary
    df_clinical_cleaned['HISTOL'] = df_clinical_cleaned['HISTOL'] == 'GBM'
    df_clinical_cleaned['IDH'] = df_clinical_cleaned['IDH'] == 'IDHmut'
    df_clinical_cleaned['MGMT'] = df_clinical_cleaned['MGMT'] == 'Methylated (M)'
    df_clinical_cleaned['Extent of surgical resection'] = df_clinical_cleaned['Extent of surgical resection'] == 'Total'
    df_clinical_cleaned['BECOG'] = df_clinical_cleaned['BECOG'] != '0: Asymptomatic'
    # Convert EVENT to a binary variable (1 for event, 0 for censored)
    df_clinical_cleaned['EVENT'] = df_clinical_cleaned['EVENT'] == 'Dead'
    # Ensure TIME is numeric
    df_clinical_cleaned['TIME'] = pd.to_numeric(df_clinical_cleaned['TIME'], errors='coerce')
    
    # Merge radiomics and clinical data on USUBJID
    merged_data = pd.merge(df_radiomics, df_clinical_cleaned, on='USUBJID', how='inner')

    # Prepare features and target variable
    X = merged_data.drop(columns=['USUBJID', target_col])
    y = merged_data[target_col]=='Dead'  # Ensure target is binary

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit a Random Forest classifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Get feature importances
    feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)

    # Select the top 10 features
    top_features = feature_importances.head(10)

    return top_features
# Rank features using Random Forest
top_features = rank_features_with_rf(reduced_radiomics_filtered, df_clinic_filtered)
# Print the top 10 features
print("Top 10 Features Ranked by Random Forest:")
print(top_features)
# %%

'''
What if try PCA on the radiomics data to reduce the dimensionality and then use the top components as features for the Cox model?
Let's generate the PCA components and a skree plot to visualize the explained variance.
'''

from sklearn.decomposition import PCA

def perform_pca(df_radiomics, n_components=10):
    """
    Perform PCA on the radiomics data and return the top components.

    Parameters:
    df_radiomics (pd.DataFrame): Radiomics data DataFrame.
    n_components (int): Number of PCA components to return.

    Returns:
    pd.DataFrame: DataFrame with PCA components.
    """
    # Drop USUBJID column for PCA
    X = df_radiomics.drop(columns=['USUBJID'])

    # Standardize the data
    X_standardized = (X - X.mean()) / X.std()

    # Perform PCA
    pca = PCA(n_components=n_components)
    pca_components = pca.fit_transform(X_standardized)

    # Create a DataFrame with PCA components
    pca_df = pd.DataFrame(data=pca_components, columns=[f'PC{i+1}' for i in range(n_components)])
    
    # Add USUBJID back to the DataFrame
    pca_df['USUBJID'] = df_radiomics['USUBJID'].reset_index(drop=True)

    return pca_df, pca.explained_variance_ratio_
# Perform PCA on the reduced radiomics data
pca_df, explained_variance = perform_pca(reduced_radiomics_filtered, n_components=10)
# Print the PCA components
print("PCA Components:")
print(pca_df.head())

# Plot the explained variance ratio
import matplotlib.pyplot as plt
def plot_explained_variance(explained_variance):
    """
    Plot the explained variance ratio from PCA.

    Parameters:
    explained_variance (array-like): Explained variance ratio from PCA.
    """
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, color='blue')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance Ratio by Principal Component')
    plt.xticks(range(1, len(explained_variance) + 1))
    plt.grid(axis='y')
    plt.show()
# Plot the explained variance ratio
plot_explained_variance(explained_variance)

# %%

from sklearn.preprocessing import StandardScaler

def standardize_features(df_radiomics):
    # Drop USUBJID column for standardization
    X = df_radiomics.drop(columns=['USUBJID'])
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)
    
    # Create a standardized DataFrame
    df_standardized = pd.DataFrame(X_standardized, columns=X.columns)
    df_standardized['USUBJID'] = df_radiomics['USUBJID'].reset_index(drop=True)
    return df_standardized

# Standardize radiomics features
reduced_radiomics_filtered = standardize_features(reduced_radiomics_filtered)


# %%


from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

def evaluate_cox_model(df_clinical, df_radiomics=None, use_pca=False):
    """
    Fit and evaluate a Cox proportional hazards model with clinical and optional radiomics features.

    Parameters:
    df_clinical (pd.DataFrame): Clinical data DataFrame.
    df_radiomics (pd.DataFrame): Radiomics data DataFrame (optional).
    use_pca (bool): Whether to use PCA components instead of raw radiomics features.

    Returns:
    float: Concordance index of the fitted model.
    """
    # Merge clinical and radiomics data if radiomics features are provided
    if df_radiomics is not None:
        df_combined = pd.merge(df_clinical, df_radiomics, on='USUBJID', how='inner')
    else:
        df_combined = df_clinical

    # Select features for the Cox model
    vars_to_keep = ['AGE', 'HISTOL', 'IDH', 'MGMT',
                    'Extent of surgical resection', 
                    'BECOG', 'EVENT', 'TIME'] + (list(df_radiomics.columns) if df_radiomics is not None else [])
    
    # Drop rows with missing values
    df_combined_cleaned = df_combined[vars_to_keep].dropna()

    # Variable encoding where necessary
    df_combined_cleaned['HISTOL'] = df_combined_cleaned['HISTOL'] == 'GBM'
    df_combined_cleaned['IDH'] = df_combined_cleaned['IDH'] == 'IDHmut'
    df_combined_cleaned['MGMT'] = df_combined_cleaned['MGMT'] == 'Methylated (M)'
    df_combined_cleaned['Extent of surgical resection'] = df_combined_cleaned['Extent of surgical resection'] == 'Total'
    df_combined_cleaned['BECOG'] = df_combined_cleaned['BECOG'] != '0: Asymptomatic'
    # Convert EVENT to a binary variable (1 for event, 0 for censored)
    df_combined_cleaned['EVENT'] = df_combined_cleaned['EVENT'] == 'Dead'
    # Ensure TIME is numeric
    df_combined_cleaned['TIME'] = pd.to_numeric(df_combined_cleaned['TIME'], errors='coerce')
    
    # if there is a USUBJID column, drop it
    if 'USUBJID' in df_combined_cleaned.columns:
        df_combined_cleaned = df_combined_cleaned.drop(columns=['USUBJID'])
    
    try:
        # Fit the Cox proportional hazards model
        cox_model = CoxPHFitter(penalizer=0.1)  # Add a small penalty to avoid overfitting
        cox_model.fit(df_combined_cleaned, duration_col='TIME', event_col='EVENT')

        # Calculate concordance index
        c_index = concordance_index(df_combined_cleaned['TIME'], -cox_model.predict_partial_hazard(df_combined_cleaned), df_combined_cleaned['EVENT'])

        return cox_model,c_index
    except Exception as e:
        print(f"Error fitting Cox model: {e}")
        return df_combined_cleaned
# Perform PCA on the reduced radiomics data
pca_df, explained_variance = perform_pca(reduced_radiomics_filtered, n_components=10)

# Evaluate baseline clinical model
baseline_model, baseline_c_index = evaluate_cox_model(df_clinic_filtered)

# Evaluate combined model with PCA components and clinical features
pca_model, combined_pca_c_index = evaluate_cox_model(df_clinic_filtered, pca_df)

# Print results
print(f"Baseline Clinical Model Concordance Index: {baseline_c_index:.4f}")
print("Baseline Clinical Model Summary:")
baseline_model.print_summary()

print(f"\nCombined Model with PCA Components Concordance Index: {combined_pca_c_index:.4f}")
print("Combined Model with PCA Components Summary:")
pca_model.print_summary()
# %%

'''
Let's try a RandomSurvivalForest model to see if we can improve the concordance index.
We'll try with both radiomics features and clinical features, and look at feature importance (if possible)
'''

from sksurv.ensemble import RandomSurvivalForest

def evaluate_random_survival_forest(df_clinical, df_radiomics=None, use_pca=False): 
    """
    Fit and evaluate a Random Survival Forest model with clinical and optional radiomics features.

    Parameters:
    df_clinical (pd.DataFrame): Clinical data DataFrame.
    df_radiomics (pd.DataFrame): Radiomics data DataFrame (optional).
    use_pca (bool): Whether to use PCA components instead of raw radiomics features.

    Returns:
    float: Concordance index of the fitted model.
    """
    # Merge clinical and radiomics data if radiomics features are provided
    if df_radiomics is not None:
        df_combined = pd.merge(df_clinical, df_radiomics, on='USUBJID', how='inner')
    else:
        df_combined = df_clinical

    # Select features for the Random Survival Forest model
    vars_to_keep = ['AGE', 'HISTOL', 'IDH', 'MGMT',
                    'Extent of surgical resection', 
                    'BECOG', 'EVENT', 'TIME'] + (list(df_radiomics.columns) if df_radiomics is not None else [])
    
    # Drop rows with missing values
    df_combined_cleaned = df_combined[vars_to_keep].dropna()

    # Variable encoding where necessary
    df_combined_cleaned['HISTOL'] = df_combined_cleaned['HISTOL'] == 'GBM'
    df_combined_cleaned['IDH'] = df_combined_cleaned['IDH'] == 'IDHmut'
    df_combined_cleaned['MGMT'] = df_combined_cleaned['MGMT'] == 'Methylated (M)'
    df_combined_cleaned['Extent of surgical resection'] = df_combined_cleaned['Extent of surgical resection'] == 'Total'
    df_combined_cleaned['BECOG'] = df_combined_cleaned['BECOG'] != '0: Asymptomatic'
    
    # Convert EVENT to a boolean array for survival analysis
    event_array = np.array(df_combined_cleaned['EVENT']) == 'Dead'
    
    # Ensure TIME is numeric
    df_combined_cleaned['TIME'] = pd.to_numeric(df_combined_cleaned['TIME'], errors='coerce')

    # Prepare the structured array for survival analysis
    structured_array = np.array(list(zip(event_array, df_combined_cleaned['TIME'])), 
                                dtype=[('event', '?'), ('time', '<f8')])
    
    # drop the EVENT and TIME columns from the DataFrame
    df_combined_cleaned = df_combined_cleaned.drop(columns=['EVENT', 'TIME'], errors='ignore')
    
    # if there is a USUBJID column, drop it
    if 'USUBJID' in df_combined_cleaned.columns:
        df_combined_cleaned = df_combined_cleaned.drop(columns=['USUBJID'])
    try:
        # Fit the Random Survival Forest model
        rsf_model = RandomSurvivalForest(n_estimators=100, random_state=42)
        rsf_model.fit(df_combined_cleaned, structured_array)

        # Calculate concordance index
        c_index = rsf_model.score(df_combined_cleaned, structured_array)

        return rsf_model, c_index, df_combined_cleaned, structured_array
    except Exception as e:
        print(f"Error fitting Random Survival Forest model: {e}")
        return None, None, None, None
# Evaluate baseline clinical model with Random Survival Forest
baseline_rsf_model, baseline_rsf_c_index, clin_dat, structured_array = evaluate_random_survival_forest(df_clinic_filtered)

# Evaluate combined model with PCA components and clinical features
pca_rsf_model, combined_pca_rsf_c_index, pca_dat, structured_array = evaluate_random_survival_forest(df_clinic_filtered, pca_df)
# Print results
print(f"Baseline Clinical Model (Random Survival Forest) Concordance Index: {baseline_rsf_c_index:.4f}")
print("Baseline Clinical Model (Random Survival Forest) Summary:")
print(baseline_rsf_model)
print(f"\nCombined Model with PCA Components (Random Survival Forest) Concordance Index: {combined_pca_rsf_c_index:.4f}")
print("Combined Model with PCA Components (Random Survival Forest) Summary:")
print(pca_rsf_model)


# p-values???
# %%
from sksurv.metrics import concordance_index_censored
from scipy.stats import ttest_ind
import nibabel as nib
import numpy as np
import os
def compare_c_index_significance(model1, model2, X1, X2, y, n_iterations=1000):
    """
    Compare the significance of concordance indices between two Random Survival Forest models.

    Parameters:
    model1 (RandomSurvivalForest): First fitted Random Survival Forest model.
    model2 (RandomSurvivalForest): Second fitted Random Survival Forest model.
    X (pd.DataFrame): Feature matrix.
    y (structured array): Survival data (event and time).
    n_iterations (int): Number of bootstrap iterations for significance testing.

    Returns:
    float: p-value indicating the significance of the difference in concordance indices.
    """
    c_index1 = concordance_index_censored(y['event'], y['time'], -model1.predict(X1))
    c_index2 = concordance_index_censored(y['event'], y['time'], -model2.predict(X2))

    # Bootstrap to estimate the distribution of the difference in concordance indices
    differences = []
    for _ in range(n_iterations):
        indices = np.random.choice(range(len(y)), size=len(y), replace=True)
        boot_y = y[indices]
        boot_X1 = X1.iloc[indices]
        boot_X2 = X2.iloc[indices]
        
        boot_c_index1 = concordance_index_censored(boot_y['event'], boot_y['time'], -model1.predict(boot_X1))[0]
        boot_c_index2 = concordance_index_censored(boot_y['event'], boot_y['time'], -model2.predict(boot_X2))[0]
        
        differences.append((boot_c_index1, boot_c_index2))

        # Separate the c-indices into two lists
        c_index1_list, c_index2_list = zip(*differences)

        # Perform a t-test on the two sets of c-indices
        t_statistic, p_value = ttest_ind(c_index2_list, c_index1_list, alternative='greater')
    
    return p_value
# Compare the significance of concordance indices between the baseline RSF model and the combined RSF model with PCA components
p_value = compare_c_index_significance(baseline_rsf_model, pca_rsf_model, clin_dat, pca_dat, structured_array)
# Print the p-value
print(f"p-value for the difference in concordance indices: {p_value:.4f}")

# %% Try subsetting by IDH mutation statius

idh_status = 'IDHwt'  # or 'IDHwt' for wild-type
wt_flag = True 
use_subset = False

# using df_clinical_cleaned and pca_dat (where we need to add the TIME and EVENT columns back)
if use_subset:
    clinical_data_subset = df_clinic_filtered.copy()[df_clinic_filtered['IDH'] == idh_status].reset_index(drop=True)
    # print(clinical_data_subset['HISTOL'].value_counts())
    # clinical_data_subset.loc[clinical_data_subset['IDH'] == 'IDHwt', 'HISTOL'] = 'GBM'
    # print(clinical_data_subset['HISTOL'].value_counts())
else:
    clinical_data_subset = df_clinic_filtered.copy()
    # classify all IDHwt as GBM
    # clinical_data_subset.loc[clinical_data_subset['IDH'] == 'IDHwt', 'HISTOL'] = 'GBM'
print(clinical_data_subset['HISTOL'].value_counts())
# Cleanup the clinical data as above
clinical_data_subset['HISTOL'] = clinical_data_subset['HISTOL'] == 'GBM'
if use_subset and wt_flag:
    # drop histology as they should all be GBM
    clinical_data_subset = clinical_data_subset.drop(columns=['HISTOL'], errors='ignore')
clinical_data_subset['MGMT'] = clinical_data_subset['MGMT'] == 'Methylated (M)'
clinical_data_subset['Extent of surgical resection'] = clinical_data_subset['Extent of surgical resection'] == 'Total'
clinical_data_subset['BECOG'] = clinical_data_subset['BECOG'] != '0: Asymptomatic'
clinical_data_subset['EVENT'] = clinical_data_subset['EVENT'] == 'Dead'

imaging_data_subset = pca_df.copy()[pca_df['USUBJID'].isin(clinical_data_subset['USUBJID'])].reset_index(drop=True)
# Add TIME and EVENT columns back to the imaging data
imaging_data_subset['TIME'] = clinical_data_subset['TIME'].reset_index(drop=True)
imaging_data_subset['EVENT'] = clinical_data_subset['EVENT'].reset_index(drop=True)


# Conditionally drop columns based on use_subset
if use_subset:
    clinical_data_subset = clinical_data_subset.drop(columns=['IDH', 'USUBJID'], errors='ignore')
    if wt_flag:
        # drop histology as they should all be GBM
        clinical_data_subset = clinical_data_subset.drop(columns=['HISTOL'], errors='ignore')
    imaging_data_subset = imaging_data_subset.drop(columns=['IDH', 'USUBJID'], errors='ignore')
else:
    clinical_data_subset = clinical_data_subset.drop(columns=['USUBJID'], errors='ignore')
    imaging_data_subset = imaging_data_subset.drop(columns=['USUBJID'], errors='ignore')
    clinical_data_subset['IDH'] = clinical_data_subset['IDH'] == 'IDHwt'

# drop missing values
clinical_data_subset = clinical_data_subset.dropna()
imaging_data_subset = imaging_data_subset.dropna()

print(clinical_data_subset.columns)


cox_model = CoxPHFitter(penalizer=0.001)  # Add a small penalty to avoid overfitting
# Fit the Cox proportional hazards model for IDH-mutant patients
cox_model.fit(clinical_data_subset, duration_col='TIME', event_col='EVENT')
# Calculate concordance index for IDH-mutant patients
idh_mutant_c_index = concordance_index(clinical_data_subset['TIME'], -cox_model.predict_partial_hazard(clinical_data_subset), clinical_data_subset['EVENT'])
# Results
cox_model.print_summary()
overall_p_value = cox_model.log_likelihood_ratio_test().p_value
print(f"Clinical-Only Concordance Index: {idh_mutant_c_index:.4f}")
print(f"Overall p-value for the model: {overall_p_value:.4f}")

# imaging data subset (Cox model with PCA components)
pca_model = CoxPHFitter(penalizer=0.001)  # Add a small penalty to avoid overfitting
pca_model.fit(imaging_data_subset, duration_col='TIME', event_col='EVENT')
# Calculate concordance index for IDH-mutant patients with PCA components
idh_mutant_pca_c_index = concordance_index(imaging_data_subset['TIME'], -pca_model.predict_partial_hazard(imaging_data_subset), imaging_data_subset['EVENT'])
# Results
pca_model.print_summary()
overall_p_value = pca_model.log_likelihood_ratio_test().p_value
print(f"PCA Components Concordance Index: {idh_mutant_pca_c_index:.4f}")
print(f"Overall p-value for the model: {overall_p_value:.4f}")

# now I want to concatenate the two DataFrames (drop the TIME and EVENT columns from the imaging data)
combined_idh_mutant_data = pd.concat([clinical_data_subset, imaging_data_subset.drop(columns=['TIME', 'EVENT'], errors='ignore')], axis=1)
combo_cox = CoxPHFitter(penalizer=0.001)  # Add a small penalty to avoid overfitting
combo_cox.fit(combined_idh_mutant_data, duration_col='TIME', event_col='EVENT')
# Calculate concordance index for IDH-mutant patients with combined data
idh_mutant_combo_c_index = concordance_index(combined_idh_mutant_data['TIME'], -combo_cox.predict_partial_hazard(combined_idh_mutant_data), combined_idh_mutant_data['EVENT'])
# Results
combo_cox.print_summary()
overall_p_value = combo_cox.log_likelihood_ratio_test().p_value
print(f"Combined Data Concordance Index: {idh_mutant_combo_c_index:.4f}")
print(f"Overall p-value for the model: {overall_p_value:.4f}")

# %%

import qrcode

def generate_qr_code(url, output_file):
    """Generate a QR code that directs to a website and save as PNG with a transparent background."""
    # Ensure URL is a string and has a scheme
    if not isinstance(url, str):
        url = str(url)
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url

    # Transparent background requires PNG
    if not output_file.lower().endswith('.png'):
        raise ValueError("output_file must be a .png to support transparency")

    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(url)
    qr.make(fit=True)

    # Create an RGBA image and make the white background fully transparent
    img = qr.make_image(fill_color="black", back_color="white").convert("RGBA")
    # Replace white background pixels (and near-white anti-aliased pixels) with transparent pixels
    datas = img.getdata()
    newData = []
    for item in datas:
        # item is (R, G, B, A)
        r, g, b, a = item
        # treat near-white as background to preserve anti-aliased edges
        if r > 240 and g > 240 and b > 240:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)
    img.putdata(newData)
    img.save(output_file)
    
# Example usage
generate_qr_code("https://bhklab.ca/", "bhklab-qr.png")
# %%
