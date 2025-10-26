import os
import pandas as pd
import shutil

df_path = pd.read_csv('regDat.csv')

'''
We have a DataFrame `df_path` that contains the paths to DICOM files and nifti files.
The DataFrame has the following columns:
    - 'RegPath': Path to the registration DICOM file
    - 'nii_filepath': Path to the corresponding nifti file
    - 'gtv_filepath': Path to the GTV mask nifti file
    - 'brain_filepath': Path to the Brain mask nifti file
What we want to do is go through each row of the DataFrame and sort the files into directories based on a patient ID.
Generally speakking, the ID is of the format NGSXXXX, where XXXX is a number.
The ID is isolated from the filepaths in different ways depending on the file type:
    - in 'RegPath', the ID is 'raw/NGSXXXX/..'
    - in the remaining columns, the ID is 'YYYY_NGSXXXX/..'
Once we isolate the ID, we will create a directory for that ID if it does not already exist and we copy the files into that directory.
'''

def extract_patient_id(file_path, is_registration=False):
    """
    Extracts the patient ID from the file path.
    If is_registration is True, the ID is extracted from 'raw/NGSXXXX/...'.
    Otherwise, it is extracted from 'YYYY_NGSXXXX/...'.
    """
    if not isinstance(file_path, str) or not file_path:
        return None
    if is_registration:
        # Extract ID from 'raw/NGSXXXX/...'
        return file_path.split('/')[1]
    else:
        # Extract ID from 'YYYY_NGSXXXX/...'
        return file_path.split('/')[0].split('_')[1]

def sort_files_by_patient_id(df):
    """
    Sorts files into directories based on patient ID.
    Creates directories if they do not exist and copies files into them.
    """
    for index, row in df.iterrows():
        # Extract patient ID
        reg_id = extract_patient_id(row['RegPath'], is_registration=True)
        nii_id = extract_patient_id(row['nii_filepath'])
        gtv_id = extract_patient_id(row['gtv_filepath'])
        brain_id = extract_patient_id(row['brain_filepath'])

        # Create directory for the patient ID if it does not exist
        patient_dir = os.path.join('sorted_files', reg_id)
        os.makedirs(patient_dir, exist_ok=True)

        # Copy files to the patient's directory
        for col, file_path in zip(
            ['RegPath', 'nii_filepath', 'gtv_filepath', 'brain_filepath'],
            [row['RegPath'], row['nii_filepath'], row['gtv_filepath'], row['brain_filepath']]
        ):
            if pd.notna(file_path) and file_path:
                if col == 'RegPath':
                    full_path = os.path.join('../Data/', file_path)
                else:
                    full_path = os.path.join('../Data/proc/', file_path)
                if os.path.exists(full_path):
                    dest_path = os.path.join(patient_dir, os.path.basename(file_path))
                    if not os.path.exists(dest_path):
                        shutil.copy2(full_path, dest_path)
                else:
                    print(f"File not found: {full_path}")
            else:
                print(f"File not found: {file_path}")

def main():
    # Sort files by patient ID
    sort_files_by_patient_id(df_path)
    print("Files sorted by patient ID successfully.")
if __name__ == "__main__":
    main()
# This script sorts DICOM and NIfTI files into directories based on patient IDs extracted from their file paths.
# It creates directories for each patient ID and moves the files into the corresponding directories.
# Ensure you have the necessary libraries installed
# (pandas, os) and that the file paths in the DataFrame are correct.
# Make sure to run this script in the same directory where 'regDat.csv' is located.
# Ensure the script is run in an environment where the necessary libraries are installed.