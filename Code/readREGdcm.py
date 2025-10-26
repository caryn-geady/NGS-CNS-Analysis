import pydicom
import numpy as np
import pandas as pd
import json, os
import nibabel as nib
from scipy.ndimage import affine_transform
import SimpleITK as sitk

'''
PROCESSING STEPS:
1. Identify all REG files in the db output from MIT;
2. From each REG file, construct a DataFrame with the following columns:
    a. SOPInstanceUID
    b. FrameOfReferenceUID
    c. TransformationMatrix
    d. Path to nii file (Same FrameOfReferenceUID and Modality = CT or MR (likely CT));
    e. Path to GTV mask (Same FrameOfReferenceUID and Modality = RTSTRUCT); <-- if this is not available, we can use the GTV mask from the image with the same SOPInstanceUID;
    f. Path to brain mask (Same FrameOfReferenceUID and Modality = RTSTRUCT);
3. For each REG file, find the corresponding fixed and moving images (fixed would be CT and moving would be MR);
4. Use the transformation matrix to transform the moving image to the fixed image space;
5. Perform bias correction to the MR image using N4ITK and save the bias corrected image in the same space as the fixed image; 
6. Perform morphological operations on the GTV mask to isolate the peritumoral region (we're looking for a 2cm margin around the GTV that is contained within the brain mask);
7. Perform radiomic feature extraction on the bias corrected MR image and the CT for the following regions:
    a. GTV
    b. Peritumoral region
    c. Brain mask
8. Save the features in a CSV file.
'''


# MIT output
crawler_output_path = '../Data/.imgtools/raw/crawl_db.json'
with open(crawler_output_path, "r") as f:
    crawler_data = json.load(f)

proc_data = pd.read_csv('../Data/proc/proc_index.csv')

# STEP 1: Extract the REG files from the crawler output
reg_files = []
for key, value in crawler_data.items():
    # Some entries may themselves be dictionaries of studies/series
    if isinstance(value, dict):
        for subkey, subvalue in value.items():
            if isinstance(subvalue, dict) and subvalue.get('Modality') == 'REG':
                folder = subvalue.get('folder')
                instances = subvalue.get('instances', {}).values()
                for instance in instances:
                    reg_files.append(os.path.join(folder, instance))
    # Also check the top-level in case REG is not nested
    if isinstance(value, dict) and value.get('Modality') == 'REG':
        folder = value.get('folder')
        instances = value.get('instances', {}).values()
        for instance in instances:
            reg_files.append(os.path.join(folder, instance))

# STEP 2: Construct the DataFrame with the required columns
if 'reg_df' not in locals():
    reg_df = pd.DataFrame(columns=['RegPath','SOPInstanceUID', 'FrameOfReferenceUID', 'TransformationMatrix'])

for reg_file in reg_files:
    ds = pydicom.dcmread(os.path.join('../Data/',reg_file))

    for i in range(2):
        for_uid = ds.RegistrationSequence[i].FrameOfReferenceUID # FrameOfReferenceUID
        matrix = ds.RegistrationSequence[i].MatrixRegistrationSequence[0].MatrixSequence[0].FrameOfReferenceTransformationMatrix # TransformationMatrix
        mss_uid = ds.SOPInstanceUID # SOPInstanceUID
        reg_df = pd.concat([
            reg_df,
            pd.DataFrame([{
                'RegPath': reg_file,
                'SOPInstanceUID': mss_uid,
                'FrameOfReferenceUID': for_uid,
                'TransformationMatrix': matrix
            }])
        ], ignore_index=True)

# for each row in FrameOfReferenceUID, find the corresponding filepath in proc_data - looking for matching FrameOfReferenceUID and Modality = CT or MR
# Create a list to store the paths to nii files
nii_files = []
gtv_masks = []
brain_masks = []

for ref_id in reg_df['FrameOfReferenceUID']:
    # Find the corresponding rows in proc_data for CT and MR
    matching_rows = proc_data[(proc_data['FrameOfReferenceUID'] == ref_id) & (proc_data['Modality'].isin(['CT', 'MR']))]
    if not matching_rows.empty:
        for _, row in matching_rows.iterrows():
            nii_files.append(row['filepath'])
    else:
        nii_files.append(None)

    # Find the corresponding RTSTRUCT row for GTV mask
    gtv_row = proc_data[
        (proc_data['FrameOfReferenceUID'] == ref_id) &
        (proc_data['Modality'] == 'RTSTRUCT') &
        (proc_data['filepath'].str.contains('GTV', case=False, na=False))
    ]
    if not gtv_row.empty:
        gtv_masks.append(gtv_row.iloc[0]['filepath'])
    else:
        gtv_masks.append(None)

    # Find the corresponding RTSTRUCT row for Brain mask
    brain_row = proc_data[
        (proc_data['FrameOfReferenceUID'] == ref_id) &
        (proc_data['Modality'] == 'RTSTRUCT') &
        (proc_data['filepath'].str.contains('Brain', case=True, na=False)) &
        (~proc_data['filepath'].str.contains('stem', case=False, na=False)) &
        (~proc_data['filepath'].str.contains('minus', case=False, na=False))
    ]
    if not brain_row.empty:
        brain_masks.append(brain_row.iloc[0]['filepath'])
    else:
        brain_masks.append(None)

# Add the new columns to the DataFrame
reg_df['nii_filepath'] = nii_files
reg_df['gtv_filepath'] = gtv_masks
reg_df['brain_filepath'] = brain_masks


# # %%
# # testing the affine transformation - use the second row of the reg_df
# ind = 1
# # Load the image
# fixed_image = nib.load(os.path.join('../Data/proc/', reg_df['nii_filepath'][ind-1]))
# moving_image = nib.load(os.path.join('../Data/proc/', reg_df['nii_filepath'][ind]))

# fixed_data = fixed_image.get_fdata()
# moving_data = moving_image.get_fdata()

# # Get the transformation matrix (assumed to be in DICOM patient coordinates)
# transformation_matrix = np.array(reg_df['TransformationMatrix'][ind]).reshape(4, 4)

# # Compute the affine to apply: from moving image voxel space to fixed image voxel space
# # This is: fixed_affine_inv @ transformation_matrix @ moving_affine
# fixed_affine = fixed_image.affine
# moving_affine = moving_image.affine
# fixed_affine_inv = np.linalg.inv(fixed_affine)
# affine_to_apply = fixed_affine_inv @ transformation_matrix @ moving_affine
# # affine_to_apply = transformation_matrix

# # Apply the affine transformation to the moving image data
# # affine_transform expects the inverse of the transform
# transformed_data = affine_transform(
#     moving_data,
#     np.linalg.inv(affine_to_apply)[:3, :3],
#     offset=np.linalg.inv(affine_to_apply)[:3, 3],
#     # output_shape=fixed_data.shape,
#     order=1
# )

# # %%
# # Save the transformed image as NIfTI
# transformed_img = nib.Nifti1Image(transformed_data, fixed_affine)
# output_path = os.path.join('../Data/proc/', reg_df['nii_filepath'][ind].replace('.nii', '_transformed.nii'))
# os.makedirs(os.path.dirname(output_path), exist_ok=True)
# nib.save(transformed_img, output_path)

# # %%
# # Convert the numpy array to a SimpleITK image (use the fixed image affine for spacing/origin/direction)
# # Transpose the numpy array to match SimpleITK's (x, y, z) axis order
# # Convert the transformed numpy array directly to a SimpleITK image, preserving axis order
# sitk_img = sitk.GetImageFromArray(transformed_data.astype(np.float32), isVector=False)
# # Set spacing, origin, and direction from the fixed image
# fixed_sitk = sitk.ReadImage(os.path.join('../Data/proc/', reg_df['nii_filepath'][ind-1]))
# sitk_img.SetSpacing(fixed_sitk.GetSpacing())
# sitk_img.SetOrigin(fixed_sitk.GetOrigin())
# sitk_img.SetDirection(fixed_sitk.GetDirection())

# # Run N4ITK bias correction
# corrector = sitk.N4BiasFieldCorrectionImageFilter()
# corrected_img = corrector.Execute(sitk_img)

# # Convert back to numpy for saving with nibabel (axis order matches original)
# corrected_data = sitk.GetArrayFromImage(corrected_img)

# # Save the bias-corrected image as NIfTI
# corrected_nifti = nib.Nifti1Image(corrected_data, fixed_affine)
# output_path_corr = os.path.join('../Data/proc/', reg_df['nii_filepath'][ind].replace('.nii', '_transformed_n4.nii'))
# nib.save(corrected_nifti, output_path_corr)

# %%
'''
PROCESSING STEPS (continued):
5. Perform bias correction to the MR image using N4ITK and save the bias corrected image in the same space as the fixed image; 
6. Perform morphological operations on the GTV mask to isolate the peritumoral region (we're looking for a 2cm margin around the GTV that is contained within the brain mask);
7. Perform radiomic feature extraction on the bias corrected MR image and the CT for the following regions:
    a. GTV
    b. Peritumoral region
    c. Brain mask
8. Save the features in a CSV file.
'''


