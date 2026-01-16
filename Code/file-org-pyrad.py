import os
import pandas as pd

def create_dataframe(base_path):
    """Create a DataFrame with image/mask pairs and ROI information."""
    data = []

    # Iterate through subfolders
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        
        if not os.path.isdir(folder_path):
            continue  # Skip files, only process directories
        
        # Define paths for images
        ct_image_path = os.path.join(folder_path, 'CT.nii.gz')
        mr_image_path = os.path.join(folder_path, 'MR_REG.nii_N4.nii.gz')
        
        # Define paths for masks
        brain_mask_path = None
        tumor_mask_path = None
        peritumoral_mask_path = os.path.join(folder_path, 'peritumoral.nii.gz')
        
        # Search for brain and tumor masks (case-insensitive, must end with .nii.gz)
        for file_name in os.listdir(folder_path):
            lower = file_name.lower()
            if not lower.endswith('.nii.gz'):
                continue
            if 'brain' in lower:
                brain_mask_path = os.path.join(folder_path, file_name)
            elif 'gtv' in lower:
                tumor_mask_path = os.path.join(folder_path, file_name)
        
        # Ensure all required files exist
        if not os.path.exists(ct_image_path) or not os.path.exists(mr_image_path):
            print(f"Missing CT or MR image in folder: {folder_name}")
            continue
        if not brain_mask_path or not tumor_mask_path or not os.path.exists(peritumoral_mask_path):
            print(f"Missing required masks in folder: {folder_name}")
            continue
        
        # Add rows to the DataFrame for each image/mask pair
        data.append({'USUBJID': folder_name, 'ROI': 'Brain', 'Image': ct_image_path, 'Mask': brain_mask_path})
        data.append({'USUBJID': folder_name, 'ROI': 'Tumor', 'Image': ct_image_path, 'Mask': tumor_mask_path})
        data.append({'USUBJID': folder_name, 'ROI': 'Peritumoral', 'Image': ct_image_path, 'Mask': peritumoral_mask_path})
        data.append({'USUBJID': folder_name, 'ROI': 'Brain', 'Image': mr_image_path, 'Mask': brain_mask_path})
        data.append({'USUBJID': folder_name, 'ROI': 'Tumor', 'Image': mr_image_path, 'Mask': tumor_mask_path})
        data.append({'USUBJID': folder_name, 'ROI': 'Peritumoral', 'Image': mr_image_path, 'Mask': peritumoral_mask_path})

    # Create a DataFrame
    df = pd.DataFrame(data)
    return df

# Base path to the folders
base_path = '/Users/caryngeady/Documents/GitHub/NGS-CNS-Analysis/Cohort2'

# Create the DataFrame
df = create_dataframe(base_path)

# Save the DataFrame to a CSV file
df.to_csv('pyrad-cohort2.csv', index=False)

# Print the DataFrame
print(df)