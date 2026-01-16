import sys
import os
import argparse
from pathlib import Path
import SimpleITK as sitk
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation

#!/usr/bin/env python3
"""
Usage:
    python process_folders.py /path/to/top/folder

For each immediate subfolder:
  1) Find a file with "MR_REG" in the filename (case-sensitive). Run N4 bias correction and save with suffix "_N4.nii.gz".
  2) Find one "brain" mask (filename contains "brain" or "BRAIN") and one "GTV" mask (case-insensitive "GTV").
     Dilate the GTV with a 5x5x5 structuring element, subtract the original GTV (producing the peritumoral shell),
     then keep only voxels that are also inside the brain mask. Save as "peritumoral.nii.gz".
Notes:
  - Requires: SimpleITK, nibabel, numpy, scipy
  - If shapes mismatch or files missing, that subfolder is skipped (with a printed warning).
"""

def find_file_with_keyword(files, keywords, case_sensitive=False):
    for f in files:
        name = f.name if case_sensitive else f.name.lower()
        keys = keywords if case_sensitive else [k.lower() for k in keywords]
        if any(k in name for k in keys):
            return f
    return None

def n4_bias_correction(in_path, out_path):
    image = sitk.ReadImage(str(in_path))
    image_cast = sitk.Cast(image, sitk.sitkFloat32)
    # create mask by non-zero voxels
    mask = sitk.BinaryThreshold(image_cast, lowerThreshold=1e-6, upperThreshold=1e9, insideValue=1, outsideValue=0)
    corrected = sitk.N4BiasFieldCorrection(image_cast, mask)
    # Cast back to original pixel type if desired; here keep float32
    sitk.WriteImage(corrected, str(out_path))

def load_nifti_as_array(path):
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    return img, data

def save_nifti_from_array(data, reference_img, out_path, dtype=np.uint8):
    nii = nib.Nifti1Image(data.astype(dtype), affine=reference_img.affine, header=reference_img.header)
    nib.save(nii, str(out_path))

def process_subfolder(subfolder: Path):
    nii_files = [p for p in subfolder.iterdir() if p.is_file() and (p.suffix == '.nii' or p.suffixes == ['.nii', '.gz'] or p.suffix == '.gz')]
    if not nii_files:
        return

    # 1) MR_REG bias correction
    mr_reg = find_file_with_keyword(nii_files, ["MR_REG"], case_sensitive=True)
    if mr_reg:
        out_mr = subfolder / (mr_reg.stem + "_N4.nii.gz")
        if not out_mr.exists():
            print(f"  Bias-correcting: {mr_reg.name} -> {out_mr.name}")
            try:
                n4_bias_correction(mr_reg, out_mr)
            except Exception as e:
                print(f"    N4 failed for {mr_reg.name}: {e}")
        else:
            print(f"  Skipping N4 (exists): {out_mr.name}")
    else:
        print(f"  No MR_REG file found in {subfolder.name}")

    # 2) find brain and GTV masks and create peritumoral mask
    brain_file = find_file_with_keyword(nii_files, ["brain", "BRAIN"], case_sensitive=False)
    gtv_file = find_file_with_keyword(nii_files, ["gtv"], case_sensitive=False)

    if not brain_file or not gtv_file:
        print(f"  Brain or GTV mask missing in {subfolder.name}; skipping mask processing.")
        return

    print(f"  Found brain: {brain_file.name}, GTV: {gtv_file.name}")

    brain_img, brain_data = load_nifti_as_array(brain_file)
    gtv_img, gtv_data = load_nifti_as_array(gtv_file)

    # Threshold to binary masks
    brain_mask = brain_data > 0
    gtv_mask = gtv_data > 0

    if brain_mask.shape != gtv_mask.shape:
        print(f"  Shape mismatch (brain {brain_mask.shape} vs gtv {gtv_mask.shape}) in {subfolder.name}; skipping.")
        return

    struct = np.ones((5,5,5), dtype=bool)
    dilated = binary_dilation(gtv_mask, structure=struct)
    peritumoral_shell = np.logical_and(dilated, np.logical_not(gtv_mask))
    peritumoral_mask = np.logical_and(peritumoral_shell, brain_mask)

    out_name = f"peritumoral.nii.gz"
    out_path = subfolder / out_name
    save_nifti_from_array(peritumoral_mask.astype(np.uint8), gtv_img, out_path)
    print(f"  Saved peritumoral mask: {out_name}")

def main(root_folder):
    root = Path(root_folder)
    if not root.exists() or not root.is_dir():
        print("Invalid folder:", root_folder)
        return

    subfolders = [p for p in root.iterdir() if p.is_dir()]
    if not subfolders:
        print("No subfolders found in", root_folder)
        return

    for sf in sorted(subfolders):
        print("Processing:", sf.name)
        try:
            process_subfolder(sf)
        except Exception as e:
            print(f"  Error processing {sf.name}: {e}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Batch N4 and GTV dilation processing")
    p.add_argument("root", help="Top-level folder containing case subfolders")
    args = p.parse_args()
    main(args.root)
