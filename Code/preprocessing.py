import os
import shutil

def extract_ngs_ids(folder, pattern_type):
    ngs_ids = set()
    for name in os.listdir(folder):
        if os.path.isdir(os.path.join(folder, name)):
            if pattern_type == 'with_prefix':
                # Format: XXXX__NGSYYYY
                parts = name.split('__')
                if len(parts) == 2 and parts[1].startswith('NGS'):
                    ngs_ids.add(parts[1])
            elif pattern_type == 'no_prefix':
                # Format: NGSYYYY
                if name.startswith('NGS'):
                    ngs_ids.add(name)
    return ngs_ids

# folder paths
folder1 = '../Data/proc'  # Contains XXXX__NGSYYYY
folder2 = '/Users/caryngeady/Desktop/NGS-Data/PMCC_NGS-CNS'  # Contains NGSYYYY

ids1 = extract_ngs_ids(folder1, 'with_prefix')
ids2 = extract_ngs_ids(folder2, 'no_prefix')

intersection = ids1 & ids2
only_in_folder1 = ids1 - ids2
only_in_folder2 = ids2 - ids1

print("Intersection:", intersection)
print("Only in folder1:", only_in_folder1)
print("Only in folder2:", only_in_folder2)

# %%
# dest_folder = '../Data/proc/raw2'
# os.makedirs(dest_folder, exist_ok=True)

# for ngs_id in only_in_folder2:
#     src_path = os.path.join(folder2, ngs_id)
#     dest_path = os.path.join(dest_folder, ngs_id)
#     if os.path.isdir(src_path):
#         shutil.copytree(src_path, dest_path, dirs_exist_ok=True)
# %%
