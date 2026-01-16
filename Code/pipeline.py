# ...existing code...
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

# ---------- I/O / config ----------
CLINICAL_PATH = "merged-clinical.csv"       # adjust as needed
RADIOMICS_PATH = "reduced_radiomics.csv"    # optional
VARS_CORE = ['AGE', 'HISTOL', 'IDH', 'MGMT',
             'Extent of surgical resection', 'BECOG', 'EVENT', 'TIME']

# ---------- Helpers ----------
def preprocess_clinical(df, drop_last_row=True):
    df = df.copy()
    # keep only required vars if present (and preserve AnonMRN so we can rename it)
    keep = [v for v in VARS_CORE if v in df.columns]
    if 'AnonMRN' in df.columns:
        keep.append('AnonMRN')
    df = df[keep].copy()
    # rename AnonMRN to USUBJIC (drop original)
    if 'AnonMRN' in df.columns:
        df['USUBJID'] = df['AnonMRN'].astype(str)
        df = df.drop(columns=['AnonMRN'])
    if drop_last_row and len(df) > 0:
        df = df[:-1]
    # encode
    if 'HISTOL' in df.columns:
        df['HISTOL'] = df['HISTOL'] == 'GBM'
    if 'IDH' in df.columns:
        df['IDH'] = df['IDH'] == 'IDHmut'
    if 'MGMT' in df.columns:
        df['MGMT'] = df['MGMT'] == 'Methylated (M)'
    if 'Extent of surgical resection' in df.columns:
        df['Extent of surgical resection'] = df['Extent of surgical resection'] == 'Total'
    if 'BECOG' in df.columns:
        df['BECOG'] = df['BECOG'] != '0: Asymptomatic'
    if 'EVENT' in df.columns:
        df['EVENT'] = df['EVENT'] == 'Dead'
    if 'TIME' in df.columns:
        df['TIME'] = pd.to_numeric(df['TIME'], errors='coerce')
    return df

# ---------- Cox functions ----------
def fit_cox(df, duration_col='TIME', event_col='EVENT', penalizer=0.001, dropna=True):
    df = df.copy()
    if dropna:
        df = df.dropna()
    # ensure event/duration exist
    if duration_col not in df.columns or event_col not in df.columns:
        raise ValueError("Missing TIME/EVENT columns for Cox fit")
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(df, duration_col=duration_col, event_col=event_col)

    c_index = concordance_index(df[duration_col], -cph.predict_partial_hazard(df), df[event_col])
    return cph, c_index


# ---------- Minimal PCA / radiomics helpers (optional) ----------
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def compute_pca(radiomics_df, n_components=10):
    df = radiomics_df.copy()
    if 'USUBJID' in df.columns:
        ids = df['USUBJID'].reset_index(drop=True)
        X = df.drop(columns=['USUBJID'])
    else:
        ids = None
        X = df
    X = X.select_dtypes(include=[np.number]).fillna(0)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=n_components)
    comps = pca.fit_transform(Xs)
    pca_df = pd.DataFrame(comps, columns=[f'PC{i+1}' for i in range(comps.shape[1])])
    if ids is not None:
        pca_df['USUBJID'] = ids
    return pca_df, pca.explained_variance_ratio_

# ---------- Example main flow ----------
if __name__ == "__main__":
    # load clinical
    clin = preprocess_clinical(pd.read_csv(CLINICAL_PATH))
    # drop any rows with NaNs in the clinical dataframe
    clin = clin.dropna().reset_index(drop=True)

    # optional radiomics: keep only rows matching USUBJID in clin
    try:
        rad = pd.read_csv(RADIOMICS_PATH)
        if 'USUBJID' in rad.columns and 'USUBJID' in clin.columns:
            rad = rad[rad['USUBJID'].isin(clin['USUBJID'])].reset_index(drop=True)
    except Exception:
        rad = None

    # drop USUBJID columns from both dataframes if present
    if 'USUBJID' in clin.columns:
        clin = clin.drop(columns=['USUBJID'])
    if rad is not None and 'USUBJID' in rad.columns:
        rad = rad.drop(columns=['USUBJID'])

    # PCA on radiomics if available
    if rad is not None:
        pca_rad, var_ratios = compute_pca(rad, n_components=10)
    else:
        pca_rad = None
    
    # run Cox model on clinical data
    # set IDH == IDHmut to avoid float issues
    cph_clin, c_index_clin = fit_cox(clin)
    print("Cox model on clinical data:")
    cph_clin.print_summary()
    print(f"Concordance index: {c_index_clin:.4f}")
    
    # run Cox model on clinical + radiomics PCA if available
    if pca_rad is not None:
        clin_rad = pd.concat([clin.reset_index(drop=True), pca_rad.reset_index(drop=True)], axis=1)
        cph_clin_rad, c_index_clin_rad = fit_cox(clin_rad)
        print("\nCox model on clinical + radiomics PCA data:")
        cph_clin_rad.print_summary()
        print(f"Concordance index: {c_index_clin_rad:.4f}")     
        
    # run Cox model on IDHmut subsets (both combined clinical+radiomics PCA and clin-only)
    if 'IDH' in clin.columns:
        # --- IDH == False (IDH wild-type) ---
        # combined (clinical + radiomics PCA if available)
        if pca_rad is not None:
            clin_combined = pd.concat([clin.reset_index(drop=True), pca_rad.reset_index(drop=True)], axis=1)
        else:
            clin_combined = clin.copy()

        clin_idhwt_comb = clin_combined[clin_combined['IDH'] == False].reset_index(drop=True)
        if 'IDH' in clin_idhwt_comb.columns:
            clin_idhwt_comb = clin_idhwt_comb.drop(columns=['IDH'])

        if len(clin_idhwt_comb) == 0:
            print("\nNo IDH==False cases found in the combined data; skipping combined IDHwt fit.")
        else:
            cph_idhwt_comb, c_index_idhwt_comb = fit_cox(clin_idhwt_comb)
            print("\nCox model on IDH==False subset (clinical + radiomics):")
            cph_idhwt_comb.print_summary()
            print(f"Concordance index: {c_index_idhwt_comb:.4f}")

        # clin-only
        clin_idhwt_clin = clin[clin['IDH'] == False].reset_index(drop=True)
        if 'IDH' in clin_idhwt_clin.columns:
            clin_idhwt_clin = clin_idhwt_clin.drop(columns=['IDH'])

        if len(clin_idhwt_clin) == 0:
            print("\nNo IDH==False cases found in the clinical data; skipping clin-only IDHwt fit.")
        else:
            cph_idhwt_clin, c_index_idhwt_clin = fit_cox(clin_idhwt_clin)
            print("\nCox model on IDH==False subset (clinical only):")
            cph_idhwt_clin.print_summary()
            print(f"Concordance index: {c_index_idhwt_clin:.4f}")

        # --- IDH == True (IDHmut) ---
        # combined (clinical + radiomics PCA if available)
        if pca_rad is not None:
            clin_combined = pd.concat([clin.reset_index(drop=True), pca_rad.reset_index(drop=True)], axis=1)
        else:
            clin_combined = clin.copy()

        clin_idhmut_comb = clin_combined[clin_combined['IDH'] == True].reset_index(drop=True)
        if 'IDH' in clin_idhmut_comb.columns:
            clin_idhmut_comb = clin_idhmut_comb.drop(columns=['IDH'])
            print(clin_idhmut_comb.columns)
            clin_idhmut_comb.columns = ['Patient Age', 'Histologic Classification', 'MGMT Methylation Status', 'Extent of Surgical Resection', 'BECOG',
       'EVENT', 'TIME', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8',
       'PC9', 'PC10']

        if len(clin_idhmut_comb) == 0:
            print("\nNo IDH==True cases found in the combined data; skipping combined IDHmut fit.")
        else:
            cph_idhmut_comb, c_index_idhmut_comb = fit_cox(clin_idhmut_comb)
            print("\nCox model on IDH==True subset (clinical + radiomics):")
            cph_idhmut_comb.print_summary()
            print(f"Concordance index: {c_index_idhmut_comb:.4f}")

        # clin-only
        clin_idhmut_clin = clin[clin['IDH'] == True].reset_index(drop=True)
        if 'IDH' in clin_idhmut_clin.columns:
            clin_idhmut_clin = clin_idhmut_clin.drop(columns=['IDH'])

        if len(clin_idhmut_clin) == 0:
            print("\nNo IDH==True cases found in the clinical data; skipping clin-only IDHmut fit.")
        else:
            cph_idhmut_clin, c_index_idhmut_clin = fit_cox(clin_idhmut_clin)
            print("\nCox model on IDH==True subset (clinical only):")
            cph_idhmut_clin.print_summary()
            print(f"Concordance index: {c_index_idhmut_clin:.4f}")
            
# %%

# plot with custom color and transparent background
ax = cph_idhwt_comb.plot()  # lifelines returns a matplotlib Axes

# desired color (replace with any hex)
color = 'white'

# appearance settings
line_width = 3.0       # thicker lines
label_fs = 14          # axis label font size
tick_fs = 12           # tick label font size
legend_fs = 12         # legend font size
spine_width = 1.5      # spine thickness

# set all plot lines to the color and thicker (includes point/coef lines and CI lines typically)
for line in ax.get_lines():
    line.set_color(color)
    line.set_linewidth(line_width)
    # also set marker edge/face if present
    try:
        line.set_markeredgecolor(color)
    except Exception:
        pass
    try:
        line.set_markerfacecolor(color)
    except Exception:
        pass

# some CI elements are drawn as collections (e.g., LineCollection / PatchCollection) or patches
for coll in ax.collections:
    try:
        coll.set_edgecolor(color)
    except Exception:
        pass
    try:
        coll.set_facecolor('none')  # keep CI transparent but edge in target color if applicable
    except Exception:
        pass
for p in ax.patches:
    try:
        p.set_edgecolor(color)
    except Exception:
        pass
    try:
        p.set_facecolor('none')
    except Exception:
        pass

# legend text color and size (if present)
leg = ax.get_legend()
if leg is not None:
    for text in leg.get_texts():
        text.set_color(color)
        text.set_fontsize(legend_fs)
    # legend title (if any)
    try:
        leg.get_title().set_fontsize(legend_fs)
        leg.get_title().set_color(color)
    except Exception:
        pass
    # make legend background transparent
    try:
        leg.get_frame().set_alpha(0.0)
    except Exception:
        pass

# --- Update y-axis labels ---
# Build a mapping from original variable names to nicer labels.
# Edit this dict to suit your variables.
label_map = {
    'AGE': 'Patient Age',
    'HISTOL': 'Histologic Classification',
    'MGMT': 'MGMT Methylation Status',
    'Extent of surgical resection': 'Extent of Surgical Resection',
    'BECOG': 'Baseline ECOG',
    'PC1': 'PC1', 'PC2': 'PC2', 'PC3': 'PC3', 'PC4': 'PC4', 'PC5': 'PC5',
    'PC6': 'PC6', 'PC7': 'PC7', 'PC8': 'PC8', 'PC9': 'PC9', 'PC10': 'PC10'
}

# get current y tick labels (text strings)
orig_yticks = [t.get_text() for t in ax.get_yticklabels()]
# fallback: some backends populate labels differently; try using tick positions' string reprs
if not any(orig_yticks):
    orig_yticks = [str(t) for t in ax.get_yticks()]

# map labels
new_yticks = [label_map.get(lbl, lbl) for lbl in orig_yticks]

# apply new labels (set font size and color)
ax.set_yticklabels(new_yticks, fontsize=tick_fs, color=color)

# axis labels / ticks color and font sizes
ax.xaxis.label.set_color(color)
ax.xaxis.label.set_fontsize(label_fs)
ax.yaxis.label.set_color(color)
ax.yaxis.label.set_fontsize(label_fs)
ax.tick_params(axis='both', colors=color, labelsize=tick_fs)

# spines color and thickness
for spine in ax.spines.values():
    spine.set_color(color)
    spine.set_linewidth(spine_width)

# add vertical line at x=0
ax.axvline(x=0, color=color, linewidth=spine_width, linestyle='--', zorder=0)

# make axes and figure backgrounds transparent
ax.set_facecolor('none')           # axes background
fig = ax.get_figure()
fig.patch.set_alpha(0.0)           # figure background fully transparent

# redraw canvas
fig.canvas.draw()

# %%
