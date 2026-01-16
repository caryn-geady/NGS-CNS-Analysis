import argparse
import os
import pandas as pd

#!/usr/bin/env python3
"""
unify_keys.py

Create a unified key mapping between anonymized NGS and CNS IDs using MRN as the join key.

Usage:
    python unify_keys.py --ngs-file ngs.csv --cns-file cns.csv \
        --mrn-col MRN --ngs-col NGS_ID --cns-col CNS_ID \
        --out unified_mapping.csv

Outputs:
 - unified_mapping.csv : rows with UnifiedID, MRN, NGS_ID, CNS_ID
 - ngs_to_unified.csv  : NGS_ID -> UnifiedID
 - cns_to_unified.csv  : CNS_ID -> UnifiedID
"""

def read_table(path):
        ext = os.path.splitext(path)[1].lower()
        if ext in (".xls", ".xlsx"):
                return pd.read_excel(path, dtype=str)
        return pd.read_csv(path, dtype=str)

def write_csv(df, path):
        df.to_csv(path, index=False)

def main():
        p = argparse.ArgumentParser(description="Create unified key from two MRN mappings")
        p.add_argument("--ngs-file", required=True, help="File mapping MRN -> NGS (csv or xlsx)")
        p.add_argument("--cns-file", required=True, help="File mapping MRN -> CNS (csv or xlsx)")
        p.add_argument("--mrn-col", default="MRN", help="Column name for MRN in both files")
        p.add_argument("--ngs-col", default="NGS_ID", help="Column name for NGS ID")
        p.add_argument("--cns-col", default="CNS_ID", help="Column name for CNS ID")
        p.add_argument("--merge", choices=("inner","left","right","outer"), default="inner",
                                     help="Merge type when joining on MRN (default: inner)")
        p.add_argument("--dropna", action="store_true", help="Drop rows missing NGS or CNS after merge")
        p.add_argument("--out", default="unified_mapping.csv", help="Output CSV base path")
        args = p.parse_args()

        a = read_table(args.ngs_file)
        b = read_table(args.cns_file)

        if args.mrn_col not in a.columns:
                raise SystemExit(f"MRN column '{args.mrn_col}' not found in {args.ngs_file}")
        if args.mrn_col not in b.columns:
                raise SystemExit(f"MRN column '{args.mrn_col}' not found in {args.cns_file}")
        if args.ngs_col not in a.columns:
                raise SystemExit(f"NGS column '{args.ngs_col}' not found in {args.ngs_file}")
        if args.cns_col not in b.columns:
                raise SystemExit(f"CNS column '{args.cns_col}' not found in {args.cns_file}")

        a = a[[args.mrn_col, args.ngs_col]].drop_duplicates().rename(columns={args.mrn_col: "MRN", args.ngs_col: "NGS_ID"})
        b = b[[args.mrn_col, args.cns_col]].drop_duplicates().rename(columns={args.mrn_col: "MRN", args.cns_col: "CNS_ID"})

        merged = pd.merge(a, b, on="MRN", how=args.merge)

        if args.dropna:
               merged = merged.dropna(subset=["NGS_ID","CNS_ID"])

        # Create a UnifiedID per unique MRN in the merged set
        unique_mrns = merged["MRN"].dropna().unique()
        uid_map = {mrn: f"UNIFIED_{i:06d}" for i, mrn in enumerate(sorted(unique_mrns), start=1)}
        merged["UnifiedID"] = merged["MRN"].map(uid_map)

        # Reorder columns
        out_df = merged[["UnifiedID", "MRN", "NGS_ID", "CNS_ID"]]

        base = os.path.splitext(args.out)[0]
        unified_path = base + ".csv"
        ngs_map_path = base + "_ngs_to_unified.csv"
        cns_map_path = base + "_cns_to_unified.csv"
        ngs_to_cns_path = base + "_ngs_to_cns.csv"

        write_csv(out_df, unified_path)

        # NGS -> Unified (deduplicated)
        ngs_map = out_df[["NGS_ID", "UnifiedID"]].dropna().drop_duplicates()
        write_csv(ngs_map, ngs_map_path)

        # CNS -> Unified (deduplicated)
        cns_map = out_df[["CNS_ID", "UnifiedID"]].dropna().drop_duplicates()
        write_csv(cns_map, cns_map_path)

        # NGS -> CNS mapping (direct), deduplicated
        ngs_to_cns = out_df[["NGS_ID", "CNS_ID"]].dropna().drop_duplicates()
        write_csv(ngs_to_cns, ngs_to_cns_path)

        print(f"Wrote: {unified_path}")
        print(f"Wrote: {ngs_map_path}")
        print(f"Wrote: {cns_map_path}")
        print(f"Wrote: {ngs_to_cns_path}")
        print(f"Rows in unified mapping: {len(out_df)}")

if __name__ == "__main__":
        main()