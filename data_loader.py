"""Data loading and preprocessing functions."""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from config import FEATS_CSV, SNAPSHOTS_CSV, RATIONALE_CSV

def load_and_merge_data() -> pd.DataFrame:
    """
    Load and merge features, snapshots, and rationale data.
    """
    feats_df = pd.read_csv(FEATS_CSV)
    snapshots_df = pd.read_csv(SNAPSHOTS_CSV)
    rationale_df = pd.read_excel(RATIONALE_CSV)
    
    # Drop common junk col
    for df in (feats_df, snapshots_df):
        if "Unnamed: 0" in df.columns:
            df.drop(columns=["Unnamed: 0"], inplace=True)
    
    # Parse datetime columns if present
    for col in ["snapshot_time", "timestamp"]:
        if col in snapshots_df.columns:
            snapshots_df[col] = pd.to_datetime(snapshots_df[col], errors="coerce")
    
    # Merge logic
    merge_keys = ["user_id", "exercise_id"]
    
    merged_df = pd.merge(snapshots_df, feats_df, on=merge_keys, how="left")
    merged_df = pd.merge(merged_df, rationale_df, on=merge_keys, how="left")
    
    print("snapshots_df:", snapshots_df.shape)
    print("feats_df    :", feats_df.shape)
    print("merged_df   :", merged_df.shape)
    
    # sanity check
    required_cols = ["user_id", "exercise_id", "code_snapshot"]
    for c in required_cols:
        if c not in merged_df.columns:
            raise ValueError(f"Missing required column '{c}' in merged_df. Available columns: {merged_df.columns.tolist()[:30]} ...")
    
    return merged_df

def downsample_evenly(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    """Evenly spaced downsample; always includes first and last rows."""
    if len(df) <= max_rows:
        return df.copy()
    
    idx = np.linspace(0, len(df) - 1, num=max_rows)
    idx = np.unique(np.round(idx).astype(int))
    return df.iloc[idx].reset_index(drop=True)

def detect_feature_blocks(
    session_df: pd.DataFrame,
    ignore_cols: Optional[List[str]] = None,
) -> Tuple[List[str], List[str]]:
    """
    Automatically split columns into:
      - session-level (constant within session)
      - snapshot-level (vary within session)
    """
    if ignore_cols is None:
        ignore_cols = []
    
    cols = [c for c in session_df.columns if c not in ignore_cols]
    
    session_cols = []
    snapshot_cols = []
    
    for c in cols:
        # skip obviously unhelpful text blobs / ids in feature lists
        if c in ["code_snapshot", "inserted_text", "deleted_text", "raw_log", "full_text"]:
            continue
        
        # If all values are NA, skip
        if session_df[c].isna().all():
            continue
        
        nun = session_df[c].nunique(dropna=True)
        if nun <= 1:
            session_cols.append(c)
        else:
            snapshot_cols.append(c)
    
    return session_cols, snapshot_cols

def select_top_cols(cols: List[str], max_n: int) -> List[str]:
    """Keep order but cap length."""
    return cols[:max_n]