"""Cache management for LLM responses."""

import pandas as pd
import os
import json

def load_cache(path: str) -> pd.DataFrame:
    """Load cache from CSV file."""
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=["session_key", "prompt_hash", "provider", "model", "label", "rationale", "key_evidence", "raw"])

def save_cache(df: pd.DataFrame, path: str) -> None:
    """Save cache to CSV file."""
    df.to_csv(path, index=False)