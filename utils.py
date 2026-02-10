"""Utility functions for parsing and processing."""

import json
import hashlib
import re
from typing import Dict, Any, List

import pandas as pd

from config import label_names

VALID_LABELS = set(label_names())

def parse_llm_output(text: str, raw_response: str = None) -> Dict[str, Any]:
    """
    Parse JSON with validation. If label invalid, set label="other_or_uncertain".
    
    Args:
        text: The text to parse (may be cleaned/fixed)
        raw_response: Original raw response (for logging)
    """
    if text is None:
        text = ""
    if raw_response is None:
        raw_response = text
        
    raw_to_store = raw_response  # Store the original raw response
    
    # Fix: The issue was that Gemini returned truncated JSON
    # Check if JSON is incomplete and try to complete it
    if text.strip().startswith("{") and not text.strip().endswith("}"):
        # Try to find a complete JSON object
        start = text.find("{")
        end = text.rfind("}")
        if end > start:
            text = text[start:end+1]
        else:
            # If no closing brace, try to add one
            text = text.strip() + "}"
    
    # Try strict JSON
    data = {}
    parsed_ok = False
    try:
        data = json.loads(text)
        parsed_ok = True
    except Exception:
        # attempt to extract JSON object substring
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                data = json.loads(text[start:end+1])
                parsed_ok = True
            except Exception:
                data = {}
    
    label = str(data.get("label", "")).strip()
    rationale = str(data.get("rationale", "")).strip()
    key_evidence = data.get("key_evidence", [])

    # If JSON parsing failed, attempt to salvage only the label from raw text
    if not parsed_ok:
        label_match = re.search(r'["\']label["\']\s*:\s*["\']([^"\']+)["\']', raw_response or text)
        if label_match:
            label = label_match.group(1).strip()
            rationale = ""
            key_evidence = []
    
    # Clean the label
    if label and label not in VALID_LABELS:
        # Check for partial matches
        for valid_label in VALID_LABELS:
            if valid_label.startswith(label) or label in valid_label:
                label = valid_label
                break
    
    if label and label not in VALID_LABELS:
        # light fuzzy fallback: if label contains a valid label name
        for v in VALID_LABELS:
            if v in label or label in v:
                label = v
                break
    
    if label not in VALID_LABELS:
        label = "other_or_uncertain"
    
    if not isinstance(key_evidence, list):
        key_evidence = [str(key_evidence)]
    
    # cap evidence items
    key_evidence = [str(x)[:300] for x in key_evidence[:6]]
    
    return {
        "label": label,
        "rationale": rationale[:2000],
        "key_evidence": key_evidence,
        "raw": raw_to_store[:8000],  # Store original raw response
    }

def make_session_key(row) -> str:
    """Create a unique session key."""
    # Prefer session_id if present
    if "session_id" in row.index and pd.notna(row["session_id"]):
        return f"{row['user_id']}|{row['exercise_id']}|{row['session_id']}"
    # fallback
    return f"{row['user_id']}|{row['exercise_id']}"

def prompt_hash(prompt: str) -> str:
    """Create hash of prompt for caching."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
