"""Prompt building functions for LLM classification."""

import numpy as np
import math
import pandas as pd
from typing import List, Any, Optional
from config import (
    MAX_SNAPSHOTS_PER_SESSION, 
    GEMINI_PRO_MAX_SNAPSHOTS_PER_SESSION,
    MAX_CODE_CHARS_PER_SNAPSHOT, 
    MAX_FEATURES_PER_BLOCK,
    IGNORE_FOR_SPLIT,
    LABELS,
    format_label_options
)
from data_loader import downsample_evenly, detect_feature_blocks, select_top_cols

def _is_scalarish(x: Any) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating, str, bool)) or x is None

def format_value(x: Any) -> str:
    """Readable + stable formatting."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NA"
    if isinstance(x, (np.integer, int)):
        return str(int(x))
    if isinstance(x, (np.floating, float)):
        # avoid scientific noise
        if math.isfinite(float(x)):
            return f"{float(x):.4f}".rstrip("0").rstrip(".")
        return "NA"
    if isinstance(x, bool):
        return "true" if x else "false"
    s = str(x)
    if len(s) > 300:
        return s[:300] + "..."
    return s

def _is_gemini_pro_preview(model_name: Optional[str]) -> bool:
    if not model_name:
        return False
    name = str(model_name).lower()
    return "gemini-3-pro" in name

def _format_label_options_compact() -> str:
    """Compact label list for token-constrained prompts."""
    return "\n".join([f"- {l['name']}" for l in LABELS])

def build_prompt_for_session(session_df: pd.DataFrame, model_name: Optional[str] = None) -> str:
    """
    Build a comprehensive prompt for session classification.
    """
    # Sort chronologically
    if "seconds_from_start" in session_df.columns:
        session_df = session_df.sort_values("seconds_from_start")
    elif "snapshot_time" in session_df.columns:
        session_df = session_df.sort_values("snapshot_time")
    
    # Downsample (tighter for gemini-3-pro-preview)
    max_snaps = GEMINI_PRO_MAX_SNAPSHOTS_PER_SESSION if _is_gemini_pro_preview(model_name) else MAX_SNAPSHOTS_PER_SESSION
    sampled = downsample_evenly(session_df, max_snaps)
    
    # Extract key information
    user_id = str(sampled["user_id"].iloc[0]) if "user_id" in sampled.columns else "NA"
    exercise_id = str(sampled["exercise_id"].iloc[0]) if "exercise_id" in sampled.columns else "NA"
    rationale = str(sampled["rationale"].iloc[0]) if "rationale" in sampled.columns else "NA"
    sess_stats = str(sampled["sess_stats"].iloc[0]) if "sess_stats" in sampled.columns else "NA"
    evidence = str(sampled["evidence"].iloc[0]) if "evidence" in sampled.columns else "NA"
    
    # Detect constant vs varying features
    session_cols, snapshot_cols = detect_feature_blocks(sampled, ignore_cols=IGNORE_FOR_SPLIT)
    session_cols = select_top_cols(session_cols, MAX_FEATURES_PER_BLOCK)
    snapshot_cols = select_top_cols(snapshot_cols, MAX_FEATURES_PER_BLOCK)
    
    # Build SESSION STATISTICS
    session_lines = []
    for c in session_cols:
        val = sampled[c].dropna().iloc[0] if sampled[c].notna().any() else None
        session_lines.append(f"{c}: {format_value(val)}")
    
    # Reorder preferred stats first
    preferred_stats = [
        "total_duration_seconds", "seconds_from_start",
        "keystrokes_before_snapshot", "executes_before_snapshot",
        "submits_before_snapshot", "saves_before_snapshot",
        "ai_feedback_before_snapshot", "chars_in_snapshot",
        "total_snapshots_in_session", "thinned_snapshots_in_session",
    ]
    
    def reorder(lines: List[str]) -> List[str]:
        bucket = []
        rest = []
        for ln in lines:
            key = ln.split(":")[0]
            if key in preferred_stats:
                bucket.append(ln)
            else:
                rest.append(ln)
        bucket_sorted = []
        for k in preferred_stats:
            bucket_sorted.extend([b for b in bucket if b.startswith(k + ":")])
        return bucket_sorted + rest
    
    session_lines = reorder(session_lines)
    
    # Build CODE SNAPSHOTS section
    snap_blocks = []
    for i, (_, row) in enumerate(sampled.iterrows(), start=1):
        # snapshot-level features for this row
        feat_lines = []
        for c in snapshot_cols:
            feat_lines.append(f"{c}: {format_value(row.get(c, None))}")
        
        # timing annotation
        t = row.get("seconds_from_start", None)
        t_str = f"{format_value(t)}" if t is not None else "NA"
        
        code = row.get("code_snapshot", "")
        if not isinstance(code, str):
            code = "" if code is None else str(code)
        if len(code) > MAX_CODE_CHARS_PER_SNAPSHOT:
            code = code[:MAX_CODE_CHARS_PER_SNAPSHOT] + "\n... [TRUNCATED] ..."
        
        snap_blocks.append(
            "\n".join([
                f"FEATURES OF CODE SNAPSHOT {i} (t={t_str} seconds):",
                *([f"- {ln}" for ln in feat_lines] if len(feat_lines) else ["- (no varying features detected)"]),
                "",
                f"CODE SNAPSHOT {i}:",
                "```python",
                code,
                "```",
                "-" * 50
            ])
        )
    
    compact = _is_gemini_pro_preview(model_name)
    labels_text = _format_label_options_compact() if compact else format_label_options()

    if compact:
        prompt = f"""
        You are an expert educational researcher analyzing student programming behavior over time.

        Task:
        - Review session rationale, stats, evidence, and code snapshots.
        - Choose EXACTLY ONE label from the list.
        - Return ONLY valid JSON: {{"label":"one_label_name_from_list"}}.

        Context:
        Student: {user_id}
        Exercise: {exercise_id}\n\n

        ******* SESSION RATIONALE, KEY STATISTICS AND EVIDENCE ******
        Rationale: {rationale} \n\n
        Session Statistics: {sess_stats}\n\n
        Evidence: {evidence} \n\n

        ******* CODE SNAPSHOTS ******
        {"\n".join(snap_blocks)} \n\n

        ****** LABELS ******
        Choose ONE of:
        {labels_text}
        """.strip()
    else:
        prompt = f"""
        You are an expert educational researcher analyzing student programming behavior over time.

        Your task:
        - Read SESSION RATIONALE, KEY STATISTICS AND EVIDENCE and CODE SNAPSHOTS section (each snapshot has its own features + code).
        - Avoid jumping to conclusions based on any mention of label in the session rationale, statistics or evidence. These are summaries of the code evolution obtained using an LLM, focus on all data before deciding a label.
        - Assign EXACTLY ONE label from the label set below. The label set have detailed descriptions providing one example of snapshot rationale, session statistics and evidence from session.
        - Return ONLY valid JSON matching the schema.

        Context:
        Student: {user_id}
        Exercise: {exercise_id}\n\n


        ******* SESSION RATIONALE, KEY STATISTICS AND EVIDENCE ******
        Rationale Provided: {rationale} \n\n

        Session Statistics Provided: {sess_stats}\n\n

        Evidence from Session Provided: {evidence} \n\n


        ******* CODE SNAPSHOTS ******
        {"\n".join(snap_blocks)} \n\n


        ******LABELS*********
        Choose ONE of:
        {labels_text}

        Return JSON ONLY in this exact schema:
        {{
        "label": "one_label_name_from_list",
        }}

        Important:
        - The label MUST be exactly one of the provided label names.
        - Base your decision on BOTH the session statistics AND how the code evolves across snapshots.
        """.strip()
    print(prompt)
    return prompt

    # "rationale": "2-4 sentences explaining why",
    # "key_evidence": ["evidence item 1", "evidence item 2", "evidence item 3"]
