"""Main script for LLM profiling."""

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import os
import json
from typing import List, Optional
from datetime import datetime
from config import (
    LLMConfig,
    RANDOM_SEED,
    OUTPUT_CSV,
    CACHE_CSV,
    FEATS_CSV,
    SNAPSHOTS_CSV,
    RATIONALE_CSV,
    DATA_DIR,
    LOG_DIR,
)
from data_loader import load_and_merge_data
from prompt_builder import build_prompt_for_session
from llm_client import llm_classify
from utils import parse_llm_output, make_session_key, prompt_hash
from cache_manager import load_cache, save_cache
from logger import get_logger

# Set random seed
np.random.seed(RANDOM_SEED)

def check_data_files():
    """Check if required data files exist."""
    required_files = [
        (FEATS_CSV, "Features CSV"),
        (SNAPSHOTS_CSV, "Snapshots CSV"),
        (RATIONALE_CSV, "Rationale Excel"),
    ]
    
    missing_files = []
    for filepath, description in required_files:
        if not os.path.exists(filepath):
            missing_files.append((filepath, description))
    
    if missing_files:
        print("ERROR: Missing required data files:")
        for filepath, description in missing_files:
            print(f"  - {description}: {filepath}")
        print(f"\nMake sure to place files in the 'data/' directory.")
        return False
    
    print("All required data files found.")
    return True

def run_llm_profiling(
    df: pd.DataFrame,
    provider: str,
    model: str,
    output_csv: str = OUTPUT_CSV,
    cache_csv: str = CACHE_CSV,
    limit_sessions: Optional[int] = None,
    only_these_students: Optional[List[str]] = None,
    only_these_exercises: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Run LLM profiling on student sessions."""
    
    work = df.copy()
    
    # Optional filters
    if only_these_students is not None:
        work = work[work["user_id"].isin(only_these_students)]
    if only_these_exercises is not None:
        work = work[work["exercise_id"].isin(only_these_exercises)]
    
    # Define grouping for "a session"
    group_cols = ["user_id", "exercise_id"]
    
    grouped = list(work.groupby(group_cols))
    if limit_sessions is not None:
        grouped = grouped[:limit_sessions]
    
    print(f"Grouping by {group_cols}")
    print(f"Total sessions to process: {len(grouped)}")
    
    # Load cache
    cache_df = load_cache(cache_csv)
    cache_map = {}
    for _, r in cache_df.iterrows():
        cache_key = (r["session_key"], r["prompt_hash"], r["provider"], r["model"])
        cache_map[cache_key] = r
    
    # Initialize logger for this model
    logger = get_logger(model_name=model)
    
    cfg = LLMConfig(provider=provider, model=model, temperature=0.1, max_tokens=800)
    
    results = []
    pbar = tqdm(grouped, desc="LLM profiling", total=len(grouped))
    
    for key_vals, sess_df in pbar:
        # Build session_key
        if isinstance(key_vals, tuple):
            key_dict = dict(zip(group_cols, key_vals))
            session_key = "|".join([str(key_dict[c]) for c in group_cols])
        else:
            session_key = str(key_vals)
        
        # Build prompt
        prompt = build_prompt_for_session(sess_df, model_name=model)
        print(prompt)
        ph = prompt_hash(prompt)
        
        cache_key = (session_key, ph, provider, model)
        if cache_key in cache_map:
            cached = cache_map[cache_key]
            # Parse cached data
            key_evidence = cached["key_evidence"]
            if isinstance(key_evidence, str) and key_evidence.startswith("["):
                try:
                    key_evidence = json.loads(key_evidence)
                except:
                    key_evidence = []
            
            parsed = {
                "label": cached["label"],
                "rationale": cached["rationale"],
                "key_evidence": key_evidence,
                "raw": cached.get("raw", ""),
            }
            
            # Log cached response too (but mark it as cached)
            metadata = {
                "provider": provider,
                "model": model,
                "cached": True,
                "timestamp": datetime.now().isoformat(),
            }
            logger.log_response(
                session_key=session_key,
                prompt=prompt,
                response=parsed["raw"],
                parsed_output=parsed,
                metadata=metadata
            )
            
        else:
            # Call LLM
            try:
                text = llm_classify(prompt, cfg)
                if text is None:
                    raise ValueError("LLM returned no text content (None)")
                if not isinstance(text, str):
                    text = str(text)
                print(f"\nLLM output for {key_vals}:")
                print(text)
                print("-" * 80)
                
                # Fix: Check if the response is truncated
                original_text = text
                if text.strip().startswith("{") and not text.strip().endswith("}"):
                    print("WARNING: Truncated JSON detected! Attempting to fix...")
                    # Try to complete the JSON
                    text = text.strip() + "}"
                
                parsed = parse_llm_output(text, raw_response=original_text)
                
                # Debug: Print what was parsed
                print(f"Parsed label: {parsed['label']}")
                
                # Log the response
                metadata = {
                    "provider": provider,
                    "model": model,
                    "cached": False,
                    "timestamp": datetime.now().isoformat(),
                    "prompt_hash": ph,
                }
                logger.log_response(
                    session_key=session_key,
                    prompt=prompt,
                    response=original_text,  # Store original response
                    parsed_output=parsed,
                    metadata=metadata
                )
                
                # Update cache
                new_row = {
                    "session_key": session_key,
                    "prompt_hash": ph,
                    "provider": provider,
                    "model": model,
                    "label": parsed["label"],
                    "rationale": parsed["rationale"],
                    "key_evidence": json.dumps(parsed["key_evidence"]),
                    "raw": parsed["raw"],
                }
                cache_df = pd.concat([cache_df, pd.DataFrame([new_row])], ignore_index=True)
                save_cache(cache_df, cache_csv)
                
            except Exception as e:
                print(f"Error calling LLM: {e}")
                parsed = {
                    "label": "error",
                    "rationale": f"LLM call failed: {e}",
                    "key_evidence": [],
                    "raw": "",
                }
                
                # Log the error
                metadata = {
                    "provider": provider,
                    "model": model,
                    "cached": False,
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e),
                }
                logger.log_response(
                    session_key=session_key,
                    prompt=prompt,
                    response=f"ERROR: {e}",
                    parsed_output=parsed,
                    metadata=metadata
                )
        
        # Collect metadata (take from first row)
        r0 = sess_df.iloc[0].to_dict()
        
        out = {
            "session_key": session_key,
            "user_id": r0.get("user_id", "NA"),
            "exercise_id": r0.get("exercise_id", "NA"),
            "session_id": r0.get("session_id", "NA"),
            "exercise_type": r0.get("exercise_type", "NA"),
            "provider": provider,
            "model": model,
            "label": parsed["label"],
            "rationale": parsed["rationale"],
            "key_evidence": json.dumps(parsed["key_evidence"]),
            "prompt_hash": ph,
        }
        results.append(out)
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"\nSaved predictions -> {output_csv}")
    
    # Print logging information
    print(f"\nLogs saved to:")
    print(f"  - Structured: logs/logs_{model}.json (JSON format)")
    print(f"  - Readable: logs/logs_{model}_readable.txt (text format)")
    
    # Show sample of predictions
    print("\nSample predictions:")
    print(results_df[["user_id", "exercise_id", "label"]].head())

    # At the end of the function, add file size info
    if os.path.exists(output_csv):
        size = os.path.getsize(output_csv)
        print(f"Output file size: {size / 1024:.1f} KB")
    
    return results_df

def analyze_logs(model_name: str = None):
    """Analyze and summarize logs for debugging."""
    if model_name is None:
        # Find all log files in logs directory
        import glob
        log_pattern = os.path.join(LOG_DIR, "logs_*.json")
        log_files = glob.glob(log_pattern)
        model_names = [os.path.basename(f)[5:-5] for f in log_files]
        
        if not model_names:
            print("No log files found in logs/ directory")
            return
        
        for model in model_names:
            print(f"\n{'='*60}")
            print(f"Analyzing logs for model: {model}")
            print('='*60)
            _analyze_single_model_logs(model)
    else:
        _analyze_single_model_logs(model_name)

def _analyze_single_model_logs(model_name: str):
    """Analyze logs for a specific model."""
    logger = get_logger(model_name)
    logs = logger.get_all_logs()
    
    if not logs:
        print(f"No logs found for model: {model_name}")
        return
    
    print(f"Total sessions logged: {len(logs)}")
    
    # Count cached vs fresh calls
    cached_count = 0
    fresh_count = 0
    error_count = 0
    label_distribution = {}
    
    for session_key, log_entry in logs.items():
        metadata = log_entry.get("metadata", {})
        parsed_output = log_entry.get("parsed_output", {})
        
        if metadata.get("cached", False):
            cached_count += 1
        else:
            fresh_count += 1
            
        if "error" in metadata:
            error_count += 1
            
        label = parsed_output.get("label", "unknown")
        label_distribution[label] = label_distribution.get(label, 0) + 1
    
    print(f"\nCall Statistics:")
    print(f"  Fresh API calls: {fresh_count}")
    print(f"  Cached responses: {cached_count}")
    print(f"  Errors: {error_count}")
    
    print(f"\nLabel Distribution:")
    for label, count in sorted(label_distribution.items(), key=lambda x: x[1], reverse=True):
        print(f"  {label}: {count}")
    
    # Show example of truncated responses
    print(f"\nChecking for truncated responses...")
    truncated_count = 0
    for session_key, log_entry in logs.items():
        response = log_entry.get("response", "")
        if response.strip().startswith("{") and not response.strip().endswith("}"):
            truncated_count += 1
            if truncated_count == 1:  # Show first example
                print(f"  Example truncated response from {session_key}:")
                print(f"    {response[:100]}...")
    
    if truncated_count > 0:
        print(f"  Total truncated responses: {truncated_count}")

def main():
    """Main execution function."""
    
    print("=" * 60)
    print("LLM Profiling System")
    print("=" * 60)
    
    # Check for data files
    if not check_data_files():
        return
    
    print("\nLoading data...")
    merged_df = load_and_merge_data()
    
    # Get provider and model from config or user input
    from config import LLM_PROVIDER, LLM_MODEL
    
    print(f"\nUsing LLM Provider: {LLM_PROVIDER}")
    print(f"Using LLM Model: {LLM_MODEL}")
    
    # Ask for confirmation
    response = input("\nProceed with profiling? (y/n): ").strip().lower()
    if response not in ['y', 'yes']:
        print("Profiling cancelled.")
        return
    
    # Run profiling
    print("\nRunning LLM profiling...")
    results = run_llm_profiling(
        merged_df,
        provider=LLM_PROVIDER,
        model=LLM_MODEL,
        limit_sessions=1,  # Start with small number for testing
    )
    
    # Analyze logs
    print("\n" + "="*60)
    print("Analyzing logs...")
    print("="*60)
    analyze_logs(LLM_MODEL)
    
    # Evaluate predictions
    print("\nEvaluating predictions...")
    try:
        final_labels_path = os.path.join(DATA_DIR, "final_labels.csv")
        if os.path.exists(final_labels_path):
            from evaluator import evaluate_predictions
            evaluate_predictions(
                pred_csv=OUTPUT_CSV,
                true_csv=final_labels_path
            )
        else:
            print(f"Note: final_labels.csv not found in {DATA_DIR}")
            print("Skipping evaluation. To evaluate, add final_labels.csv to data directory.")
    except Exception as e:
        print(f"Evaluation failed: {e}")
    
    print("\n" + "="*60)
    print("Profiling complete!")
    print("="*60)

if __name__ == "__main__":
    main()
