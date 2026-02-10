"""Configuration settings for LLM profiling."""

import os
from dataclasses import dataclass
from typing import List, Dict, Any
# ----------------------------
# Base Directory (ALWAYS DEFINE THIS FIRST)
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------------------
# Path Configuration
# ----------------------------
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# Create directories if they don't exist
for directory in [DATA_DIR, OUTPUT_DIR, LOG_DIR]:
    os.makedirs(directory, exist_ok=True)

# ----------------------------
# LLM TOGGLE (edit these)
# ----------------------------
LLM_PROVIDER = "gemini"   # one of: "openai", "deepseek", "together", "groq", "ollama", "custom_openai_compat"
LLM_MODEL = "gemini-3-pro-preview"  # gemini-3-pro-preview, gemini-3-flash-preview, 

TEMPERATURE = 0.1
MAX_TOKENS = 800        # classification + rationale + evidence

# Snapshot prompt controls
MAX_SNAPSHOTS_PER_SESSION = 20   # downsample snapshots (evenly spaced)
# Gemini 3 Pro preview can return empty parts when prompt is too long
GEMINI_PRO_MAX_SNAPSHOTS_PER_SESSION = 8
MAX_CODE_CHARS_PER_SNAPSHOT = 2500  # truncate code per snapshot
MAX_FEATURES_PER_BLOCK = 50      # cap number of features shown in each block (session/snapshot)

# Files (using absolute paths)
FEATS_CSV = os.path.join(DATA_DIR, "all_features_df.csv")
SNAPSHOTS_CSV = os.path.join(DATA_DIR, "thinned_snapshots_60s.csv")
RATIONALE_CSV = os.path.join(DATA_DIR, "rationale_from_llm.xlsx")

OUTPUT_CSV = os.path.join(OUTPUT_DIR, "llm_profile_predictions.csv")
CACHE_CSV = os.path.join(OUTPUT_DIR, "llm_profile_cache.csv")

RANDOM_SEED = 42

# ----------------------------
# Labels Definition
# ----------------------------
LABELS: List[Dict[str, str]] = [
    {
        "name": "incremental_builder",
        "description": (
            "Progresses steadily through small, iterative code changes with minimal errors. "
            "Usually low AI feedback reliance. Low to moderate executes. Confident behviour. Less time spent."
            "Example of Snapshot Rationale: The student builds the solution in a linear, incremental way: defining the function, adding base cases, then completing the recursive return, and finally adding a simple main call. Changes are small and logically sequenced across evenly spaced snapshots, with no major rewrites, experimentation bursts, or external help-seeking behaviors."
            "Session Statistics: {'total_snapshots': 6,  'total_duration_seconds': 329.886, 'keystrokes_total': 1323, 'ai_feedback_total': 0,  'executes_total': 2}"
            "Evidence from Session: ['Progression is stepwise: function header  'base cases'  'recursive return'  'main() call', 'Snapshots occur at regular ~10s intervals with consistent small edits rather than large pasted blocks or frequent backtracking', 'No AI feedback requests, executions, or submissions; code evolves through straightforward construction rather than trial-and-error validation']"
        ),
    },
    {
        "name": "plan_then_burst_implementer",
        "description": (
            "Reflection before action; Writes significant code blocks after planning pauses. Writes in chunks instead of line-by-line."
            "Example of Snapshot Rationale: The student builds the solution in a small number of logical steps: defining the recursive function skeleton, completing the base case and recursive return, then adding input handling and the final print call. The code evolves steadily without exploratory rewrites, external help, or repeated execution/submission cycles, indicating a methodical construction approach."
            "Session Statistics: {'total_snapshots': 4,  'total_duration_seconds': 139.922, 'keystrokes_total': 631, 'ai_feedback_total': 0,  'executes_total': 0}"
            "Evidence from Session: ['Clear incremental progression: function header' base case/recursive step → input' final output', 'No AI feedback requests, no executions, and no submissions suggests focused implementation rather than experimentation', 'Few snapshots with coherent additions and minimal restructuring (only completing missing pieces)']"
        ),
    },
    {
        "name": "tinkerer_explorer",
        "description": (
            "Low to Moderate feedback calls. Large amounts of executes. Frequent code changes with exploratory edits. Not always relying on running code to validate but also uses AI feedback."
            "Example of Snapshot Rationale: The student iterates through multiple incomplete/incorrect Fibonacci implementations and fixes issues through successive attempts (missing helper function, syntax error, then corrected base case and recursion). The short session with several executions and frequent saves suggests they are testing and adjusting code experimentally rather than planning a full solution upfront."
            "Session Statistics: {'total_snapshots': 4,  'total_duration_seconds': 166.176, 'keystrokes_total': 865, 'ai_feedback_total': 0,  'executes_total': 6}"
            "Evidence from Session: ['Snapshot 2 introduces a helper fib() but fibonacci() calls fib() without ensuring full base-case coverage (fib lacks return for n>1), indicating an initial flawed attempt', 'Snapshot 3 switches to direct recursion but contains a syntax error (missing colon after elif n==1), then corrects it in Snapshot 4', 'Behavioral pattern: 6 code executions and 14 saves in a 166s session, consistent with rapid test-and-fix experimentation']"
        ),
    },
    {
        "name": "debugger_centric_fixer",
        "description": (
            "Usually have a high amount of executes, low to moderate AI feedback calls. The studen is relying on running the code to identify issues and fix them iteratively rather than planning or AI feedback."
            "Example of Snapshot Rationale: The student iteratively experiments with multiple approaches to flattening (list comprehension, then partial loop-based accumulation, then a corrected recursive accumulator), with several intermediate non-working states and quick revisions. Progress converges through debugging and incremental fixes rather than starting from a clear complete plan, and there is no sign of heavy external dependence (only 2 AI requests, no submissions)."
            "Session Statistics: {'total_snapshots': 9,  'total_duration_seconds': 579.754, 'keystrokes_total': 3479, 'ai_feedback_total': 2,  'executes_total': 10}"
            "Evidence from Session: ['Multiple exploratory rewrites: list comprehension attempt debug prints and abandoned comprehension (Snapshot 5) incomplete loop and variable mistakes, final working recursive accumulation', 'Frequent execution/testing behavior (10 executions) during a short session (~580s), consistent with experimenting to converge on a solution', 'Code evolution shows fixing errors via iteration (e.g., returning scalar vs list, correcting isinstance check to (list, tuple), switching to always return a list in base case)']"
        ),
    },
    {
        "name": "strategy_shifter",
        "description": (
            "Makes radical code changes/rewrites, abandoning earlier approaches or mental models. "
            "Example of Snapshot Rationale: The student repeatedly abandons and rewrites the core function, cycling through multiple approaches (split/reverse, incomplete recursion call, index-swapping, and list reconstruction) before arriving at a working solution. Progress is driven by experimentation and backtracking rather than a stable incremental plan, with several snapshots reverting to an empty function body and then trying a different strategy."
            "Session Statistics: {'total_snapshots': 12,  'total_duration_seconds': 499.873, 'keystrokes_total': 2063, 'ai_feedback_total': 1,  'executes_total': 5}"
            "Evidence from Session: ['Multiple major approach shifts: split()+reverse(), incomplete recursion attempt' iterative concatenation (Snapshot 4) → index-swapping idea (Snapshots 6/8/11) → final list-to-string build (Snapshot 12)', 'Frequent resets/backtracks where recursive_reverse is cleared and restarted', 'High edit activity in a short session (25 saves, 2063 keystrokes in ~500s) with limited executions (5) and no submissions, consistent with local tinkering and iterative guessing']"
        ),
    },
    {
        "name": "ai_guided_integrator",
        "description": (
            "Uses AI feedback and then applies targeted fixes toward convergence. Shows corrections after AI calls with moderate executes. Validates AI suggestions through code runs. Shows a pattern of seeking help and then integrating that help effectively. No overreliance on AI."
            "Example of Snapshot Rationale: The student iterates through many partial, syntactically incorrect class definitions (e.g., misusing `super()`, misspelling `Square`, malformed `def`), gradually converging to a correct inheritance-based solution by repeated experimentation. The code evolution shows exploratory edits and corrections rather than a planned structure, with multiple executions and frequent saves but no final submission."
            "Session Statistics: {'total_snapshots': 10,  'total_duration_seconds': 929.9, 'keystrokes_total': 2152, 'ai_feedback_total': 5,  'executes_total': 9}"
            "Evidence from Session: ['Early snapshots contain fragmented/incorrect attempts (e.g., `class Shape: .super`, `super().__init__sh`, `def ()`) that are repeatedly rewritten into a working class design', 'Progression shows convergence via incremental fixes: correcting class name from `Sqaure` to `Square`, moving `super().__init__(color)` into a proper `__init__`, and fixing `self.length = length` before computing area', 'Behavioral pattern supports tinkering: 33 saves and 9 executions within ~930s, indicating repeated test-and-adjust cycles rather than a single planned build']"
        ),
    },
    {
        "name": "low_validation_completer",
        "description": (
            "Minimal testing/iteration and early stopping; may over-rely on AI. Student seems to be relying on AI without multiple revisions and large executes. Shows student is basically taking AI for it's word and not really putting a lot of effort. The student doesn't seems to converge to a final solution."
            "Example of Snapshot Rationale: The student repeatedly experiments with Python class/inheritance syntax (e.g., misusing `super()` in parameters, changing class headers, moving `super().__init__` around) without reaching a stable, correct structure. The snapshots show iterative guessing and patching of errors (switching between `side` vs `self.side`, altering constructors) rather than a planned build, and there are multiple executions but no final submission."
            "Session Statistics: {'total_snapshots': 7,  'total_duration_seconds': 429.811, 'keystrokes_total': 1475, 'ai_feedback_total': 5,  'executes_total': 5}"
            "Evidence from Session: ['Multiple exploratory rewrites of class definitions and inheritance syntax across snapshots (e.g., `def __init__(self, super(), side)`, `class Shape(color)`, `super().def __init__(color)`)', 'Frequent small edits attempting to fix runtime/attribute issues (changing `return side * side` to `return self.side * self.side`), indicating convergence by experimentation', '5 code executions and 5 AI feedback requests within a short session (429.8s) but no submission, consistent with iterative trial-based debugging rather than completing a planned solution']"
        ),
    },
]

def format_label_options() -> str:
    return "\n".join([f"- {l['name']}: {l['description']}" for l in LABELS])

def label_names() -> List[str]:
    return [l["name"] for l in LABELS]

@dataclass
class LLMConfig:
    provider: str
    model: str
    temperature: float = 0.1
    max_tokens: int = 800

IGNORE_FOR_SPLIT = [
    # identity / grouping
    "user_id", "exercise_id", "session_id",
    # timing + snapshot index-ish
    "snapshot_time", "timestamp",
]
