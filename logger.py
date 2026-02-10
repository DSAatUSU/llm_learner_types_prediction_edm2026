"""Logging module for storing raw LLM responses."""

import json
import os
import re
from datetime import datetime
from typing import Dict, Any
from config import LOG_DIR  # Import from config

class LLMLogger:
    """Logger for storing raw LLM responses."""
    
    def __init__(self, model_name: str = "unknown", log_dir: str = None):
        """
        Initialize logger.
        
        Args:
            model_name: Name of the LLM model (e.g., "gemini-3-pro-preview")
            log_dir: Directory to store log files (defaults to LOG_DIR from config)
        """
        self.model_name = model_name
        self.model_name_safe = self._sanitize_model_name(model_name)
        
        # Use config LOG_DIR if not specified
        if log_dir is None:
            self.log_dir = LOG_DIR
        else:
            self.log_dir = log_dir
        
        self.log_file = os.path.join(self.log_dir, f"logs_{self.model_name_safe}.json")  # Changed to .json
        self.text_log_file = os.path.join(self.log_dir, f"logs_{self.model_name_safe}_readable.txt")
        
        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize or load existing logs
        self.logs = self._load_logs()
    
    def _load_logs(self) -> Dict[str, Any]:
        """Load existing logs from JSON file."""
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                print(f"Warning: Could not load log file {self.log_file}, starting fresh.")
                return {}
        return {}

    @staticmethod
    def _sanitize_model_name(model_name: str) -> str:
        """
        Make model name safe for filenames across platforms.
        Replaces path separators and other unsafe characters with underscores.
        """
        if not model_name:
            return "unknown"
        # Replace path separators first to avoid nested directories
        safe = model_name.replace("/", "_").replace("\\", "_")
        # Replace any remaining unsafe characters
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", safe)
        # Avoid empty or dot-only names
        safe = safe.strip("._")
        return safe or "unknown"
    
    def _save_logs(self):
        """Save logs to file."""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(self.logs, f, indent=2, ensure_ascii=False)
    
    def log_response(self, session_key: str, prompt: str, response: str, 
                     parsed_output: Dict[str, Any], metadata: Dict[str, Any] = None):
        """
        Log an LLM response.
        
        Args:
            session_key: Unique identifier for the session
            prompt: The prompt sent to the LLM
            response: Raw response from the LLM
            parsed_output: Parsed output (label, rationale, etc.)
            metadata: Additional metadata (provider, model, timestamp, etc.)
        """
        if metadata is None:
            metadata = {}
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "session_key": session_key,
            "prompt": prompt,
            "response": response,
            "parsed_output": parsed_output,
            "metadata": {
                **metadata,
                "model": self.model_name,
            }
        }
        
        # Store using session_key as the key
        self.logs[session_key] = log_entry
        
        # Also append to a human-readable text file
        self._append_to_text_log(log_entry)
        
        # Save the structured logs
        self._save_logs()
    
    def _append_to_text_log(self, log_entry: Dict[str, Any]):
        """Append log entry to human-readable text file."""
        text_log_file = os.path.join(self.log_dir, f"logs_{self.model_name_safe}_readable.txt")
        
        with open(text_log_file, 'a', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"Timestamp: {log_entry['timestamp']}\n")
            f.write(f"Session: {log_entry['session_key']}\n")
            f.write(f"Model: {log_entry['metadata'].get('model', 'unknown')}\n")
            f.write("-" * 40 + "\n")
            f.write("PROMPT:\n")
            f.write(log_entry['prompt'][:1000] + ("..." if len(log_entry['prompt']) > 1000 else "") + "\n")
            f.write("-" * 40 + "\n")
            f.write("RESPONSE:\n")
            f.write(log_entry['response'] + "\n")
            f.write("-" * 40 + "\n")
            f.write("PARSED OUTPUT:\n")
            f.write(json.dumps(log_entry['parsed_output'], indent=2, ensure_ascii=False) + "\n")
            f.write("=" * 80 + "\n\n")
    
    def get_log(self, session_key: str) -> Dict[str, Any]:
        """Get log entry for a specific session."""
        return self.logs.get(session_key, {})
    
    def get_all_logs(self) -> Dict[str, Any]:
        """Get all logs."""
        return self.logs
    
    def clear_logs(self):
        """Clear all logs (use with caution!)."""
        self.logs = {}
        self._save_logs()
        # Also clear text log
        text_log_file = os.path.join(self.log_dir, f"logs_{self.model_name_safe}_readable.txt")
        if os.path.exists(text_log_file):
            os.remove(text_log_file)

# Global logger instance (optional, for convenience)
_global_logger = None

def get_logger(model_name: str = "unknown") -> LLMLogger:
    """Get or create a global logger instance."""
    global _global_logger
    if _global_logger is None or _global_logger.model_name != model_name:
        _global_logger = LLMLogger(model_name)
    return _global_logger
