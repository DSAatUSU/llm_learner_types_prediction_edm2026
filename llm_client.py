"""LLM client implementations for various providers."""

import os
import json
from typing import Tuple
from openai import OpenAI
import together
import anthropic
from dotenv import load_dotenv

from config import LLMConfig

load_dotenv()

# Prefer the new Google GenAI SDK if available; fall back to legacy SDK
try:
    from google import genai
    from google.genai import types as genai_types
    _GENAI_SDK = "new"
except Exception:
    import google.generativeai as genai
    genai_types = None
    _GENAI_SDK = "legacy"

def _get_provider_settings(provider: str) -> Tuple[str, str]:
    """Get provider name and API key."""
    p = provider.lower().strip()
    
    if p == "openai":
        return ("openai", os.getenv("OPENAI_API_KEY"))
    elif p == "deepseek":
        return ("deepseek", os.getenv("DEEPSEEK_API_KEY"))
    elif p == "together":
        return ("together", os.getenv("TOGETHER_API_KEY"))
    elif p == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
        # Remove quotes if present
        if api_key and (api_key.startswith('"') and api_key.endswith('"')) or (api_key.startswith("'") and api_key.endswith("'")):
            api_key = api_key.strip('"').strip("'")
        return ("gemini", api_key)
    elif p == "anthropic":
        return ("anthropic", os.getenv("ANTHROPIC_API_KEY"))
    
    raise ValueError(f"Unknown provider: {provider}")

def make_client(cfg: LLMConfig):
    """Create LLM client for the specified provider."""
    provider, key = _get_provider_settings(cfg.provider)
    
    if provider == "openai":
        return OpenAI(api_key=key)
    elif provider == "deepseek":
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        return OpenAI(api_key=key, base_url=base_url)
    elif provider == "together":
        return together.Together(api_key=key)
    elif provider == "gemini":
        # Gemini client will be created in _call_gemini
        return None
    elif provider == "anthropic":
        return anthropic.Client(api_key=key)
    
    raise ValueError("Could not build client")

def llm_classify(prompt: str, cfg: LLMConfig) -> str:
    """Route classification request to appropriate LLM provider."""
    provider, _ = _get_provider_settings(cfg.provider)
    
    if provider == "openai" or provider == "deepseek":
        return _call_openai(prompt, cfg)
    elif provider == "together":
        return _call_together(prompt, cfg)
    elif provider == "gemini":
        return _call_gemini(prompt, cfg)
    elif provider == "anthropic":
        return _call_anthropic(prompt, cfg)
    
    raise ValueError("Unsupported LLM provider")

def _call_openai(prompt: str, cfg: LLMConfig) -> str:
    """Call OpenAI or DeepSeek API."""
    client = make_client(cfg)
    messages = [
        {"role": "system", "content": "You classify student learning behavior."},
        {"role": "user", "content": prompt},
    ]
    resp = client.chat.completions.create(
        model=cfg.model,
        messages=messages,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens
    )
    return resp.choices[0].message.content

def _call_together(prompt: str, cfg: LLMConfig) -> str:
    """Call Together AI API."""
    client = together.Together(api_key=os.getenv("TOGETHER_API_KEY"))
    resp = client.chat.completions.create(
        model=cfg.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens
    )
    return resp.choices[0].message.content

def _call_gemini(prompt: str, cfg: LLMConfig) -> str:
    """Call Google Gemini API."""
    _, api_key = _get_provider_settings(cfg.provider)
    
    if not api_key:
        raise ValueError("Set GOOGLE_API_KEY")

    def _gemini_debug_info(resp) -> str:
        info = {
            "prompt_feedback": getattr(resp, "prompt_feedback", None),
            "candidates": [],
        }
        for cand in getattr(resp, "candidates", []) or []:
            cand_info = {
                "finish_reason": getattr(cand, "finish_reason", None),
                "safety_ratings": getattr(cand, "safety_ratings", None),
            }
            content = getattr(cand, "content", None)
            if content is not None:
                parts = getattr(content, "parts", None)
                if parts is not None:
                    try:
                        cand_info["parts_count"] = len(parts)
                    except Exception:
                        cand_info["parts_count"] = None
            info["candidates"].append(cand_info)
        return json.dumps(info, default=str)

    if _GENAI_SDK == "new":
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=cfg.model,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                temperature=cfg.temperature,
                max_output_tokens=cfg.max_tokens,
            ),
        )

        text = getattr(response, "text", None)
        if text:
            return text

        parts = []
        for part in getattr(response, "parts", []) or []:
            text_part = getattr(part, "text", None)
            if text_part:
                parts.append(text_part)
        if parts:
            return "\n".join(parts)

        debug_info = _gemini_debug_info(response)
        print(f"Gemini debug info (no text returned): {debug_info}")
        finish_reasons = []
        for candidate in getattr(response, "candidates", []) or []:
            fr = getattr(candidate, "finish_reason", None)
            if fr is not None:
                finish_reasons.append(str(fr))
        detail = ""
        if finish_reasons:
            detail += f" finish_reason={','.join(finish_reasons)}."
        prompt_feedback = getattr(response, "prompt_feedback", None)
        if prompt_feedback:
            detail += f" prompt_feedback={prompt_feedback}."
        raise ValueError(f"Gemini response contained no text content.{detail} debug={debug_info}")

    # Legacy SDK path
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(cfg.model)
    generation_config = genai.GenerationConfig(
        temperature=cfg.temperature,
        max_output_tokens=cfg.max_tokens,
    )
    response = model.generate_content(
        prompt,
        generation_config=generation_config,
    )

    # response.text can raise if no valid Part was returned (e.g., blocked / empty)
    text = None
    try:
        if hasattr(response, "text"):
            text = response.text
    except Exception:
        text = None
    if text:
        return text

    parts = []
    for candidate in getattr(response, "candidates", []):
        content = getattr(candidate, "content", None)
        if not content:
            continue
        for part in getattr(content, "parts", []):
            text_part = getattr(part, "text", None)
            if text_part:
                parts.append(text_part)
    if parts:
        return "\n".join(parts)

    debug_info = _gemini_debug_info(response)
    print(f"Gemini debug info (no text returned): {debug_info}")
    finish_reasons = []
    for candidate in getattr(response, "candidates", []) or []:
        fr = getattr(candidate, "finish_reason", None)
        if fr is not None:
            finish_reasons.append(str(fr))
    detail = ""
    if finish_reasons:
        detail += f" finish_reason={','.join(finish_reasons)}."
    prompt_feedback = getattr(response, "prompt_feedback", None)
    if prompt_feedback:
        detail += f" prompt_feedback={prompt_feedback}."
    raise ValueError(f"Gemini response contained no text content.{detail} debug={debug_info}")

def _call_anthropic(prompt: str, cfg: LLMConfig) -> str:
    """Call Anthropic Claude API."""
    client = anthropic.Client(api_key=os.getenv("ANTHROPIC_API_KEY"))
    completion = client.completions.create(
        model=cfg.model,
        prompt=f"{anthropic.HUMAN_PROMPT}{prompt}{anthropic.AI_PROMPT}",
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
    )
    return completion.text
