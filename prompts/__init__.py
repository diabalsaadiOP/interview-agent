"""
Prompts Module

Contains prompt templates for LLM-based agents.
"""

from prompts.role_identification import (
    get_role_identification_prompt,
    get_verification_prompt,
    get_summary_prompt,
    get_simple_prompt,
    ROLE_IDENTIFICATION_COT_PROMPT,
    ROLE_VERIFICATION_COV_PROMPT,
    ROLE_SUMMARY_PROMPT,
    SIMPLE_ROLE_IDENTIFICATION_PROMPT
)

__all__ = [
    "get_role_identification_prompt",
    "get_verification_prompt",
    "get_summary_prompt",
    "get_simple_prompt",
    "ROLE_IDENTIFICATION_COT_PROMPT",
    "ROLE_VERIFICATION_COV_PROMPT",
    "ROLE_SUMMARY_PROMPT",
    "SIMPLE_ROLE_IDENTIFICATION_PROMPT"
]
