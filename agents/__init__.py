"""
Agents Module

Contains all agent implementations for the multi-speaker video analysis system.
"""

from agents.base_agent import BaseAgent
from agents.role_identification import RoleIdentificationAgent

__all__ = [
    "BaseAgent",
    "RoleIdentificationAgent",
]
