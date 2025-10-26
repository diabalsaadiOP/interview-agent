"""
Base Agent Class for Multi-Speaker Video Analysis System

Provides abstract base class for all agents with:
- StateManager integration for shared memory
- Standardized logging and error handling
- Common interface for agent execution
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from datetime import datetime
import logging


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the system.

    All agents should inherit from this class and implement the execute() method.
    Provides standardized state management, logging, and error handling.
    """

    def __init__(self, state_manager, agent_name: str):
        """
        Initialize the base agent.

        Args:
            state_manager: StateManager instance for shared state
            agent_name: Name of the agent for logging purposes
        """
        self.state_manager = state_manager
        self.agent_name = agent_name
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Set up logger for the agent."""
        logger = logging.getLogger(self.agent_name)
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'[%(asctime)s] {self.agent_name} - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def log(self, message: str, level: str = "info"):
        """
        Log a message with the specified level.

        Args:
            message: Message to log
            level: Log level (info, warning, error, debug)
        """
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        log_method(message)

    def update_state(self, key: str, value: Any):
        """
        Update a single value in shared state.

        Args:
            key: State key
            value: State value
        """
        self.state_manager.set_state(key, value)
        self.log(f"Updated state: {key}")

    def get_state(self, key: str) -> Optional[Any]:
        """
        Get a value from shared state.

        Args:
            key: State key

        Returns:
            State value or None if not found
        """
        return self.state_manager.get_state(key)

    def log_execution_start(self):
        """Log the start of agent execution."""
        self.log(f"Starting execution...", "info")
        self.update_state(f"{self.agent_name}_start_time", datetime.now().isoformat())

    def log_execution_end(self, success: bool = True):
        """
        Log the end of agent execution.

        Args:
            success: Whether execution was successful
        """
        status = "completed successfully" if success else "failed"
        self.log(f"Execution {status}", "info")
        self.update_state(f"{self.agent_name}_end_time", datetime.now().isoformat())
        self.update_state(f"{self.agent_name}_status", "success" if success else "failed")

    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the agent's main logic.

        This method must be implemented by all subclasses.

        Args:
            **kwargs: Agent-specific input parameters

        Returns:
            Dictionary containing agent results
        """
        pass

    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Run the agent with error handling and logging.

        Args:
            **kwargs: Agent-specific input parameters

        Returns:
            Dictionary containing agent results or error information
        """
        self.log_execution_start()

        try:
            result = self.execute(**kwargs)
            self.log_execution_end(success=True)
            return result

        except Exception as e:
            self.log(f"Error during execution: {str(e)}", "error")
            self.log_execution_end(success=False)

            error_result = {
                "success": False,
                "error": str(e),
                "agent": self.agent_name
            }
            self.update_state(f"{self.agent_name}_error", str(e))
            return error_result
