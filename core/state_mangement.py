class StateManager:
    """A simple state management class for agents."""

    def __init__(self):
        self.state = {}

    """Get the value associated with a key in the state."""

    def get_state(self, key):
        return self.state.get(key, None)

    """Set the value associated with a key in the state."""

    def set_state(self, key, value):
        self.state[key] = value

    """Clear the state."""

    def clear_state(self):
        self.state.clear()

    """Get the entire state dictionary."""

    def get_all_state(self):
        return self.state

    """Update multiple key-value pairs in the state."""

    def update_state(self, updates):
        if isinstance(updates, dict):
            self.state.update(updates)
        else:
            raise ValueError("Updates must be provided as a dictionary.")

    """Remove a key from the state."""

    def remove_state(self, key):
        if key in self.state:
            del self.state[key]
        else:
            raise KeyError(f"Key '{key}' not found in state.")
