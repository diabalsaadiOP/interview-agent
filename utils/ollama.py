import subprocess
import json

def query_ollama(prompt: str, model: str = "llama3.2") -> str:
    """Query Ollama locally and return model response."""
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
        )

        output = result.stdout.decode("utf-8").strip()
        return output
    except subprocess.TimeoutExpired:
        return "Error: Ollama query timed out"
    except Exception as e:
        return f"Error: {e}"
