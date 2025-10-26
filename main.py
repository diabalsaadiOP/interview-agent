from core.state_mangement import StateManager
from agents.orchestrator_agent import OrchestratorAgent
import os

if __name__ == "__main__":
    state = StateManager()
    orchestrator_agent = OrchestratorAgent(state)

    # Use the actual video file in the data directory
    video_path = os.path.join("data", "interview.mp4")

    try:
        audio_path = orchestrator_agent.run(video_path)

    except Exception as e:
        print(f"Error during audio extraction: {e}")
