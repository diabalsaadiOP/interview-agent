from core.state_mangement import StateManager
from agents.audio_extraction_agent import AudioExtractionAgent
import os

if __name__ == "__main__":
    state = StateManager()
    audio_agent = AudioExtractionAgent(state)

    # Use the actual video file in the data directory
    video_path = os.path.join("data", "interview.mp4")

    try:
        audio_path = audio_agent.run(video_path)
        print(f"Audio extraction completed successfully!")
        print(f"Audio file saved at: {audio_path}")
    except Exception as e:
        print(f"Error during audio extraction: {e}")
