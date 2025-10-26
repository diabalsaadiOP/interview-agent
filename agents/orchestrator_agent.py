from agents.audio_extraction_agent import AudioExtractionAgent

class OrchestratorAgent:
    def __init__(self, state):
        self.state = state
        self.audio_agent = AudioExtractionAgent(state)

    def run(self, video_path: str):
        print("\n🚀 Starting multi-speaker analysis pipeline...\n")
        audio_path = self.audio_agent.run(video_path)

        print("\n✅ Audio extraction completed.")
