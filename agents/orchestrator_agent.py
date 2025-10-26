from agents.audio_extraction_agent import AudioExtractionAgent

class OrchestratorAgent:
    def __init__(self, state):
        self.state = state
        self.audio_agent = AudioExtractionAgent(state)

    def run(self, video_path: str):
        audio_path = self.audio_agent.run(video_path)
        self.state.set_state("audio_path", audio_path)
        print(f"Audio extraction completed successfully!")
        print(f"Audio file saved at: {audio_path}")
