from agents.audio_extraction_agent import AudioExtractionAgent
from agents.transcription_agent import TranscriptionAgent

class OrchestratorAgent:
    def __init__(self, state):
        self.state = state
        self.audio_agent = AudioExtractionAgent(state)
        self.transcription_agent = TranscriptionAgent(state)

    def run(self, video_path: str):
        print("\n🚀 Starting multi-speaker analysis pipeline...\n")

        # Step 1: Extract audio
        print("Step 1: Audio Extraction")
        audio_path = self.audio_agent.run(video_path)
        print(f"✅ Audio extracted to: {audio_path}\n")

        # Step 2: Transcribe audio
        print("Step 2: Audio Transcription")
        transcript_data = self.transcription_agent.run(audio_path)
        print(f"✅ Transcription completed\n")

        # Print summary
        print("📊 PIPELINE SUMMARY:")
        print("=" * 50)
        print(f"🎥 Video file: {video_path}")
        print(f"🎵 Audio file: {audio_path}")
        print(f"📝 Transcript file: {self.state.get_state('transcript_file')}")
        print(f"� Segments JSON: {self.state.get_state('segments_json_file')}")
        print(f"�🗣️  Detected language: {transcript_data['language']}")
        print(f"📏 Transcript length: {len(transcript_data['text'])} characters")
        print(f"🔢 Number of segments: {len(transcript_data['segments'])}")
        print("=" * 50)

        return {
            "audio_path": audio_path,
            "transcript_data": transcript_data,
            "transcript_file": self.state.get_state('transcript_file'),
            "segments_json_file": self.state.get_state('segments_json_file')
        }

