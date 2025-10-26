from utils.audio_utils import transcribe_audio, save_transcript, save_segments_json
from agents.diarization_agent import DiarizationAgent


class TranscriptionAgent:
    def __init__(self, state):
        self.state = state
        self.diarization_agent = DiarizationAgent(state)

    def run(self, audio_path: str):
        print("📝 Transcribing audio...")
        transcript_data = transcribe_audio(audio_path)

        # Step 2: Perform speaker diarization
        print("🎭 Performing speaker diarization...")
        enriched_segments = self.diarization_agent.analyze_speakers(
            transcript_data["segments"]
        )

        # Update transcript data with speaker information
        transcript_data["segments"] = enriched_segments

        # Save to state manager
        self.state.set_state("transcript_data", transcript_data)
        self.state.set_state("transcript_text", transcript_data["text"])
        self.state.set_state("segments", enriched_segments)

        # Save transcript to file
        transcript_file_path = save_transcript(transcript_data)
        self.state.set_state("transcript_file", transcript_file_path)

        # Save segments in JSON format (now with speaker info)
        json_file_path = save_segments_json(transcript_data)
        self.state.set_state("segments_json_file", json_file_path)

        print("✅ Transcription and diarization completed.")
        return transcript_data
