from utils.audio_utils import transcribe_audio, save_transcript, save_segments_json

class TranscriptionAgent:
    def __init__(self, state):
        self.state = state

    def run(self, audio_path: str):
        print("📝 Transcribing audio...")
        transcript_data = transcribe_audio(audio_path)

        # Save to state manager
        self.state.set_state("transcript_data", transcript_data)
        self.state.set_state("transcript_text", transcript_data["text"])
        self.state.set_state("segments", transcript_data["segments"])

        # Save transcript to file
        transcript_file_path = save_transcript(transcript_data)
        self.state.set_state("transcript_file", transcript_file_path)

        # Save segments in JSON format
        json_file_path = save_segments_json(transcript_data)
        self.state.set_state("segments_json_file", json_file_path)

        print("✅ Transcription completed.")
        return transcript_data