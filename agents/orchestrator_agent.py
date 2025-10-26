from agents.audio_extraction_agent import AudioExtractionAgent
from agents.transcription_agent import TranscriptionAgent
from agents.summarization_agent import SummarizationAgent


class OrchestratorAgent:
    def __init__(self, state):
        self.state = state
        self.audio_agent = AudioExtractionAgent(state)
        self.transcription_agent = TranscriptionAgent(state)
        self.summarization_agent = SummarizationAgent(state)

    def run(self, video_path: str):
        print("\n🚀 Starting multi-speaker analysis pipeline...\n")

        # Step 1: Extract audio
        print("Step 1: Audio Extraction")
        audio_path = self.audio_agent.run(video_path)
        print(f"✅ Audio extracted to: {audio_path}\n")

        # Step 2: Transcribe audio & Speaker Diarization
        print("Step 2: Audio Transcription & Speaker Diarization")
        transcript_data = self.transcription_agent.run(audio_path)
        print(f"✅ Transcription and diarization completed\n")

        # Step 3: Interview Analysis and Summarization
        print("Step 3: Interview Analysis & Summarization")
        segments_json_file = self.state.get_state('segments_json_file')
        summary_report = self.summarization_agent.run(segments_json_file)
        print(f"✅ Interview analysis and summarization completed\n")

        # Save additional orchestrator-level state
        self.state.set_state("video_path", video_path)
        self.state.set_state("audio_path", audio_path)
        self.state.set_state("pipeline_completed", True)
        self.state.set_state("detected_language", transcript_data["language"])
        self.state.set_state("segment_count", len(transcript_data["segments"]))

        # Print summary
        print("📊 PIPELINE SUMMARY:")
        print("=" * 50)
        print(f"🎥 Video file: {video_path}")
        print(f"🎵 Audio file: {audio_path}")
        print(f"📝 Transcript file: {self.state.get_state('transcript_file')}")
        print(f"📋 Segments JSON: {self.state.get_state('segments_json_file')}")
        print(f"� Summary Report: {self.state.get_state('summary_json_file')}")
        print(f"📄 Analysis Report: {self.state.get_state('summary_text_file')}")
        print(f"�🗣️  Detected language: {transcript_data['language']}")
        print(f"📏 Transcript length: {len(transcript_data['text'])} characters")
        print(f"🔢 Number of segments: {len(transcript_data['segments'])}")

        # Show diarization summary if available
        speaker_analysis = self.state.get_state("speaker_analysis")
        if speaker_analysis and "summary" in speaker_analysis:
            summary = speaker_analysis["summary"]
            print(f"🎭 Speaker Analysis:")
            print(
                f"   - Interviewer segments: {summary.get('interviewer_segments', 'N/A')}"
            )
            print(
                f"   - Candidate segments: {summary.get('candidate_segments', 'N/A')}"
            )
            print(
                f"   - Average confidence: {summary.get('average_confidence', 0):.2f}"
            )

        # Show summarization insights
        if summary_report:
            print(f"🎯 Interview Analysis:")
            print(f"   - Candidate: {summary_report.get('candidate_name', 'Unknown')}")
            print(f"   - Speaking time: {summary_report.get('candidate_speaking_time', 0):.1f}s")
            print(f"   - Key strengths identified: {len([t for t, d in summary_report.get('topic_analysis', {}).items() if d.get('strength_score', 0) >= 5])}")
            print(f"   - Total recommendations: {len(summary_report.get('recommendations', []))}")

        print("=" * 50)

        print("✅ All data saved to state management. Pipeline completed successfully!")
        return True  # Simple success indicator
