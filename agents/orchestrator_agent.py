
import os
from agents.audio_extraction_agent import AudioExtractionAgent
from agents.transcription_agent import TranscriptionAgent
from agents.sentiment_analysis_agent import SentimentAnalysisAgent
from langchain.schema.runnable import RunnableLambda

def make_audio_extraction_runnable(audio_agent):
    return RunnableLambda(lambda inputs: {
        "audio_path": audio_agent.run(inputs["video_path"])
    })

def make_transcription_runnable(transcription_agent):
    return RunnableLambda(lambda inputs: {
        "transcript_data": transcription_agent.run(inputs["audio_path"])
    })

def make_audio_extraction_runnable(audio_agent):
    return RunnableLambda(lambda inputs: {
        "audio_path": audio_agent.run(inputs["video_path"])
    })

def make_transcription_runnable(transcription_agent):
    return RunnableLambda(lambda inputs: {
        "transcript_data": transcription_agent.run(inputs["audio_path"])
    })


class OrchestratorAgent:

    def __init__(self, state):
        self.state = state
        self.audio_agent = AudioExtractionAgent(state)
        self.transcription_agent = TranscriptionAgent(state)
        self.sentiment_agent = SentimentAnalysisAgent(state)

    def run(self, video_path: str):
        print("\n🚀 Starting multi-speaker analysis pipeline...\n")

        audio_runnable = make_audio_extraction_runnable(self.audio_agent)
        transcription_runnable = make_transcription_runnable(self.transcription_agent)
        chain = audio_runnable | transcription_runnable

        # Run the chain
        result = chain.invoke({"video_path": video_path})
        transcript_data = result["transcript_data"]
        audio_path = result.get("audio_path", self.state.get_state("audio_path") or "(not set)")

        # Sentiment analysis step (LLM, segment-based)
        segments_json_file = self.state.get_state('segments_json_file') or "data/interview_segments.json"
        sentiment_md_path = f"sentiment_results_{os.path.splitext(os.path.basename(video_path))[0]}.md"
        sentiment_md = self.sentiment_agent.run(
            transcript_text=transcript_data['text'],
            segments_path=segments_json_file,
            output_md_path=sentiment_md_path
        )

        # Print summary
        print("📊 PIPELINE SUMMARY:")
        print("=" * 50)
        print(f"🎥 Video file: {video_path}")
        print(f"🎵 Audio file: {audio_path}")
        print(f"📝 Transcript file: {self.state.get_state('transcript_file')}")
        print(f"📋 Segments JSON: {self.state.get_state('segments_json_file')}")
        print(f"🗣️  Detected language: {transcript_data['language']}")
        print(f"📏 Transcript length: {len(transcript_data['text'])} characters")
        print(f"🔢 Number of segments: {len(transcript_data['segments'])}")
        print(f"📄 Sentiment markdown file: {sentiment_md_path}")
        print("=" * 50)

        print("✅ All data saved to state management. Pipeline completed successfully!")
        return True  # Simple success indicator
