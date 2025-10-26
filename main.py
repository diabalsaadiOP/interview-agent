from core.state_mangement import StateManager
from agents.orchestrator_agent import OrchestratorAgent
import os

if __name__ == "__main__":
    state = StateManager()
    orchestrator_agent = OrchestratorAgent(state)

    # Use the actual video file in the data directory
    video_path = os.path.join("data", "interview.mp4")

    try:
        success = orchestrator_agent.run(video_path)
        
        if success:
            print("\n🎯 ACCESS YOUR DATA:")
            print("=" * 30)
            print(f"📁 Audio: {state.get_state('audio_path')}")
            print(f"📝 Transcript: {state.get_state('transcript_file')}")
            print(f"📋 JSON Segments: {state.get_state('segments_json_file')}")
            print(f"🗣️  Language: {state.get_state('detected_language')}")
            print(f"📊 Total Segments: {state.get_state('segment_count')}")
        else:
            print("❌ Pipeline failed")

    except Exception as e:
        print(f"Error during pipeline execution: {e}")
