"""
Test Role Identification with Custom User Data

This test uses the exact data provided by the user.
"""

import os

from core.state_mangement import StateManager
from agents.role_identification import RoleIdentificationAgent

# User's custom transcript data
CUSTOM_TRANSCRIPT = [
    {
        "speaker": "SPEAKER_00",
        "start": 0.52,
        "end": 5.87,
        "confidence": 0.96,
        "text_snippet": "Good morning, thank you for joining us today."
    },
    {
        "speaker": "SPEAKER_01",
        "start": 6.10,
        "end": 14.40,
        "confidence": 0.93,
        "text_snippet": "Thank you for having me. I'm excited to discuss my background."
    },
    {
        "speaker": "SPEAKER_00",
        "start": 14.75,
        "end": 22.22,
        "confidence": 0.95,
        "text_snippet": "Can you walk me through your experience leading a project team?"
    },
    {
        "speaker": "SPEAKER_01",
        "start": 22.45,
        "end": 40.10,
        "confidence": 0.92,
        "text_snippet": "Absolutely. In my last role, I managed a cross-functional team of six engineers..."
    }
]


def normalize_transcript(transcript):
    """
    Normalize the transcript format.
    The agent expects 'text' field, but user data has 'text_snippet'.
    """
    normalized = []
    for segment in transcript:
        normalized_segment = {
            "speaker": segment["speaker"],
            "start": segment["start"],
            "end": segment["end"],
            "text": segment.get("text", segment.get("text_snippet", ""))
        }
        normalized.append(normalized_segment)
    return normalized


def main():
    print("=" * 80)
    print("CUSTOM DATA TEST - Role Identification")
    print("=" * 80)
    print()

    print("Input Data:")
    print("-" * 80)
    for i, segment in enumerate(CUSTOM_TRANSCRIPT, 1):
        print(f"{i}. {segment['speaker']} [{segment['start']:.2f}s - {segment['end']:.2f}s]")
        print(f"   Confidence: {segment['confidence']}")
        print(f"   Text: \"{segment['text_snippet']}\"")
        print()

    print("=" * 80)
    print()

    # Normalize the data
    print("[1] Normalizing data format...")
    normalized_transcript = normalize_transcript(CUSTOM_TRANSCRIPT)
    print(f"✓ Converted {len(normalized_transcript)} segments")
    print(f"  (Changed 'text_snippet' → 'text' field for agent compatibility)")
    print()

    # Initialize
    print("[2] Initializing Role Identification Agent...")
    state_manager = StateManager()
    state_manager.set_state("merged_transcript", normalized_transcript)

    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        raise RuntimeError(
            "Set the GEMINI_API_KEY environment variable before running this test."
        )

    agent = RoleIdentificationAgent(
        state_manager=state_manager,
        gemini_api_key=gemini_api_key,
        model_name="gemini-2.5-flash"
    )
    print("✓ Agent initialized")
    print()

    # Execute
    print("[3] Running Role Identification...")
    print("  → Phase 1: Chain of Thought (CoT)")
    print("  → Phase 2: Chain of Verification (CoV)")
    print("  → Phase 3: Summary Generation")
    print()

    result = agent.run(use_verification=True)

    # Results
    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()

    if result.get("success"):
        print("✓ SUCCESS")
        print()

        # Role assignments
        print("IDENTIFIED ROLES:")
        print("-" * 80)
        role_assignments = result.get("role_assignments", {})

        for speaker_id, data in sorted(role_assignments.items()):
            role = data.get("role", "Unknown")
            confidence = data.get("confidence", "N/A")
            print(f"  {speaker_id}")
            print(f"    Role: {role}")
            print(f"    Confidence: {confidence}")
            print()

        # Statistics
        print("SPEAKER STATISTICS:")
        print("-" * 80)
        speaker_stats = result.get("speaker_stats", {})

        for speaker_id, stats in sorted(speaker_stats.items()):
            print(f"  {speaker_id}")
            print(f"    Segments: {stats.get('num_segments', 0)}")
            print(f"    Speaking time: {stats.get('total_duration', 0):.2f}s")
            print(f"    Avg segment: {stats.get('avg_segment_length', 0):.2f}s")
            print()

        # Summary
        print("SUMMARY:")
        print("-" * 80)
        summary = result.get("summary", "")
        print(summary)
        print()

        # Show reasoning
        print("=" * 80)
        print("AI REASONING (from CoT)")
        print("=" * 80)
        cot_response = state_manager.get_state("role_identification_cot_response")
        if cot_response:
            print(cot_response[:800])
            print()
            print(f"... (showing first 800 of {len(cot_response)} characters)")
        print()

    else:
        print("✗ FAILED")
        print(f"Error: {result.get('error', 'Unknown error')}")
        print()

    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
