import whisper
import os
import json
from typing import Dict, Any, List


def transcribe_audio(audio_path: str, model_size: str = "base") -> Dict[str, Any]:
    """
    Transcribe audio file using OpenAI Whisper

    Args:
        audio_path (str): Path to the audio file
        model_size (str): Whisper model size ("tiny", "base", "small", "medium", "large")

    Returns:
        Dict containing transcript text, segments in JSON format, and metadata
    """
    try:
        # Check if audio file exists
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        print(f"Loading Whisper model ({model_size})...")
        model = whisper.load_model(model_size)

        print(f"Transcribing audio from {audio_path}...")
        result = model.transcribe(audio_path)

        # Convert segments to the requested JSON format
        formatted_segments = []
        for segment in result["segments"]:
            formatted_segments.append(
                {
                    "start": round(segment["start"], 2),
                    "end": round(segment["end"], 2),
                    "text": segment["text"].strip(),
                }
            )

        # Extract key information
        transcript_data = {
            "text": result["text"].strip(),
            "language": result["language"],
            "segments": formatted_segments,  # Now in the requested format
            "audio_path": audio_path,
        }

        print(f"Transcription completed. Detected language: {result['language']}")
        print(f"Transcript length: {len(transcript_data['text'])} characters")
        print(f"Number of segments: {len(formatted_segments)}")

        return transcript_data

    except Exception as e:
        print(f"Error during transcription: {str(e)}")
        raise e


def save_transcript(transcript_data: Dict[str, Any], output_path: str = None) -> str:
    """
    Save transcript to a text file

    Args:
        transcript_data: Dictionary containing transcript data
        output_path: Optional path for output file

    Returns:
        Path to the saved transcript file
    """
    try:
        if output_path is None:
            # Generate output path based on audio file
            audio_path = transcript_data["audio_path"]
            base_name = os.path.splitext(audio_path)[0]
            output_path = f"{base_name}_transcript.txt"

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"Language: {transcript_data['language']}\n")
            f.write(f"Audio file: {transcript_data['audio_path']}\n")
            f.write("-" * 50 + "\n\n")
            f.write(transcript_data["text"])

            # Add segment timestamps if available
            if transcript_data.get("segments"):
                f.write("\n\n" + "=" * 50 + "\n")
                f.write("DETAILED SEGMENTS WITH SPEAKERS:\n")
                f.write("=" * 50 + "\n\n")

                for i, segment in enumerate(transcript_data["segments"]):
                    start_time = format_timestamp(segment["start"])
                    end_time = format_timestamp(segment["end"])
                    speaker = segment.get("speaker", "UNKNOWN")
                    confidence = segment.get("confidence", 0.0)
                    
                    f.write(f"[{start_time} - {end_time}] {speaker} (conf: {confidence:.2f}): {segment['text']}\n")

        print(f"Transcript saved to: {output_path}")
        return output_path

    except Exception as e:
        print(f"Error saving transcript: {str(e)}")
        raise e


def save_segments_json(transcript_data: Dict[str, Any], output_path: str = None) -> str:
    """
    Save transcript segments in JSON format

    Args:
        transcript_data: Dictionary containing transcript data
        output_path: Optional path for output JSON file

    Returns:
        Path to the saved JSON file
    """
    try:
        if output_path is None:
            # Generate output path based on audio file
            audio_path = transcript_data["audio_path"]
            base_name = os.path.splitext(audio_path)[0]
            output_path = f"{base_name}_segments.json"

        # Save segments in the requested format
        segments = transcript_data["segments"]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, indent=4, ensure_ascii=False)

        print(f"Segments JSON saved to: {output_path}")
        return output_path

    except Exception as e:
        print(f"Error saving segments JSON: {str(e)}")
        raise e


def format_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS format"""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"
