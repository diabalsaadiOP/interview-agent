import os
import subprocess


class AudioExtractionAgent:
    def __init__(self, state_manager):
        self.state_manager = state_manager

    def run(self, video_path: str):
        # Logic to extract audio from video
        audio_path = self.extract_audio(video_path)
        self.state_manager.set_state("audio_path", audio_path)
        return audio_path

    def extract_audio(self, video_path: str) -> str:
        """Extract audio from video file using ffmpeg"""
        try:
            # Check if video file exists
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")

            # Generate output audio path
            base_name = os.path.splitext(video_path)[0]
            extracted_audio_path = f"{base_name}.wav"

            print(f"Extracting audio from {video_path}...")

            # Use ffmpeg to extract audio
            command = [
                "ffmpeg",
                "-i",
                video_path,
                "-vn",  # No video
                "-acodec",
                "pcm_s16le",  # Audio codec
                "-ar",
                "16000",  # Sample rate (16kHz is good for speech)
                "-ac",
                "1",  # Mono audio
                "-y",  # Overwrite output file if it exists
                extracted_audio_path,
            ]

            # Run ffmpeg command
            result = subprocess.run(command, capture_output=True, text=True)

            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg failed with error: {result.stderr}")

            # Check if output file was created
            if not os.path.exists(extracted_audio_path):
                raise RuntimeError("Audio extraction failed - output file not created")

            print(f"Audio successfully extracted to {extracted_audio_path}")
            return extracted_audio_path

        except Exception as e:
            print(f"Error extracting audio: {str(e)}")
            raise e
