
import os
import json
from utils.ollama import query_ollama


class SentimentAnalysisAgent:
    def __init__(self, state):
        self.state = state

    def run(self, transcript_text: str, segments_path: str = "data/interview_segments.json", output_md_path: str = None):
        """
        Analyze sentiment of each segment using LLM and output results in markdown format.
        """
        # Load segments
        with open(segments_path, 'r') as f:
            segments = json.load(f)

        segment_results = []
        for i, seg in enumerate(segments):
            prompt = (
                f"Analyze the sentiment of the following interview segment. "
                f"Return a one-word sentiment (Positive, Negative, Neutral) and a short explanation.\n\n"
                f"Segment {i+1}: {seg['text']}"
            )
            llm_response = query_ollama(prompt)
            segment_results.append({
                "index": i+1,
                "text": seg['text'],
                "llm_sentiment": llm_response.strip()
            })

        # Compose markdown
        md_content = "# Sentiment Analysis Results by Segment\n\n"
        for res in segment_results:
            md_content += f"## Segment {res['index']}\n"
            md_content += f"> {res['text']}\n\n"
            md_content += f"- **LLM Sentiment:** {res['llm_sentiment']}\n\n"
        md_content += "---\n\n"
        md_content += f"Total segments analyzed: {len(segment_results)}\n"

        if output_md_path:
            with open(output_md_path, 'w') as f:
                f.write(md_content)
            self.state.set_state('sentiment_md_file', output_md_path)
        return md_content
