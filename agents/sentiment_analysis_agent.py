
import os
import json
from utils.ollama import query_ollama


class SentimentAnalysisAgent:
    def __init__(self, state):
        self.state = state

    def run(self, transcript_text: str, segments_path: str = "data/interview_segments.json", output_md_path: str = None):
        """
        Analyze sentiment of each segment using LLM and output results in markdown format, with summary, highlights, and recommendations.
        """
        # Load segments
        with open(segments_path, 'r') as f:
            segments = json.load(f)

        segment_results = []
        sentiment_counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
        for i, seg in enumerate(segments):
            print(f"Analyzing sentiment for segment {i+1}/{len(segments)}...")
            prompt = (
                f"Analyze the sentiment of the following interview segment. "
                f"Return a one-word sentiment (Positive, Negative, Neutral) and a short explanation.\n\n"
                f"Segment {i+1}: {seg['text']}"
            )
            llm_response = query_ollama(prompt)
            # Try to extract the one-word sentiment
            sentiment = "Neutral"
            explanation = llm_response.strip()
            for s in ["Positive", "Negative", "Neutral"]:
                if s.lower() in llm_response.lower():
                    sentiment = s
                    break
            sentiment_counts[sentiment] += 1
            segment_results.append({
                "index": i+1,
                "text": seg['text'],
                "llm_sentiment": sentiment,
                "explanation": explanation
            })

        # Find key positive/negative segments
        key_positive = [r for r in segment_results if r["llm_sentiment"] == "Positive"][:2]
        key_negative = [r for r in segment_results if r["llm_sentiment"] == "Negative"][:2]

        # Generate strengths, improvements, and recommendations using LLM
        summary_prompt = (
            "Given the following interview segments and their sentiment analysis, "
            "summarize the candidate's strengths, areas for improvement, and provide 2-3 actionable recommendations.\n\n"
            "Segments and Sentiments:\n" +
            "\n".join([f"Segment {r['index']}: {r['text']}\nSentiment: {r['llm_sentiment']}\nExplanation: {r['explanation']}" for r in segment_results])
        )
        summary_response = query_ollama(summary_prompt)

        # Compose markdown
        md_content = "# Sentiment Analysis Report\n\n"
        md_content += "## Overall Sentiment Summary\n"
        md_content += "| Sentiment | Count |\n|-----------|-------|\n"
        md_content += f"| 😊 Positive | {sentiment_counts['Positive']} |\n"
        md_content += f"| 😐 Neutral  | {sentiment_counts['Neutral']} |\n"
        md_content += f"| 😞 Negative | {sentiment_counts['Negative']} |\n\n"

        # Visual timeline
        timeline = ''.join([
            "😊" if r["llm_sentiment"] == "Positive" else "😞" if r["llm_sentiment"] == "Negative" else "😐"
            for r in segment_results
        ])
        md_content += f"**Sentiment Timeline:** {timeline}\n\n"

        # Key moments
        if key_positive:
            md_content += "### Key Positive Moments\n"
            for r in key_positive:
                md_content += f"> {r['text']}\n\n"
        if key_negative:
            md_content += "### Key Negative Moments\n"
            for r in key_negative:
                md_content += f"> {r['text']}\n\n"

        # LLM-generated summary
        md_content += "## Candidate Strengths, Improvements, and Recommendations\n"
        md_content += summary_response + "\n\n"

        # Per-segment details
        md_content += "---\n\n## Segment-by-Segment Details\n\n"
        for res in segment_results:
            md_content += f"### Segment {res['index']}\n"
            md_content += f"> {res['text']}\n\n"
            md_content += f"- **LLM Sentiment:** {res['llm_sentiment']}\n"
            md_content += f"- **Explanation:** {res['explanation']}\n\n"
        md_content += "---\n\n"
        md_content += f"Total segments analyzed: {len(segment_results)}\n"

        if output_md_path:
            with open(output_md_path, 'w') as f:
                f.write(md_content)
            self.state.set_state('sentiment_md_file', output_md_path)
        return md_content
