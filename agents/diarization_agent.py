from langchain.schema import BaseOutputParser
from langchain.prompts import PromptTemplate
from utils.ollama import query_ollama
import json
import re
from typing import Dict, List, Any


class SpeakerDiarizationParser(BaseOutputParser):
    """Parser for speaker diarization LLM output"""

    def parse(self, text: str) -> Dict[str, Any]:
        """Parse LLM response into structured speaker data"""
        try:
            # Try to extract JSON from the response
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # Fallback parsing if no JSON found
                return {"error": "Could not parse LLM response", "raw_response": text}
        except json.JSONDecodeError:
            return {"error": "Invalid JSON in LLM response", "raw_response": text}


class DiarizationAgent:
    def __init__(self, state_manager):
        self.state_manager = state_manager
        self.parser = SpeakerDiarizationParser()


    def analyze_speakers(self, segments: List[Dict], chunk_size: int = 8) -> List[Dict]:
        """Analyze transcript segments to identify speakers with confidence scores, in chunks."""

        all_parsed_results = {"speaker_analysis": [], "summary": {}}
        enriched_segments = []
        total_segments = len(segments)
        num_chunks = (total_segments + chunk_size - 1) // chunk_size

        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min((chunk_idx + 1) * chunk_size, total_segments)
            chunk = segments[chunk_start:chunk_end]

            transcript_text = " ".join([seg["text"] for seg in chunk])

            # Prepare prompt for speaker diarization
            prompt_template = PromptTemplate(
                input_variables=["transcript", "segments"],
                template="""
You are an expert in analyzing interview conversations and identifying speakers.
Analyze the following interview transcript and identify who is speaking in each segment.

CONTEXT: This is a job interview with typically 2 speakers:
- INTERVIEWER: The person asking questions (usually introduces themselves, asks about experience, qualifications, etc.)
- CANDIDATE: The person being interviewed (responds to questions, talks about their background, asks about the job)

TRANSCRIPT:
{transcript}

SEGMENTS TO ANALYZE:
{segments}

For each segment, determine:
1. Who is most likely speaking (INTERVIEWER or CANDIDATE)
2. Confidence score (0.0 to 1.0) based on:
   - Content type (question vs answer)
   - Language patterns (formal vs personal)
   - Context clues (introductions, responses)

Return ONLY a JSON object in this exact format:
{{
    "speaker_analysis": [
        {{
            "segment_index": 0,
            "speaker": "INTERVIEWER",
            "confidence": 0.95,
            "reasoning": "Introduces themselves and asks initial question"
        }},
        {{
            "segment_index": 1,
            "speaker": "CANDIDATE",
            "confidence": 0.90,
            "reasoning": "Responds with personal experience details"
        }}
    ],
    "summary": {{}}
}}
""",
            )

            # Format segments for the prompt
            segments_text = ""
            for i, seg in enumerate(chunk):
                segments_text += f"Segment {chunk_start + i}: [{seg['start']:.1f}s-{seg['end']:.1f}s] \"{seg['text']}\"\n"

            prompt = prompt_template.format(
                transcript=transcript_text, segments=segments_text
            )

            print(f"🤖 Analyzing speakers with LLM for segments {chunk_start + 1}-{chunk_end} of {total_segments}...")
            llm_response = query_ollama(prompt, model="llama3.2")
            print(f"✅ LLM response received for segments {chunk_start + 1}-{chunk_end}.")

            # Parse the LLM response
            parsed_result = self.parser.parse(llm_response)

            if "error" in parsed_result:
                print(f"⚠️  Warning: {parsed_result['error']}")
                raise Exception("LLM diarization failed")

            # Merge speaker info back into chunk
            chunk_enriched = self._merge_speaker_data(chunk, parsed_result)
            enriched_segments.extend(chunk_enriched)
            all_parsed_results["speaker_analysis"].extend(parsed_result.get("speaker_analysis", []))

        # Save analysis to state
        self.state_manager.set_state("speaker_analysis", all_parsed_results)
        self.state_manager.set_state("diarization_completed", True)

        return enriched_segments

    def _merge_speaker_data(self, segments: List[Dict], analysis: Dict) -> List[Dict]:
        """Merge speaker analysis back into segment data"""
        enriched_segments = []

        speaker_data = {
            item["segment_index"]: item for item in analysis.get("speaker_analysis", [])
        }

        for i, segment in enumerate(segments):
            enriched_segment = segment.copy()

            if i in speaker_data:
                speaker_info = speaker_data[i]
                enriched_segment.update(
                    {
                        "speaker": speaker_info["speaker"],
                        "confidence": speaker_info["confidence"],
                        "reasoning": speaker_info.get("reasoning", ""),
                    }
                )
            else:
                # Fallback if LLM didn't analyze this segment
                enriched_segment.update(
                    {
                        "speaker": "UNKNOWN",
                        "confidence": 0.5,
                        "reasoning": "Not analyzed by LLM",
                    }
                )

            enriched_segments.append(enriched_segment)

        return enriched_segments
