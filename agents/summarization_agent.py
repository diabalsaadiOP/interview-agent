import json
from typing import Dict, List, Any
from utils.ollama import query_ollama


class SummarizationAgent:
    def __init__(self, state):
        self.state = state
        self.key_topics = {
            "experience": {
                "keywords": [
                    "experience",
                    "years",
                    "worked",
                    "position",
                    "job",
                    "role",
                    "background",
                    "expertise",
                    "skilled",
                    "proficient",
                ],
                "description": "Professional background and work history",
            },
            "teamwork": {
                "keywords": [
                    "team",
                    "collaborate",
                    "supervise",
                    "manage",
                    "work with",
                    "colleagues",
                    "staff",
                    "group",
                    "interpersonal",
                ],
                "description": "Collaboration and team management abilities",
            },
            "problem_solving": {
                "keywords": [
                    "solve",
                    "challenge",
                    "difficult",
                    "problem",
                    "issue",
                    "solution",
                    "handle",
                    "resolve",
                    "overcome",
                    "approach",
                ],
                "description": "Problem-solving capabilities and analytical thinking",
            },
            "time_management": {
                "keywords": [
                    "organize",
                    "schedule",
                    "deadline",
                    "priority",
                    "manage",
                    "efficient",
                    "time",
                    "fast",
                    "quick",
                    "productivity",
                ],
                "description": "Time management and organizational skills",
            },
        }

    def analyze_interview(self, segments_json_path: str) -> Dict[str, Any]:
        """
        Analyze interview segments and generate comprehensive summary.

        Args:
            segments_json_path: Path to the JSON file containing interview segments

        Returns:
            Dictionary containing detailed analysis and summaries
        """
        try:
            # Load segments data
            with open(segments_json_path, "r", encoding="utf-8") as f:
                segments = json.load(f)

            # Separate candidate and interviewer segments
            candidate_segments = [
                seg for seg in segments if seg.get("speaker") == "CANDIDATE"
            ]
            interviewer_segments = [
                seg for seg in segments if seg.get("speaker") == "INTERVIEWER"
            ]

            # Analyze key topics in candidate responses
            topic_analysis = self._analyze_topics(candidate_segments)

            # Generate overall summary
            overall_summary = self._generate_overall_summary(
                candidate_segments, interviewer_segments
            )

            # Extract key insights
            key_insights = self._extract_key_insights(candidate_segments)

            # Generate recommendations
            recommendations = self._generate_recommendations(
                topic_analysis, key_insights
            )

            summary_report = {
                "candidate_name": self._extract_candidate_name(segments),
                "interview_duration": (
                    f"{segments[-1]['end']:.1f} seconds" if segments else "Unknown"
                ),
                "total_segments": len(segments),
                "candidate_speaking_time": sum(
                    seg["end"] - seg["start"] for seg in candidate_segments
                ),
                "interviewer_speaking_time": sum(
                    seg["end"] - seg["start"] for seg in interviewer_segments
                ),
                "topic_analysis": topic_analysis,
                "overall_summary": overall_summary,
                "key_insights": key_insights,
                "recommendations": recommendations,
                "candidate_responses": len(candidate_segments),
                "interviewer_questions": len(interviewer_segments),
            }

            return summary_report

        except Exception as e:
            print(f"Error during interview analysis: {str(e)}")
            raise e

    def _analyze_topics(self, candidate_segments: List[Dict]) -> Dict[str, Any]:
        """Analyze candidate responses for key topics."""
        topic_results = {}

        # Combine all candidate text
        candidate_text = " ".join([seg["text"] for seg in candidate_segments])

        for topic, config in self.key_topics.items():
            # Count keyword mentions
            keyword_count = sum(
                1
                for keyword in config["keywords"]
                if keyword.lower() in candidate_text.lower()
            )

            # Find relevant segments
            relevant_segments = []
            for seg in candidate_segments:
                if any(
                    keyword.lower() in seg["text"].lower()
                    for keyword in config["keywords"]
                ):
                    relevant_segments.append(
                        {
                            "text": seg["text"],
                            "timestamp": f"{seg['start']:.1f}s - {seg['end']:.1f}s",
                            "confidence": seg.get("confidence", 0.0),
                        }
                    )

            # Generate AI analysis for this topic
            topic_summary = self._generate_topic_summary(
                topic, relevant_segments, config["description"]
            )

            topic_results[topic] = {
                "keyword_mentions": keyword_count,
                "relevant_segments": relevant_segments,
                "ai_summary": topic_summary,
                "strength_score": min(
                    10, max(1, keyword_count * 2 + len(relevant_segments))
                ),
            }

        return topic_results

    def _generate_topic_summary(
        self, topic: str, segments: List[Dict], description: str
    ) -> str:
        """Generate AI summary for a specific topic."""
        if not segments:
            return f"No clear evidence of {topic} discussed in the interview."

        segments_text = "\n".join([f"- {seg['text']}" for seg in segments])

        prompt = f"""
Analyze the following interview responses related to {topic} ({description}):

{segments_text}

Provide a concise summary (2-3 sentences) of the candidate's strengths and capabilities in {topic} based on these responses. Focus on specific examples and evidence mentioned.
"""

        return query_ollama(prompt)

    def _generate_overall_summary(
        self, candidate_segments: List[Dict], interviewer_segments: List[Dict]
    ) -> str:
        """Generate overall interview summary using AI."""
        candidate_text = "\n".join(
            [f"Candidate: {seg['text']}" for seg in candidate_segments]
        )
        interviewer_text = "\n".join(
            [f"Interviewer: {seg['text']}" for seg in interviewer_segments]
        )

        prompt = f"""
Analyze this job interview conversation and provide a comprehensive summary:

INTERVIEWER QUESTIONS/COMMENTS:
{interviewer_text}

CANDIDATE RESPONSES:
{candidate_text}

Provide a detailed summary (4-5 sentences) covering:
1. The position being discussed
2. Key qualifications mentioned by the candidate
3. Important topics covered in the interview
4. Overall impression of the candidate's suitability
"""

        return query_ollama(prompt)

    def _extract_key_insights(self, candidate_segments: List[Dict]) -> Dict[str, Any]:
        """Extract key insights from candidate responses."""
        all_text = " ".join([seg["text"] for seg in candidate_segments])

        # Extract specific information
        insights = {
            "years_of_experience": self._extract_experience_years(all_text),
            "technical_skills": self._extract_technical_skills(all_text),
            "management_experience": self._extract_management_experience(all_text),
            "salary_expectations": self._extract_salary_info(all_text),
            "availability": self._extract_availability(all_text),
        }

        return insights

    def _extract_experience_years(self, text: str) -> str:
        """Extract years of experience from text."""
        import re

        matches = re.findall(r"(\d+)\s+years?\s+(?:of\s+)?experience", text.lower())
        if matches:
            return f"{matches[0]} years"
        return "Not specified"

    def _extract_technical_skills(self, text: str) -> List[str]:
        """Extract technical skills mentioned."""
        skills = []
        text_lower = text.lower()

        skill_indicators = [
            ("computer programs", "Computer programs"),
            ("type", "Typing"),
            ("100 words per minute", "Fast typing (100+ WPM)"),
            ("organized", "Organization"),
            ("fast learner", "Quick learning ability"),
        ]

        for indicator, skill in skill_indicators:
            if indicator in text_lower:
                skills.append(skill)

        return skills

    def _extract_management_experience(self, text: str) -> str:
        """Extract management/supervisory experience."""
        import re

        text_lower = text.lower()

        if "supervised" in text_lower:
            matches = re.findall(r"supervised\s+(\w+)\s+(\w+)", text_lower)
            if matches:
                return f"Supervised {matches[0][0]} {matches[0][1]}"

        if "supervise" in text_lower:
            return "Has supervisory experience"

        return "Not mentioned"

    def _extract_salary_info(self, text: str) -> str:
        """Extract salary expectations."""
        if "going rate" in text.lower():
            return "Expects market rate"
        return "Not specified"

    def _extract_availability(self, text: str) -> str:
        """Extract availability information."""
        if "beginning of next month" in text.lower():
            return "Available beginning of next month"
        return "Not specified"

    def _extract_candidate_name(self, segments: List[Dict]) -> str:
        """Extract candidate name from segments."""
        for seg in segments:
            if "stevens" in seg["text"].lower():
                return "Mrs. Stevens"
        return "Unknown"

    def _generate_recommendations(
        self, topic_analysis: Dict, key_insights: Dict
    ) -> List[str]:
        """Generate hiring recommendations based on analysis."""
        recommendations = []

        # Analyze strengths
        strong_topics = [
            topic
            for topic, data in topic_analysis.items()
            if data["strength_score"] >= 5
        ]
        weak_topics = [
            topic
            for topic, data in topic_analysis.items()
            if data["strength_score"] < 3
        ]

        if len(strong_topics) >= 3:
            recommendations.append(
                "✅ Strong candidate with well-rounded skills across multiple key areas"
            )

        if "experience" in strong_topics:
            recommendations.append(
                "✅ Demonstrates substantial professional experience"
            )

        if "teamwork" in strong_topics:
            recommendations.append(
                "✅ Shows strong leadership and team management capabilities"
            )

        if key_insights.get("technical_skills"):
            recommendations.append(
                "✅ Possesses relevant technical skills for the position"
            )

        if weak_topics:
            recommendations.append(
                f"⚠️ Consider exploring {', '.join(weak_topics)} in follow-up interviews"
            )

        if key_insights.get("salary_expectations") == "Expects market rate":
            recommendations.append("✅ Reasonable salary expectations")

        return recommendations

    def save_summary_report(
        self, summary_report: Dict[str, Any], output_path: str = None
    ) -> str:
        """Save the summary report to a file."""
        try:
            if output_path is None:
                output_path = "data/interview_summary_report.json"

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(summary_report, f, indent=4, ensure_ascii=False)

            print(f"Summary report saved to: {output_path}")
            return output_path

        except Exception as e:
            print(f"Error saving summary report: {str(e)}")
            raise e

    def generate_human_readable_report(
        self, summary_report: Dict[str, Any], output_path: str = None
    ) -> str:
        """Generate a human-readable text report."""
        try:
            if output_path is None:
                output_path = "data/interview_analysis_report.txt"

            with open(output_path, "w", encoding="utf-8") as f:
                f.write("=" * 60 + "\n")
                f.write("INTERVIEW ANALYSIS REPORT\n")
                f.write("=" * 60 + "\n\n")

                # Basic information
                f.write("CANDIDATE INFORMATION:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Name: {summary_report.get('candidate_name', 'Unknown')}\n")
                f.write(
                    f"Interview Duration: {summary_report.get('interview_duration', 'Unknown')}\n"
                )
                f.write(
                    f"Candidate Speaking Time: {summary_report.get('candidate_speaking_time', 0):.1f} seconds\n"
                )
                f.write(
                    f"Total Responses: {summary_report.get('candidate_responses', 0)}\n\n"
                )

                # Overall summary
                f.write("OVERALL SUMMARY:\n")
                f.write("-" * 30 + "\n")
                f.write(
                    f"{summary_report.get('overall_summary', 'No summary available.')}\n\n"
                )

                # Topic analysis
                f.write("TOPIC ANALYSIS:\n")
                f.write("-" * 30 + "\n")
                for topic, data in summary_report.get("topic_analysis", {}).items():
                    f.write(f"\n{topic.upper()}:\n")
                    f.write(f"Strength Score: {data.get('strength_score', 0)}/10\n")
                    f.write(
                        f"AI Analysis: {data.get('ai_summary', 'No analysis available.')}\n"
                    )

                    if data.get("relevant_segments"):
                        f.write("Key Mentions:\n")
                        for seg in data["relevant_segments"][:3]:  # Show top 3
                            f.write(f"  • [{seg['timestamp']}] {seg['text']}\n")
                    f.write("\n")

                # Key insights
                f.write("KEY INSIGHTS:\n")
                f.write("-" * 30 + "\n")
                insights = summary_report.get("key_insights", {})
                for key, value in insights.items():
                    if value and value != "Not specified" and value != "Not mentioned":
                        f.write(f"{key.replace('_', ' ').title()}: {value}\n")
                f.write("\n")

                # Recommendations
                f.write("RECOMMENDATIONS:\n")
                f.write("-" * 30 + "\n")
                for rec in summary_report.get("recommendations", []):
                    f.write(f"{rec}\n")
                f.write("\n")

                f.write("=" * 60 + "\n")
                f.write("End of Report\n")
                f.write("=" * 60 + "\n")

            print(f"Human-readable report saved to: {output_path}")
            return output_path

        except Exception as e:
            print(f"Error generating human-readable report: {str(e)}")
            raise e

    def run(self, segments_json_path: str) -> Dict[str, Any]:
        """
        Main method to run the complete summarization analysis.

        Args:
            segments_json_path: Path to the interview segments JSON file

        Returns:
            Complete summary report dictionary
        """
        print("📊 Analyzing interview content...")

        # Generate analysis
        summary_report = self.analyze_interview(segments_json_path)

        # Save reports
        json_path = self.save_summary_report(summary_report)
        text_path = self.generate_human_readable_report(summary_report)

        # Update state
        self.state.set_state("summary_report", summary_report)
        self.state.set_state("summary_json_file", json_path)
        self.state.set_state("summary_text_file", text_path)

        print("✅ Interview analysis completed!")
        return summary_report
