"""
Role Identification Agent - Full LangChain Implementation

Uses LangChain with Google Gemini for Chain of Thought (CoT) and
Chain of Verification (CoV) prompting to identify speaker roles.

LangChain Components Used:
- ChatGoogleGenerativeAI: LLM integration
- PromptTemplate: Prompt management
- LCEL pipelines: Chain orchestration
- StrOutputParser: Output parsing
"""

import re
from typing import Dict, List, Any, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from prompts import (
    ROLE_IDENTIFICATION_COT_PROMPT,
    ROLE_VERIFICATION_COV_PROMPT,
    ROLE_SUMMARY_PROMPT,
)

from agents.base_agent import BaseAgent


class RoleIdentificationAgent(BaseAgent):
    """
    Agent for identifying speaker roles using LangChain.

    Uses LangChain's ChatGoogleGenerativeAI with Chain of Thought and
    Chain of Verification to accurately identify roles.
    """

    def __init__(
        self,
        state_manager,
        gemini_api_key: str,
        model_name: str = "gemini-2.5-flash",
        agent_name: str = "RoleIdentificationAgent"
    ):
        """
        Initialize the Role Identification Agent with LangChain.

        Args:
            state_manager: StateManager instance for shared state
            gemini_api_key: Google Gemini API key
            model_name: Name of the Gemini model to use
            agent_name: Name of this agent
        """
        super().__init__(state_manager, agent_name)
        self.model_name = model_name
        self.gemini_api_key = gemini_api_key

        # Dedicated LangChain clients per reasoning phase so we can tune sampling
        self.cot_llm = self._build_llm(temperature=0.3)
        self.cov_llm = self._build_llm(temperature=0.2)
        self.summary_llm = self._build_llm(temperature=0.5)

        self.log(
            f"Initialized LangChain pipelines with Gemini model: {model_name}",
            "info",
        )

        # Initialize prompt templates and runnable chains
        self._setup_prompts()

    def _build_llm(self, temperature: float) -> ChatGoogleGenerativeAI:
        """Create a configured LangChain LLM client for Gemini."""

        return ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.gemini_api_key,
            temperature=temperature,
            convert_system_message_to_human=True
        )

    def _setup_prompts(self):
        """Set up LangChain PromptTemplates and runnable chains."""

        self.cot_prompt_template = PromptTemplate.from_template(
            ROLE_IDENTIFICATION_COT_PROMPT
        )
        self.cov_prompt_template = PromptTemplate.from_template(
            ROLE_VERIFICATION_COV_PROMPT
        )
        self.summary_prompt_template = PromptTemplate.from_template(
            ROLE_SUMMARY_PROMPT
        )

        # Build reusable runnable chains so each phase reuses configuration
        self.cot_chain = self.cot_prompt_template | self.cot_llm | StrOutputParser()
        self.cov_chain = self.cov_prompt_template | self.cov_llm | StrOutputParser()
        self.summary_chain = (
            self.summary_prompt_template | self.summary_llm | StrOutputParser()
        )

    def _invoke_chain(
        self,
        chain,
        inputs: Dict[str, str],
        phase_name: str,
    ) -> Optional[str]:
        """Invoke a LangChain runnable with consistent logging."""
        try:
            self.log(
                f"Invoking {phase_name} pipeline (input ~{len(str(inputs))} chars)",
                "info",
            )
            response = chain.invoke(inputs)
            self.log(
                f"{phase_name} pipeline completed ({len(response)} characters)",
                "info",
            )
            return response

        except Exception as e:
            self.log(f"LangChain error during {phase_name}: {str(e)}", "error")
            return None

    def _calculate_speaker_statistics(self, diarization_data: List[Dict]) -> Dict[str, Dict]:
        """
        Calculate statistics for each speaker from diarization data.

        Args:
            diarization_data: List of diarization segments

        Returns:
            Dictionary of speaker statistics
        """
        speaker_stats = {}

        for segment in diarization_data:
            speaker = segment.get("speaker", "UNKNOWN")
            start = segment.get("start", 0.0)
            end = segment.get("end", 0.0)
            duration = end - start

            if speaker not in speaker_stats:
                speaker_stats[speaker] = {
                    "num_segments": 0,
                    "total_duration": 0.0,
                    "first_appearance": start,
                    "last_appearance": end,
                    "avg_segment_length": 0.0
                }

            speaker_stats[speaker]["num_segments"] += 1
            speaker_stats[speaker]["total_duration"] += duration
            speaker_stats[speaker]["last_appearance"] = max(
                speaker_stats[speaker]["last_appearance"], end
            )

        # Calculate average segment length
        for speaker, stats in speaker_stats.items():
            if stats["num_segments"] > 0:
                stats["avg_segment_length"] = stats["total_duration"] / stats["num_segments"]

        return speaker_stats

    def _format_transcript(self, transcript_data: List[Dict]) -> str:
        """
        Format transcript data for LLM input.

        Args:
            transcript_data: List of transcript segments with speaker labels

        Returns:
            Formatted transcript string
        """
        formatted_lines = []

        for segment in transcript_data:
            speaker = segment.get("speaker", "UNKNOWN")
            text = segment.get("text", "")
            start = segment.get("start", 0.0)
            end = segment.get("end", 0.0)

            formatted_lines.append(
                f"[{start:.2f}s - {end:.2f}s] {speaker}: {text}"
            )

        return "\n".join(formatted_lines)

    def _format_speaker_stats(self, speaker_stats: Dict[str, Dict]) -> str:
        """
        Format speaker statistics for prompt.

        Args:
            speaker_stats: Speaker statistics dictionary

        Returns:
            Formatted stats string
        """
        stats_lines = []
        for speaker, stats in speaker_stats.items():
            stats_lines.append(
                f"- {speaker}: {stats.get('num_segments', 0)} segments, "
                f"{stats.get('total_duration', 0):.2f}s total speaking time"
            )
        return "\n".join(stats_lines)

    def _parse_role_assignments(self, llm_response: str) -> Dict[str, Dict[str, str]]:
        """
        Parse role assignments from LLM response.

        Args:
            llm_response: Raw LLM response text

        Returns:
            Dictionary of speaker -> role data
        """
        role_assignments = {}

        # Pattern: Extract SPEAKER blocks
        speaker_blocks = re.findall(
            r'SPEAKER:\s*(\w+).*?ASSIGNED ROLE:\s*([^\n]+?)(?:\n|$)',
            llm_response,
            re.DOTALL | re.IGNORECASE
        )

        for speaker_id, role in speaker_blocks:
            speaker_id = speaker_id.strip()
            role = role.strip()

            # Extract confidence if present
            confidence_match = re.search(
                r'CONFIDENCE:\s*(\w+)',
                llm_response[llm_response.find(speaker_id):],
                re.IGNORECASE
            )
            confidence = confidence_match.group(1) if confidence_match else "Medium"

            role_assignments[speaker_id] = {
                "role": role,
                "confidence": confidence,
                "raw_analysis": llm_response
            }

        return role_assignments

    def _identify_roles_cot(
        self,
        transcript_data: List[Dict],
        speaker_stats: Dict[str, Dict]
    ) -> Dict[str, Dict[str, str]]:
        """
        Identify roles using Chain of Thought with LangChain.

        Args:
            transcript_data: Transcript with speaker labels
            speaker_stats: Speaker statistics

        Returns:
            Initial role assignments
        """
        self.log("Phase 1: Chain of Thought role identification (LangChain)", "info")

        formatted_transcript = self._format_transcript(transcript_data)
        formatted_stats = self._format_speaker_stats(speaker_stats)

        # Use LangChain chain
        llm_response = self._invoke_chain(
            self.cot_chain,
            {
                "transcript_data": formatted_transcript,
                "speaker_stats": formatted_stats
            },
            phase_name="Chain of Thought",
        )

        if not llm_response:
            raise Exception("Failed to get response from LLM for role identification")

        # Store raw CoT response
        self.update_state("role_identification_cot_response", llm_response)

        # Parse role assignments
        role_assignments = self._parse_role_assignments(llm_response)

        self.log(f"Identified {len(role_assignments)} speaker roles", "info")
        return role_assignments

    def _verify_roles_cov(
        self,
        role_assignments: Dict[str, Dict[str, str]],
        transcript_data: List[Dict]
    ) -> Dict[str, Dict[str, str]]:
        """
        Verify role assignments using Chain of Verification with LangChain.

        Args:
            role_assignments: Initial role assignments from CoT
            transcript_data: Transcript with speaker labels

        Returns:
            Verified role assignments
        """
        self.log("Phase 2: Chain of Verification (LangChain)", "info")

        formatted_transcript = self._format_transcript(transcript_data)

        # Format role assignments
        assignments_text = "\n".join([
            f"- {speaker}: {data.get('role', 'Unknown')}"
            for speaker, data in role_assignments.items()
        ])

        # Use LangChain chain
        llm_response = self._invoke_chain(
            self.cov_chain,
            {
                "role_assignments": assignments_text,
                "transcript_data": formatted_transcript
            },
            phase_name="Chain of Verification",
        )

        if not llm_response:
            self.log("CoV verification failed, using original assignments", "warning")
            return role_assignments

        # Store raw CoV response
        self.update_state("role_verification_cov_response", llm_response)

        # Parse verification and update assignments
        verified_assignments = self._parse_verification_response(llm_response, role_assignments)

        self.log(f"Verified {len(verified_assignments)} role assignments", "info")
        return verified_assignments

    def _parse_verification_response(
        self,
        llm_response: str,
        original_assignments: Dict[str, Dict[str, str]]
    ) -> Dict[str, Dict[str, str]]:
        """Parse verification response."""
        verified_assignments = original_assignments.copy()

        # Look for RECOMMENDED ROLE
        verification_blocks = re.findall(
            r'SPEAKER:\s*(\w+).*?RECOMMENDED ROLE:\s*([^\n]+?)(?:\n|$)',
            llm_response,
            re.DOTALL | re.IGNORECASE
        )

        for speaker_id, recommended_role in verification_blocks:
            speaker_id = speaker_id.strip()
            recommended_role = recommended_role.strip()

            if speaker_id in verified_assignments:
                if recommended_role != verified_assignments[speaker_id].get("role"):
                    self.log(
                        f"Role updated for {speaker_id}: "
                        f"{verified_assignments[speaker_id].get('role')} -> {recommended_role}",
                        "info"
                    )
                    verified_assignments[speaker_id]["role"] = recommended_role

                # Extract confidence score
                confidence_match = re.search(
                    r'CONFIDENCE SCORE:\s*(\d+)',
                    llm_response[llm_response.find(speaker_id):],
                    re.IGNORECASE
                )
                if confidence_match:
                    confidence_score = int(confidence_match.group(1))
                    if confidence_score >= 80:
                        verified_assignments[speaker_id]["confidence"] = "High"
                    elif confidence_score >= 50:
                        verified_assignments[speaker_id]["confidence"] = "Medium"
                    else:
                        verified_assignments[speaker_id]["confidence"] = "Low"

        return verified_assignments

    def _generate_summary(
        self,
        verified_roles: Dict[str, Dict[str, str]]
    ) -> str:
        """
        Generate final summary using LangChain.

        Args:
            verified_roles: Verified role assignments

        Returns:
            Summary text
        """
        self.log("Phase 3: Generating role identification summary (LangChain)", "info")

        # Format roles
        roles_text = "\n".join([
            f"- {speaker}: {data.get('role', 'Unknown')} "
            f"(Confidence: {data.get('confidence', 'N/A')})"
            for speaker, data in verified_roles.items()
        ])

        # Use LangChain chain
        llm_response = self._invoke_chain(
            self.summary_chain,
            {"verified_roles": roles_text},
            phase_name="Summary",
        )

        if not llm_response:
            # Fallback to manual summary
            summary_lines = ["SPEAKER ROLE IDENTIFICATION SUMMARY", "=" * 40, ""]
            for speaker, data in verified_roles.items():
                summary_lines.append(
                    f"{speaker}: {data.get('role', 'Unknown')} "
                    f"(Confidence: {data.get('confidence', 'N/A')})"
                )
            return "\n".join(summary_lines)

        return llm_response

    def execute(
        self,
        transcript_data: Optional[List[Dict]] = None,
        use_verification: bool = True
    ) -> Dict[str, Any]:
        """
        Execute role identification pipeline with LangChain.

        Args:
            transcript_data: List of transcript segments with speaker labels
            use_verification: Whether to use Chain of Verification

        Returns:
            Dictionary containing results
        """
        # Get transcript data from state if not provided
        if transcript_data is None:
            transcript_data = self.get_state("merged_transcript")

        if not transcript_data:
            raise ValueError("No transcript data provided or found in state")

        self.log(f"Processing transcript with {len(transcript_data)} segments", "info")

        # Calculate speaker statistics
        speaker_stats = self._calculate_speaker_statistics(transcript_data)
        self.update_state("speaker_statistics", speaker_stats)

        # Phase 1: Chain of Thought role identification (LangChain)
        role_assignments = self._identify_roles_cot(transcript_data, speaker_stats)

        # Phase 2: Chain of Verification (LangChain)
        if use_verification:
            role_assignments = self._verify_roles_cov(role_assignments, transcript_data)

        # Phase 3: Generate summary (LangChain)
        summary = self._generate_summary(role_assignments)

        # Store results in state
        self.update_state("role_assignments", role_assignments)
        self.update_state("role_identification_summary", summary)

        result = {
            "success": True,
            "role_assignments": role_assignments,
            "summary": summary,
            "speaker_stats": speaker_stats,
            "total_speakers": len(role_assignments)
        }

        self.log(f"Role identification complete: {len(role_assignments)} speakers identified", "info")

        return result
