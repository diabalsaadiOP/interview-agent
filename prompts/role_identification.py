"""
Role Identification Prompt Templates

Contains Chain of Thought (CoT) and Chain of Verification (CoV) prompts
for identifying speaker roles in multi-speaker conversations.
"""


ROLE_IDENTIFICATION_COT_PROMPT = """
You are an expert at analyzing multi-speaker conversations and identifying participant roles.

Your task is to analyze a conversation transcript with speaker diarization data and identify the role of each speaker.

**Possible Roles:**
- Interviewer: Person asking questions, leading the conversation, evaluating the candidate
- Candidate: Person being interviewed, answering questions, presenting their qualifications
- Panel Member: Additional interviewer, may ask follow-up questions or provide input
- Moderator: Person facilitating the discussion, managing time, introducing topics
- Observer: Person present but minimally participating

**Input Data:**
{transcript_data}

**Speaker Statistics:**
{speaker_stats}

**Instructions:**
Use Chain of Thought reasoning to analyze each speaker. For each speaker, think through:

1. **Speaking Pattern Analysis:**
   - How often do they speak?
   - What is the average length of their speaking segments?
   - Do they initiate conversations or respond?

2. **Content Analysis:**
   - What type of content do they contribute?
   - Are they asking questions or answering them?
   - Do they use evaluative language or demonstrative language?

3. **Interaction Pattern:**
   - Who do they primarily interact with?
   - Do they control the flow of conversation?
   - Are they being addressed by others?

4. **Language Markers:**
   - Do they use phrases like "Tell me about...", "Can you explain...", "What is your experience with..."? (Interviewer markers)
   - Do they use phrases like "I have experience in...", "In my previous role...", "I would approach this by..."? (Candidate markers)
   - Do they use phrases like "Let me follow up on that...", "I'd like to add..."? (Panel member markers)

**Output Format:**
For each speaker, provide your analysis in this format:

```
SPEAKER: [Speaker ID]

REASONING:
- Speaking pattern: [Your analysis]
- Content type: [Your analysis]
- Interaction pattern: [Your analysis]
- Language markers: [Key phrases that indicate role]

CONFIDENCE: [High/Medium/Low]

ASSIGNED ROLE: [Role name]
```

Provide your complete analysis now:
"""


ROLE_VERIFICATION_COV_PROMPT = """
You are a verification expert tasked with validating role assignments in a multi-speaker conversation.

**Original Role Assignments:**
{role_assignments}

**Conversation Transcript:**
{transcript_data}

**Task:** Verify the role assignments using Chain of Verification reasoning.

**Verification Steps:**

1. **Consistency Check:**
   - Are the assigned roles consistent with typical conversation patterns?
   - Is there exactly one primary candidate in an interview scenario?
   - Are interviewer roles appropriately assigned?

2. **Evidence Validation:**
   - Does the transcript evidence support each role assignment?
   - Are there any contradictions between assigned role and actual behavior?
   - Are there any speakers whose role might be ambiguous or misidentified?

3. **Alternative Hypothesis:**
   - For each speaker, consider: "Could this speaker have a different role?"
   - What evidence would contradict the current assignment?
   - Are there any edge cases or unusual patterns?

4. **Confidence Assessment:**
   - How confident are you in each role assignment (0-100%)?
   - Which assignments are most certain and which are most uncertain?
   - What additional information would increase confidence?

**Output Format:**

```
VERIFICATION RESULTS:

SPEAKER: [Speaker ID]
ORIGINAL ROLE: [Assigned role]
VERIFICATION STATUS: [CONFIRMED / NEEDS REVIEW / INCORRECT]
CONFIDENCE SCORE: [0-100]%
REASONING: [Your verification analysis]
RECOMMENDED ROLE: [Same as original or corrected role]
SUPPORTING EVIDENCE: [Key evidence from transcript]
CONTRADICTORY EVIDENCE: [Any contradictions found]

[Repeat for each speaker]

OVERALL ASSESSMENT:
- Number of confirmed assignments: [X]
- Number requiring review: [Y]
- Number of corrections: [Z]
- Overall confidence in role identification: [0-100]%

FINAL RECOMMENDATIONS:
[Any suggestions for improving role identification or handling ambiguities]
```

Provide your verification analysis now:
"""


ROLE_SUMMARY_PROMPT = """
You are creating a final summary of speaker role identification results.

**Verified Role Assignments:**
{verified_roles}

**Conversation Context:**
{conversation_summary}

**Task:** Create a concise, professional summary of the identified roles.

**Output Format:**

```
SPEAKER ROLE IDENTIFICATION SUMMARY
===================================

Total Speakers: [N]

ROLE ASSIGNMENTS:

1. Speaker [ID] - [ROLE]
   Confidence: [High/Medium/Low]
   Key Characteristics: [Brief description]

2. Speaker [ID] - [ROLE]
   Confidence: [High/Medium/Low]
   Key Characteristics: [Brief description]

[Continue for all speakers]

CONVERSATION TYPE: [Interview / Panel Interview / Meeting / Discussion / Other]

QUALITY ASSESSMENT:
- Role identification confidence: [Overall confidence level]
- Ambiguities detected: [Yes/No - describe if yes]
- Recommendations: [Any recommendations for interpretation]
```

Provide the summary now:
"""


SIMPLE_ROLE_IDENTIFICATION_PROMPT = """
Analyze this conversation transcript and identify the role of each speaker.

**Transcript with Speakers:**
{transcript}

**Possible Roles:**
- Interviewer: Asking questions, evaluating
- Candidate: Answering questions, being evaluated
- Panel Member: Additional interviewer
- Moderator: Facilitating discussion
- Observer: Minimal participation

For each speaker (SPEAKER_00, SPEAKER_01, etc.), determine their role based on:
1. Question vs. answer patterns
2. Language used (evaluative vs. demonstrative)
3. Conversation control and initiation
4. Speaking time and frequency

**Return a JSON object with this structure:**
```json
{
  "SPEAKER_00": {
    "role": "Role name",
    "confidence": "High/Medium/Low",
    "reasoning": "Brief explanation"
  },
  "SPEAKER_01": {
    "role": "Role name",
    "confidence": "High/Medium/Low",
    "reasoning": "Brief explanation"
  }
}
```

Analyze and respond with the JSON only:
"""


def get_role_identification_prompt(transcript_data: str, speaker_stats: dict) -> str:
    """
    Generate role identification prompt with Chain of Thought reasoning.

    Args:
        transcript_data: Full transcript with speaker labels
        speaker_stats: Statistics about each speaker (speaking time, frequency, etc.)

    Returns:
        Formatted prompt string
    """
    stats_text = "\n".join([
        f"- {speaker}: {stats.get('num_segments', 0)} segments, "
        f"{stats.get('total_duration', 0):.2f}s total speaking time"
        for speaker, stats in speaker_stats.items()
    ])

    return ROLE_IDENTIFICATION_COT_PROMPT.format(
        transcript_data=transcript_data,
        speaker_stats=stats_text
    )


def get_verification_prompt(role_assignments: dict, transcript_data: str) -> str:
    """
    Generate verification prompt with Chain of Verification reasoning.

    Args:
        role_assignments: Dictionary of speaker -> role assignments
        transcript_data: Full transcript with speaker labels

    Returns:
        Formatted prompt string
    """
    assignments_text = "\n".join([
        f"- {speaker}: {role}"
        for speaker, role in role_assignments.items()
    ])

    return ROLE_VERIFICATION_COV_PROMPT.format(
        role_assignments=assignments_text,
        transcript_data=transcript_data
    )


def get_summary_prompt(verified_roles: dict, conversation_summary: str = "") -> str:
    """
    Generate summary prompt for final role identification results.

    Args:
        verified_roles: Dictionary of verified speaker roles with confidence
        conversation_summary: Optional summary of conversation content

    Returns:
        Formatted prompt string
    """
    roles_text = "\n".join([
        f"- {speaker}: {data.get('role', 'Unknown')} "
        f"(Confidence: {data.get('confidence', 'N/A')})"
        for speaker, data in verified_roles.items()
    ])

    return ROLE_SUMMARY_PROMPT.format(
        verified_roles=roles_text,
        conversation_summary=conversation_summary or "Not provided"
    )


def get_simple_prompt(transcript: str) -> str:
    """
    Generate simple role identification prompt without CoT/CoV.

    Args:
        transcript: Transcript with speaker labels

    Returns:
        Formatted prompt string
    """
    return SIMPLE_ROLE_IDENTIFICATION_PROMPT.format(transcript=transcript)
