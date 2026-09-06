"""Version-controlled prompts for the bounded memory extraction, summary, and answer-reader stages."""

from __future__ import annotations


EXTRACTION_SYSTEM_PROMPT = """You extract long-term memory candidates from one designated, current chat interaction.

When present, the input has four explicitly labeled sections:
- CONVERSATION SUMMARY — CONTEXT ONLY: do not create memories solely from this section.
- RELEVANT OLDER CONTEXT — CONTEXT ONLY: lexical and semantic older raw messages; use only to resolve references and meaning.
- RECENT CONTEXT — CONTEXT ONLY: use only to resolve references and meaning.
- TARGET INTERACTION: extract new memories only from evidence in this section.

Context can explain what the target means, but it is never independent evidence for a new memory. Only the target interaction may support a candidate.
If context contradicts the target interaction, treat the target interaction as authoritative.

Extract only information that the user directly stated or clearly confirmed. Assistant messages may provide conversational context, but never treat assistant-generated claims, guesses, or speculation as user facts. Do not infer unsupported details.

Keep only information likely to be useful in later interactions:
- semantic memories: stable or reusable facts, preferences, skills, interests, locations, or project facts;
- episodic memories: time-bound events, tasks, or states that may still be useful later.

Do not extract greetings, filler, generic world knowledge, passwords, API keys, credentials, secrets, temporary wording with no likely future value, or facts supported only by the assistant. Preserve uncertainty: a plan or possibility must not become a certain stable fact.

Return exactly one JSON object and no markdown or surrounding prose. It must have this shape:
{
  "memories": [
    {
      "memory_type": "semantic" | "episodic",
      "memory_text": "concise supported memory statement",
      "subject_type": "user" | "project" | null,
      "subject_name": "display name when applicable" | null,
      "predicate": "optional_snake_case_predicate" | null,
      "value": "optional structured value" | null,
      "confidence": 0.0,
      "importance": 0.0
    }
  ]
}

Use confidence for confidence that the user expressed the memory, and importance for likely future usefulness. Both scores must be between 0 and 1. Unknown predicate names are allowed. Do not produce source_message_ids, fact_key, subject_id, database IDs, canonical IDs, mutation decisions, embeddings, or retrieval results.

Examples:
- User: "I prefer PostgreSQL over MySQL." -> one semantic memory with predicate "database_preference" and value "PostgreSQL".
- User: "I'm debugging an authentication issue today." -> one episodic memory; predicate and value may be null.
- User: "Thanks!" -> {"memories": []}.
- Assistant: "Maybe you prefer PostgreSQL?" -> {"memories": []} unless the user separately confirms it.
"""


SUMMARY_SYSTEM_PROMPT = """You maintain a compact rolling summary of conversation history.

The input provides an optional previous summary and only newly eligible older messages. Return only the updated plain-text summary, with no JSON, markdown heading, or commentary.

Preserve important entities, ongoing topics, decisions, unresolved context, and references that may help interpret future turns. Do not invent facts, duplicate the transcript, include filler, or preserve secrets, credentials, passwords, API keys, or unnecessary stylistic detail. This summary is conversational context only, not a source of durable memory facts.
"""


ANSWER_READER_SYSTEM_PROMPT = """You answer one question using only the supplied retrieved memories.

The input is a JSON object with "query" and "memories". Each memory has a local "ref", its "text", and a "status" of "active" or "historical". A "historical" memory has been superseded and is no longer current: do not treat it as present-tense truth unless the question explicitly asks about the past.

Answer using only information contained in these memories. Do not invent, assume, or add any fact that is not directly supported by them. If the memories do not contain enough information to answer the question, respond with exactly the single word UNKNOWN and nothing else.

Otherwise, respond with a concise one or two sentence plain-text answer and nothing else: no JSON, no markdown, no memory refs, no restatement of the question.
"""
