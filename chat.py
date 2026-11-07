from __future__ import annotations

import logging
from typing import Optional

from analyser import call_llm, check_ollama_available
from rag import format_hits_for_prompt, retrieve_legal_context

logger = logging.getLogger("nyaya_mitra")

MAX_DOCUMENT_CHARS = 6000

SUGGESTED_QUESTIONS = [
    "What happens if I miss the hearing?",
    "What should I do next?",
    "Can I get free legal aid?",
    "Explain the most important date.",
]

CHAT_PROMPT = """You are Nyaya Mitra, a legal information assistant for people in rural India.

Answer the user's question in very simple language. Use ONLY:
1. the uploaded document details below, and
2. the retrieved legal context below.

Rules:
- Reply in {language} when the user's question is in that language.
- If the answer is not supported by the document or the retrieved law, say clearly: "I could not find this in the document or the legal database."
- Cite the law you rely on in square brackets, for example [Indian Penal Code §302].
- Do not invent section numbers, dates, or facts.
- Keep the answer to 2-5 short sentences unless the user asks for more detail.
- Do not give a final legal verdict or guarantee an outcome.
- For urgent matters, tell the user to contact a lawyer or free legal aid.
- If the user asks what to do, give practical next steps.

Uploaded document:
{document_section}

Retrieved legal context:
{legal_context}

Recent conversation:
{history}

User question: {question}

Answer:"""


def _format_history(history) -> str:
    if not history:
        return "No previous questions."

    lines = []
    for turn in history[-4:]:
        if isinstance(turn, dict):
            role = str(turn.get("role", "")).lower()
            content = turn.get("content", "")
            if role and content:
                label = "User" if role == "user" else "Assistant"
                lines.append(f"{label}: {content}")
            continue

        if isinstance(turn, (list, tuple)) and len(turn) >= 2:
            user_text, assistant_text = turn[0], turn[1]
            if user_text:
                lines.append(f"User: {user_text}")
            if assistant_text:
                lines.append(f"Assistant: {assistant_text}")

    return "\n".join(lines) if lines else "No previous questions."


def _format_document(document_text: str, analysis: Optional[dict]) -> str:
    parts = []

    if not document_text and not analysis:
        return (
            "No document has been uploaded. Answer general legal questions "
            "using the retrieved law only."
        )

    if analysis:
        parts.append(
            f"Document type: {analysis.get('document_type') or 'Legal document'}"
        )
        summary = analysis.get("simple_summary", "")
        if summary:
            parts.append(f"Case summary: {summary}")

        dates = analysis.get("critical_dates") or []
        if dates:
            parts.append("Important dates:")
            for item in dates[:10]:
                if isinstance(item, dict):
                    parts.append(
                        f"- {item.get('label', 'Date')}: "
                        f"{item.get('date', '')}. "
                        f"{item.get('importance', '')}".strip()
                    )

        parties = analysis.get("parties_involved") or []
        if parties:
            parts.append("People involved:")
            for person in parties[:10]:
                if isinstance(person, dict):
                    parts.append(
                        f"- {person.get('role', 'Party')}: "
                        f"{person.get('name', 'Not specified')}"
                    )

        ignored = analysis.get("what_happens_if_ignored", "")
        if ignored:
            parts.append(f"If ignored: {ignored}")

        rights = analysis.get("your_rights") or []
        if rights:
            parts.append("Rights: " + "; ".join(str(item) for item in rights[:8]))

        steps = analysis.get("next_steps") or []
        if steps:
            parts.append("Next steps: " + "; ".join(str(item) for item in steps[:8]))

    if document_text:
        excerpt = document_text[:MAX_DOCUMENT_CHARS]
        parts.append(f"Document excerpt:\n{excerpt}")

    return "\n".join(parts)


def answer_question(
    question: str,
    document_text: str = "",
    analysis: Optional[dict] = None,
    doc_type: Optional[str] = None,
    history=None,
    language: str = "English",
) -> dict:
    """
    Answer a follow-up question using the uploaded document plus cited
    legal-context retrieval. Returns:
        {"answer": str, "sources": list[dict]}
    """
    question = (question or "").strip()
    if not question:
        return {"answer": "", "sources": []}

    query_parts = [question]
    if analysis:
        summary = analysis.get("simple_summary", "")
        if summary:
            query_parts.append(summary)
    query = "\n".join(query_parts)

    hits = retrieve_legal_context(query, k=4, doc_type=doc_type)
    if not hits and doc_type:
        # The document-type filter can be too narrow for general questions;
        # retry against the full corpus before giving up.
        hits = retrieve_legal_context(query, k=4, doc_type=None)

    legal_context = format_hits_for_prompt(hits)
    if not legal_context:
        legal_context = (
            "No matching entries were found in the legal database. "
            "Answer only from the uploaded document and say when you are unsure."
        )

    prompt = CHAT_PROMPT.format(
        document_section=_format_document(document_text, analysis),
        legal_context=legal_context,
        history=_format_history(history),
        question=question,
        language=language,
    )

    logger.info("Chat question | doc_type=%s | hits=%d | chars=%d", doc_type, len(hits), len(prompt))
    check_ollama_available()
    answer = call_llm(prompt, json_mode=False, max_tokens=900, temperature=0.2).strip()

    if not answer:
        answer = (
            "I could not find a clear answer in the document or the legal "
            "database. Please ask a lawyer or your nearest legal aid centre."
        )

    return {"answer": answer, "sources": hits}
