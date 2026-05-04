from groq import Groq
from app.core.config import settings

_client = None

def _get_client():
    global _client
    if _client is None:
        _client = Groq(api_key=settings.groq_api_key)
    return _client


def ask_llm(question: str, context_chunks: list[str]) -> str:
    """
    Send question + retrieved context to Groq LLM and return the answer.
    
    The prompt instructs the LLM to:
    - Answer only from the provided context
    - Say "I don't know" if the answer isn't in the context
    - Not hallucinate or make up information
    """
    client = _get_client()

    # Build context string from retrieved chunks
    context = "\n\n---\n\n".join(context_chunks)

    prompt = f"""You are a helpful assistant that answers questions strictly based on the provided document context.

CONTEXT FROM DOCUMENT:
{context}

RULES:
- Answer ONLY using the context above
- If the answer is not in the context, say "I could not find the answer in the provided document."
- Be concise and precise
- Do not make up or infer information not present in the context

QUESTION: {question}

ANSWER:"""

    response = client.chat.completions.create(
        model=settings.groq_model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.1,      # low temperature = factual, deterministic answers
        max_tokens=1024,
    )

    return response.choices[0].message.content.strip()