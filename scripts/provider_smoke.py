"""Manually verify configured LLM and embedding providers.

Run this script only when intentionally performing a credentialed provider
smoke test. It prints response metadata and content, never configuration
values or API keys.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Running ``python scripts/provider_smoke.py`` puts only ``scripts`` on the
# import path, so add the project root without relying on package installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_config
from src.providers import get_embedding_client, get_llm_client


def main() -> None:
    """Make one small, explicit request to each independently configured provider."""

    settings = get_config()
    if not settings.llm_model:
        raise RuntimeError("LLM_MODEL must be configured before running this smoke test.")

    completion = get_llm_client().chat.completions.create(
        model=settings.llm_model,
        messages=[{"role": "user", "content": "Reply with the word ready."}],
        max_tokens=16,
    )
    print(f"LLM model: {completion.model}")
    print(f"LLM content: {completion.choices[0].message.content}")

    embedding = get_embedding_client().embeddings.create(
        model=settings.embedding_model,
        input="memory layer provider smoke test",
    )
    print(f"Embedding model: {embedding.model}")
    print(f"Embedding dimensions: {len(embedding.data[0].embedding)}")


if __name__ == "__main__":
    main()
