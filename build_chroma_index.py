
import json
import logging
import os

import chromadb
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("build_chroma_index")

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS_PATH = os.path.join(HERE, "legal_corpus.json")
CHROMA_PATH = os.path.join(HERE, "chroma_db")
COLLECTION_NAME = "legal_corpus"

ALL_DOC_TYPES = ["court_summons", "fir", "rent_agreement", "property_document", "legal_notice"]


def build_metadata(entry: dict) -> dict:
    """One boolean flag per possible doc_type, plus display fields."""
    meta = {
        "source": entry.get("source", ""),
        "section": entry.get("section", ""),
        "doc_types_display": ", ".join(entry.get("doc_types", [])),
    }
    for dt in ALL_DOC_TYPES:
        meta[f"doc_type_{dt}"] = dt in entry.get("doc_types", [])
    return meta


def entry_display_text(entry: dict) -> str:
    """Text stored as the document content, i.e. what gets returned to the LLM prompt."""
    return f"[{entry.get('source', '')} {entry.get('section', '')}] {entry.get('plain_language', entry.get('text', ''))}"


def entry_embedding_text(entry: dict) -> str:
    """Text used to compute the embedding — the fuller legal text tends to match queries better than the short gloss alone."""
    return f"{entry.get('source', '')} {entry.get('section', '')}: {entry.get('text', '')}"


def main():
    if not os.path.exists(CORPUS_PATH):
        raise FileNotFoundError(f"legal_corpus.json not found at {CORPUS_PATH}")

    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    logger.info("Loaded %d entries from legal_corpus.json", len(corpus))

    model = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("Embedding model loaded")

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    ids = [e["id"] for e in corpus]
    documents = [entry_display_text(e) for e in corpus]
    embed_texts = [entry_embedding_text(e) for e in corpus]
    metadatas = [build_metadata(e) for e in corpus]

    logger.info("Encoding %d entries...", len(corpus))
    embeddings = model.encode(embed_texts, normalize_embeddings=True).tolist()

    # upsert (not add) so re-running this script after editing the corpus
    # is safe and idempotent, rather than erroring on duplicate IDs.
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    logger.info(
        "Indexed %d entries into ChromaDB collection '%s' at %s",
        collection.count(), COLLECTION_NAME, CHROMA_PATH,
    )


if __name__ == "__main__":
    main()