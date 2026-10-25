"""PDFs -> clean text -> chunks -> embeddings -> MongoDB Atlas nodes."""
import re
from pathlib import Path

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from pymongo.operations import SearchIndexModel
from pypdf import PdfReader

import config
from common import get_client, new_storage_context, setup_models


def clean_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"-\n(\w)", r"\1", text)          # re-join hyphenated line breaks
    text = re.sub(r"[ \t]+", " ", text)              # collapse spaces
    text = re.sub(r"\n\s*\n+", "\n\n", text)         # collapse blank lines
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if not re.fullmatch(r"(page\s*)?\d{1,4}", ln, re.I)]  # page numbers
    return "\n".join(lines).strip()


def load_pdf(path: Path, subject: str) -> list[Document]:
    docs = []
    for page_no, page in enumerate(PdfReader(path).pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text:
            docs.append(Document(
                text=text,
                metadata={"subject": subject, "source": path.name, "page": page_no},
            ))
    return docs


def ensure_vector_index(client) -> None:
    """Create the Atlas vector search index on first run (cosine, 768-d)."""
    coll = client[config.DB_NAME][config.COLLECTION_NAME]
    existing = {ix["name"] for ix in coll.list_search_indexes()}
    if config.VECTOR_INDEX_NAME in existing:
        return
    model = SearchIndexModel(
        name=config.VECTOR_INDEX_NAME,
        type="vectorSearch",
        definition={"fields": [
            {"type": "vector", "path": "embedding",
             "numDimensions": config.EMBED_DIM, "similarity": "cosine"},
            {"type": "filter", "path": "metadata.subject"},
        ]},
    )
    coll.create_search_index(model)
    print(f"Created vector index '{config.VECTOR_INDEX_NAME}' (takes ~1 min to become queryable).")


def ingest(pdf_dir: str, subject: str) -> None:
    setup_models()
    pdfs = sorted(Path(pdf_dir).glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {pdf_dir}")

    docs = [d for p in pdfs for d in load_pdf(p, subject)]
    splitter = SentenceSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    nodes = splitter.get_nodes_from_documents(docs)
    print(f"{len(pdfs)} PDFs -> {len(docs)} pages -> {len(nodes)} chunks")

    client = get_client()
    VectorStoreIndex(nodes, storage_context=new_storage_context(client), show_progress=True)
    ensure_vector_index(client)
    print("Ingestion complete.")
