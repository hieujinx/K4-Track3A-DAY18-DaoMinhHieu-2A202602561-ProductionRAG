from __future__ import annotations

"""Module 1: Advanced Chunking Strategies."""

import glob
import os
import re
import sys
from dataclasses import dataclass, field
from functools import lru_cache

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (  # noqa: E402
    DATA_DIR,
    HIERARCHICAL_CHILD_SIZE,
    HIERARCHICAL_PARENT_SIZE,
    SEMANTIC_THRESHOLD,
)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract the text layer from a PDF."""
    from pypdf import PdfReader

    return "\n\n".join(page.extract_text() or "" for page in PdfReader(path).pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load Markdown files and PDFs that contain a text layer."""
    docs = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(path, encoding="utf-8") as file:
            docs.append({"text": file.read(), "metadata": {"source": os.path.basename(path)}})
    for path in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(path)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(path)}})
        else:
            print(f"  ⚠️  Bỏ qua {os.path.basename(path)}: PDF scan ảnh, không có text layer (cần OCR).")
    return docs


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """Baseline paragraph chunking retained for comparison."""
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[Chunk] = []
    current = ""
    for paragraph in paragraphs:
        if len(current) + len(paragraph) > chunk_size and current:
            chunks.append(Chunk(current.strip(), {**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += paragraph + "\n\n"
    if current.strip():
        chunks.append(Chunk(current.strip(), {**metadata, "chunk_index": len(chunks)}))
    return chunks


@lru_cache(maxsize=1)
def _semantic_model():
    """Load the embedding model once per process."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2")


def chunk_semantic(
    text: str,
    threshold: float = SEMANTIC_THRESHOLD,
    metadata: dict | None = None,
) -> list[Chunk]:
    """Group adjacent sentences while their cosine similarity stays high."""
    from numpy import dot
    from numpy.linalg import norm

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n\n", text) if s.strip()]
    if not sentences:
        return []
    embeddings = _semantic_model().encode(sentences)
    groups = [[sentences[0]]]
    for index in range(1, len(sentences)):
        left, right = embeddings[index - 1], embeddings[index]
        similarity = float(dot(left, right) / (norm(left) * norm(right) + 1e-9))
        if similarity < threshold:
            groups.append([sentences[index]])
        else:
            groups[-1].append(sentences[index])
    base = dict(metadata or {})
    return [
        Chunk(" ".join(group), {**base, "strategy": "semantic", "chunk_index": index})
        for index, group in enumerate(groups)
    ]


def _split_with_limit(text: str, limit: int) -> list[str]:
    """Pack paragraphs into chunks no longer than ``limit`` characters."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    units: list[str] = []
    for paragraph in paragraphs:
        remainder = paragraph
        while len(remainder) > limit:
            split_at = remainder.rfind(" ", 0, limit + 1)
            if split_at <= 0:
                split_at = limit
            units.append(remainder[:split_at].strip())
            remainder = remainder[split_at:].strip()
        if remainder:
            units.append(remainder)

    chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current}\n\n{unit}" if current else unit
        if current and len(candidate) > limit:
            chunks.append(current)
            current = unit
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def chunk_hierarchical(
    text: str,
    parent_size: int = HIERARCHICAL_PARENT_SIZE,
    child_size: int = HIERARCHICAL_CHILD_SIZE,
    metadata: dict | None = None,
) -> tuple[list[Chunk], list[Chunk]]:
    """Create parent chunks and smaller children linked by ``parent_id``."""
    if parent_size <= 0 or child_size <= 0:
        raise ValueError("parent_size and child_size must be positive")
    base = dict(metadata or {})
    parents: list[Chunk] = []
    children: list[Chunk] = []
    for parent_index, parent_text in enumerate(_split_with_limit(text, parent_size)):
        parent_id = f"parent_{parent_index}"
        parents.append(Chunk(
            parent_text,
            {**base, "chunk_type": "parent", "parent_id": parent_id, "chunk_index": parent_index},
            parent_id,
        ))
        for child_text in _split_with_limit(parent_text, child_size):
            children.append(Chunk(
                child_text,
                {**base, "chunk_type": "child", "parent_id": parent_id, "chunk_index": len(children)},
                parent_id,
            ))
    return parents, children


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """Split Markdown at level 1–3 headings while preserving sections."""
    base = dict(metadata or {})
    parts = re.split(r"(^#{1,3}\s+.+$)", text, flags=re.MULTILINE)
    chunks: list[Chunk] = []
    header = ""
    content_parts: list[str] = []

    def append_section() -> None:
        content = "".join(content_parts).strip()
        section_text = "\n\n".join(item for item in (header.strip(), content) if item)
        if section_text:
            section = re.sub(r"^#{1,3}\s+", "", header.strip()) if header else ""
            chunks.append(Chunk(section_text, {
                **base,
                "section": section,
                "strategy": "structure",
                "chunk_index": len(chunks),
            }))

    for part in parts:
        if re.match(r"^#{1,3}\s+", part):
            append_section()
            header = part.strip()
            content_parts = []
        else:
            content_parts.append(part)
    append_section()
    return chunks


def compare_strategies(documents: list[dict]) -> dict:
    """Run all strategies and compare their chunk lengths."""
    def stats(chunk_list):
        lengths = [len(chunk.text) for chunk in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(document["text"] for document in documents)
    meta = {"source": "all"}
    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)
    results = {
        "basic": stats(basic),
        "semantic": stats(semantic),
        "hierarchical": {**stats(children), "parents": len(parents)},
        "structure": stats(structure),
    }
    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, item in results.items():
        print(f"{name:<15} {item['count']:>7} {item['avg_len']:>5} {item['min_len']:>5} {item['max_len']:>5}")
    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, item in results.items():
        print(f"  {name}: {item}")
