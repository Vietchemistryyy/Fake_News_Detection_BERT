"""Command-line utility to build a knowledge base from CSV files.

Usage (example):
    python -m src.rag.build_kb --csv data/processed/train.csv --out models/kb --col cleaned_content

This script uses Sentence-Transformers for embeddings and FAISS for indexing
when available. If FAISS isn't installed it will use sklearn's NearestNeighbors.

Install extras:
    pip install sentence-transformers
    pip install faiss-cpu   # optional, recommended
"""
from __future__ import annotations

import argparse
import logging
import os
from typing import Optional

import pandas as pd

from src.rag.knowledge_base import KnowledgeBaseFAISS


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def detect_text_column(df: pd.DataFrame, prefer: Optional[str] = None) -> str:
    if prefer and prefer in df.columns:
        return prefer
    for c in ["cleaned_content", "content", "text", "article"]:
        if c in df.columns:
            return c
    # fallback: first string-like column
    for c in df.columns:
        if df[c].dtype == object:
            return c
    raise ValueError("No text column found in CSV")


def build_kb_from_csv(csv_path: str, out_dir: str, col: Optional[str] = None, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
    os.makedirs(out_dir, exist_ok=True)

    # stream CSV in chunks and incrementally add to KB to avoid high memory use
    reader = pd.read_csv(csv_path, chunksize=2000, iterator=True)

    # get first chunk to detect column
    try:
        first = next(reader)
    except StopIteration:
        raise ValueError(f"CSV file {csv_path} is empty")

    text_col = detect_text_column(first, prefer=col)
    logger.info("Using text column: %s", text_col)

    kb = KnowledgeBaseFAISS(embedding_model_name=embedding_model, device=device)

    # add first chunk
    texts = first[text_col].astype(str).tolist()
    ids = first.index.astype(int).tolist()
    logger.info("Processing chunk: %d documents", len(texts))
    kb.add_texts(texts, ids=ids)

    # process remaining chunks
    total = len(texts)
    for chunk in reader:
        texts = chunk[text_col].astype(str).tolist()
        ids = chunk.index.astype(int).tolist()
        kb.add_texts(texts, ids=ids)
        total += len(texts)
        logger.info("Processed total %d documents...", total)

    kb.save(out_dir)
    logger.info("Knowledge base saved to %s (total %d docs)", out_dir, total)


def main():
    parser = argparse.ArgumentParser(description="Build FAISS knowledge base from CSV")
    parser.add_argument("--csv", required=True, help="Input CSV file (processed) containing text column")
    parser.add_argument("--out", required=True, help="Output directory to write KB files")
    parser.add_argument("--col", default=None, help="Text column name (if not provided will detect) ")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")

    args = parser.parse_args()
    build_kb_from_csv(args.csv, args.out, col=args.col, embedding_model=args.embedding_model)


if __name__ == "__main__":
    main()
