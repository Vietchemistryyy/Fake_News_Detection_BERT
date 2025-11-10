"""RAG wrapper for classifier models.

This module implements a simple Retrieval-Augmented-Generation (RAG)-style
wrapper for a discriminative classifier: it retrieves top-k documents for a
query and appends them to the input text before passing the combined text to a
sequence classification model (RoBERTa from this project).

Design choices (simple, safe):
- Retrieval: use provided retriever (KnowledgeBaseFAISS)
- Augmentation: concatenate query + SEP + join(retrieved_docs[:k])
- Classification: use HF model/tokenizer or the project's BertForSequenceClassification wrapper
"""
from __future__ import annotations

from typing import List, Optional, Tuple
import logging

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class RAGClassifier:
    """Simple RAG-style wrapper.

    Args:
        retriever: object with `retrieve(query, top_k)` returning list of (id, score, text)
        model_wrapper: either an object exposing `get_model()` and `get_tokenizer()` (project wrapper)
                       or a tuple (model, tokenizer).
        device: 'cpu' or 'cuda'
    """

    def __init__(self, retriever, model_wrapper, device: str = "cpu", sep_token: Optional[str] = None):
        """Create the RAG wrapper.

        Args:
            retriever: object with `retrieve(query, top_k)` returning list of (id, score, text)
            model_wrapper: either an object exposing `get_model()` and `get_tokenizer()` or (model, tokenizer)
            device: 'cpu' or 'cuda'
            sep_token: optional string to join retrieved docs (defaults to tokenizer.sep_token or '\n')
        """
        self.retriever = retriever
        self.device = torch.device(device)

        # Accept both project wrapper and raw (model, tokenizer)
        if hasattr(model_wrapper, "get_model") and hasattr(model_wrapper, "get_tokenizer"):
            self.model = model_wrapper.get_model()
            self.tokenizer = model_wrapper.get_tokenizer()
        elif isinstance(model_wrapper, tuple) and len(model_wrapper) == 2:
            self.model, self.tokenizer = model_wrapper
        else:
            raise ValueError("model_wrapper must be wrapper with get_model/get_tokenizer or (model, tokenizer)")

    self.model.to(self.device)
    # separator token for concatenation
    self.sep_token = sep_token if sep_token is not None else getattr(self.tokenizer, "sep_token", "\n")
    # sensible max length for inputs (tokenizer may expose model_max_length)
    self.max_length = min(512, int(getattr(self.tokenizer, "model_max_length", 512)))

    def _augment(self, query: str, retrieved: List[Tuple]) -> str:
        # retrieved is list of (id,score,text)
        docs = [t for (_id, _score, t) in retrieved]
        # keep order and join using sep
        if len(docs) == 0:
            return query
        # Join retrieved docs; if tokenizer truncates they will be truncated at tokenization time.
        joined = f" {self.sep_token} ".join(docs)
        return f"{query} {self.sep_token} {joined}"

    @torch.no_grad()
    def predict_proba(self, texts: List[str], top_k: int = 3, batch_size: int = 8) -> np.ndarray:
        """Return class probabilities for a list of texts.

        Args:
            texts: list of input queries
            top_k: number of retrieved documents to append
            batch_size: batch size for tokenization/inference

        Returns:
            numpy array shape (N, num_labels) of probabilities
        """
        outputs = []
        N = len(texts)
        for i in range(0, N, batch_size):
            batch = texts[i : i + batch_size]
            inputs = []
            for q in batch:
                retrieved = self.retriever.retrieve(q, top_k=top_k)
                aug = self._augment(q, retrieved)
                inputs.append(aug)
            enc = self.tokenizer(
                inputs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.max_length,
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            # unify interface: if model is a HF model wrapper with .model attribute
            model_obj = getattr(self, "model", None)
            try:
                logits = None
                if hasattr(model_obj, "forward"):
                    # HF model
                    out = model_obj(**enc)
                    # try common attributes
                    if hasattr(out, "logits"):
                        logits = out.logits
                    else:
                        # some models may return tuple
                        logits = out[0]
                else:
                    raise RuntimeError("Unsupported model object")
                probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
                outputs.append(probs)
            except Exception as e:
                # log input that caused error for debugging
                logger.exception("Inference failed for batch starting at %d: %s", i, e)
                raise

        if len(outputs) == 0:
            return np.zeros((0, getattr(self.model.config, "num_labels", 1)))

        return np.vstack(outputs)

    def predict(self, texts: List[str], top_k: int = 3, batch_size: int = 8) -> np.ndarray:
        probs = self.predict_proba(texts, top_k=top_k, batch_size=batch_size)
        preds = probs.argmax(axis=1)
        return preds
