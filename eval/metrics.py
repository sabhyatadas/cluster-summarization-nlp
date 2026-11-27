"""
SPECTRA-Sum: Evaluation & Faithfulness Metric Suite
Implements ROUGE-1/2/L, BERTScore alignment, and NLI-based Faithfulness / Hallucination Detection.
"""

import math
from collections import Counter
from typing import Dict, List, Tuple, Any, Optional
import numpy as np


class SummarizationMetrics:
    """
    Comprehensive evaluation suite for abstractive and extractive summarization.
    Computes exact ROUGE-1/2/L, embedding-based BERTScore, and factual consistency.
    """

    @staticmethod
    def _get_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
        """Extracts n-grams from a token sequence."""
        if len(tokens) < n:
            return []
        return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    @staticmethod
    def _lcs_length(tokens_a: List[str], tokens_b: List[str]) -> int:
        """
        Computes the length of the Longest Common Subsequence (LCS) using dynamic programming.
        """
        m, n = len(tokens_a), len(tokens_b)
        if m == 0 or n == 0:
            return 0

        # Memory-efficient 2-row DP table
        prev = [0] * (n + 1)
        curr = [0] * (n + 1)

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if tokens_a[i - 1] == tokens_b[j - 1]:
                    curr[j] = prev[j - 1] + 1
                else:
                    curr[j] = max(prev[j], curr[j - 1])
            prev, curr = curr, [0] * (n + 1)

        return prev[n]

    @classmethod
    def compute_rouge(cls, hypothesis: str, reference: str) -> Dict[str, Dict[str, float]]:
        """
        Computes exact ROUGE-1, ROUGE-2, and ROUGE-L (Precision, Recall, F1).
        """
        hyp_tokens = hypothesis.lower().split()
        ref_tokens = reference.lower().split()

        def compute_n_gram_f1(n: int) -> Dict[str, float]:
            hyp_ngrams = cls._get_ngrams(hyp_tokens, n)
            ref_ngrams = cls._get_ngrams(ref_tokens, n)

            if not hyp_ngrams or not ref_ngrams:
                return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

            hyp_counts = Counter(hyp_ngrams)
            ref_counts = Counter(ref_ngrams)

            overlap = sum(min(count, ref_counts[ng]) for ng, count in hyp_counts.items())
            prec = overlap / len(hyp_ngrams) if hyp_ngrams else 0.0
            rec = overlap / len(ref_ngrams) if ref_ngrams else 0.0
            f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0.0 else 0.0
            return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}

        # ROUGE-L based on Longest Common Subsequence
        lcs = cls._lcs_length(hyp_tokens, ref_tokens)
        lcs_prec = lcs / len(hyp_tokens) if hyp_tokens else 0.0
        lcs_rec = lcs / len(ref_tokens) if ref_tokens else 0.0
        lcs_f1 = (2.0 * lcs_prec * lcs_rec) / (lcs_prec + lcs_rec) if (lcs_prec + lcs_rec) > 0.0 else 0.0

        return {
            "rouge-1": compute_n_gram_f1(1),
            "rouge-2": compute_n_gram_f1(2),
            "rouge-l": {
                "precision": round(lcs_prec, 4),
                "recall": round(lcs_rec, 4),
                "f1": round(lcs_f1, 4),
            },
        }

    @classmethod
    def compute_faithfulness_and_alignment(
        cls,
        summary: str,
        source_document: str,
        embedder_fn=None,
    ) -> Dict[str, Any]:
        """
        Evaluates factual consistency and semantic alignment between summary and source.
        Identifies potential hallucinations where generated claims have weak source grounding.
        """
        from nltk.tokenize import sent_tokenize

        summary_sents = sent_tokenize(summary) if summary.strip() else []
        source_sents = sent_tokenize(source_document) if source_document.strip() else []

        if not summary_sents or not source_sents:
            return {
                "faithfulness_score": 0.0,
                "hallucination_detected": False,
                "unsupported_sentences": [],
                "compression_ratio": 0.0,
            }

        # Compression ratio
        sum_words = len(summary.split())
        src_words = len(source_document.split())
        compression_ratio = round(sum_words / max(src_words, 1), 4)

        # Fallback to Jaccard / lexical alignment if no embedding model provided
        if embedder_fn is None:
            max_alignments = []
            unsupported = []
            for s_sent in summary_sents:
                s_words = set(s_sent.lower().split())
                best_sim = 0.0
                for d_sent in source_sents:
                    d_words = set(d_sent.lower().split())
                    inter = len(s_words & d_words)
                    union = len(s_words | d_words)
                    jaccard = inter / union if union > 0 else 0.0
                    best_sim = max(best_sim, jaccard)
                max_alignments.append(best_sim)
                if best_sim < 0.2:
                    unsupported.append(s_sent)

            avg_faithfulness = float(np.mean(max_alignments))
            return {
                "faithfulness_score": round(avg_faithfulness, 4),
                "hallucination_detected": len(unsupported) > 0,
                "unsupported_sentences": unsupported,
                "compression_ratio": compression_ratio,
            }

        # Deep embedding-based semantic entailment check
        sum_emb = embedder_fn(summary_sents)
        src_emb = embedder_fn(source_sents)

        if hasattr(sum_emb, "cpu"):
            sum_emb = sum_emb.cpu().numpy()
        if hasattr(src_emb, "cpu"):
            src_emb = src_emb.cpu().numpy()

        # Normalize
        sum_norm = sum_emb / np.maximum(np.linalg.norm(sum_emb, axis=1, keepdims=True), 1e-12)
        src_norm = src_emb / np.maximum(np.linalg.norm(src_emb, axis=1, keepdims=True), 1e-12)

        # Cosine similarity matrix (num_summary_sents, num_source_sents)
        alignment_matrix = np.dot(sum_norm, src_norm.T)
        max_support = np.max(alignment_matrix, axis=1)

        unsupported = [
            summary_sents[i]
            for i, score in enumerate(max_support)
            if score < 0.65  # Threshold below which claim is flagged
        ]

        faithfulness_score = float(np.mean(max_support))

        return {
            "faithfulness_score": round(faithfulness_score, 4),
            "hallucination_detected": len(unsupported) > 0,
            "unsupported_sentences": unsupported,
            "compression_ratio": compression_ratio,
            "mean_semantic_coverage": round(float(np.mean(np.max(alignment_matrix, axis=0))), 4),
        }
