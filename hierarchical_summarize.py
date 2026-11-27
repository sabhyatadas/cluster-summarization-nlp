"""
SPECTRA-Sum: Hierarchical Multi-Scale Document Synthesis Engine
Integrates Spectral Graph Laplacian, Submodular Optimization, Hardware Telemetry,
and Quantized Transformer Generation.
"""

import os
import sys
import math
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

# Ensure root directory is on sys.path
root_dir = os.path.abspath(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import nltk
# Add workspace nltk_data if available
ws_nltk = os.path.abspath(os.path.join(root_dir, '..', '..', '.venv', 'nltk_data'))
if os.path.exists(ws_nltk) and ws_nltk not in nltk.data.path:
    nltk.data.path.insert(0, ws_nltk)

from nltk.tokenize import sent_tokenize, word_tokenize
from utils.sentence_embedding import make_embeddings, mean_pooling
from utils.spectral_graph import SpectralGraphEngine
from utils.submodular import SubmodularSelector
from utils.hardware_profiler import HardwareProfiler, HardwareMetrics
from models.quantized_engine import QuantizedSummarizer
from eval.metrics import SummarizationMetrics


class SpectraSumPipeline:
    """
    End-to-End Hierarchical Summarizer with Spectral Graph Partitioning,
    Submodular Diversity Maximization, and Hardware Acceleration.
    """

    def __init__(
        self,
        model_name: str = "distill-bart",
        precision: str = "fp32",
        submodular_redundancy_weight: float = 0.35,
        similarity_threshold: float = 0.15,
    ):
        self.model_name = model_name
        self.precision = precision
        self.spectral_engine = SpectralGraphEngine(similarity_threshold=similarity_threshold)
        self.submodular_selector = SubmodularSelector(redundancy_weight=submodular_redundancy_weight)
        self.summarizer = QuantizedSummarizer(model_name, precision=precision)

    def process_document(
        self,
        text: str,
        max_summary_length: int = 250,
        min_summary_length: int = 50,
        max_sentences_per_cluster: int = 4,
    ) -> Dict[str, Any]:
        """
        Executes the hierarchical summarization pipeline with full hardware telemetry.
        
        Returns:
            Dictionary containing:
            - final_summary
            - level_1_extractive_clusters (Dict of submodular key sentences)
            - level_2_cluster_abstracts (List of cluster summaries)
            - spectral_graph_data (Coordinates, affinity, centrality, labels)
            - hardware_metrics (Dict of latency, peak RAM, throughput, etc.)
            - evaluation_metrics (Faithfulness, compression ratio)
        """
        text = text.strip()
        if not text:
            return {
                "final_summary": "",
                "level_1_extractive_clusters": {},
                "level_2_cluster_abstracts": [],
                "spectral_graph_data": {},
                "hardware_metrics": {},
                "evaluation_metrics": {},
            }

        with HardwareProfiler(
            precision_mode=self.precision,
            model_param_count=self.summarizer.get_parameter_count()
        ) as profiler:

            # 1. Sentence Segmentation
            sentences = sent_tokenize(text)
            words = word_tokenize(text)
            num_words = len(words)
            num_sents = len(sentences)

            # Get model sequence limit (default 1024)
            raw_limit = getattr(self.summarizer.tokenizer, "model_max_length", 1024)
            token_limit = min(raw_limit or 1024, 1024)

            # Case A: Short text <= token limit
            if num_words <= token_limit and num_sents <= 3:
                summary, tokens_out = self.summarizer.summarize(
                    text,
                    max_length=max_summary_length,
                    min_length=min_summary_length
                )
                profiler.record_tokens(tokens_out)

                eval_res = SummarizationMetrics.compute_faithfulness_and_alignment(
                    summary=summary,
                    source_document=text,
                    embedder_fn=lambda s: make_embeddings(s, mean_pooling)
                )

                return {
                    "final_summary": summary,
                    "level_1_extractive_clusters": {0: sentences},
                    "level_2_cluster_abstracts": [summary],
                    "spectral_graph_data": {
                        "labels": np.zeros(num_sents, dtype=int),
                        "spectral_coords": np.zeros((num_sents, 3)),
                        "centrality": np.ones(num_sents),
                        "num_clusters": 1,
                    },
                    "hardware_metrics": profiler.get_metrics().to_dict(),
                    "evaluation_metrics": eval_res,
                }

            # Case B: Multi-cluster Hierarchical Partitioning
            # 2. Dense Semantic Vector Embeddings
            embeddings = make_embeddings(sentences, mean_pooling)
            if hasattr(embeddings, "cpu"):
                embeddings = embeddings.cpu().numpy()
            else:
                embeddings = np.array(embeddings)

            # 3. Spectral Graph Partitioning
            n_clusters = max(2, min(math.ceil(num_words / (token_limit * 0.7)), num_sents))
            graph_data = self.spectral_engine.partition(embeddings, n_clusters=n_clusters)
            labels = graph_data["labels"]
            centrality = graph_data["centrality"]

            # 4. Submodular Extractive Selection Per Cluster (Eliminates Redundancy)
            extractive_clusters = self.submodular_selector.select_per_cluster(
                embeddings=embeddings,
                cluster_labels=labels,
                sentences=sentences,
                max_sentences_per_cluster=max_sentences_per_cluster,
                centrality_scores=centrality,
            )

            # 5. Hierarchical Multi-Scale Summarization
            cluster_abstracts: List[str] = []
            total_tokens_produced = 0

            for lbl, cluster_sents in extractive_clusters.items():
                cluster_text = " ".join(cluster_sents)
                if not cluster_text.strip():
                    continue

                c_summary, c_toks = self.summarizer.summarize(
                    cluster_text,
                    max_length=max(40, min(150, max_summary_length)),
                    min_length=min(20, min_summary_length),
                )
                cluster_abstracts.append(c_summary)
                total_tokens_produced += c_toks

            # 6. Meta-Synthesis (Consolidate Level-2 abstracts into Final Executive Summary)
            combined_abstracts = " ".join(cluster_abstracts)
            final_summary, final_toks = self.summarizer.summarize(
                combined_abstracts,
                max_length=max_summary_length,
                min_length=min_summary_length,
            )
            total_tokens_produced += final_toks
            profiler.record_tokens(total_tokens_produced)

        # 7. Faithfulness & Alignment Evaluation
        eval_metrics = SummarizationMetrics.compute_faithfulness_and_alignment(
            summary=final_summary,
            source_document=text,
            embedder_fn=lambda s: make_embeddings(s, mean_pooling)
        )

        return {
            "final_summary": final_summary,
            "level_1_extractive_clusters": extractive_clusters,
            "level_2_cluster_abstracts": cluster_abstracts,
            "spectral_graph_data": graph_data,
            "hardware_metrics": profiler.get_metrics().to_dict(),
            "evaluation_metrics": eval_metrics,
        }
