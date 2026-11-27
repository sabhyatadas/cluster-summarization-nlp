"""
SPECTRA-Sum: Comprehensive Automated Test Suite
Validates mathematical engines, hardware profilers, and end-to-end pipelines.
"""

import os
import sys
import unittest
import numpy as np

# Ensure root directory is on sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from utils.spectral_graph import SpectralGraphEngine
from utils.submodular import SubmodularSelector
from utils.hardware_profiler import HardwareProfiler
from models.quantized_engine import QuantizedSummarizer
from eval.metrics import SummarizationMetrics
from hierarchical_summarize import SpectraSumPipeline


class TestSpectralGraphEngine(unittest.TestCase):
    def setUp(self):
        self.engine = SpectralGraphEngine(similarity_threshold=0.1, pos_weight=0.2)
        np.random.seed(42)
        self.dummy_embeddings = np.random.randn(8, 16)

    def test_affinity_and_laplacian(self):
        affinity = self.engine.build_affinity_matrix(self.dummy_embeddings)
        self.assertEqual(affinity.shape, (8, 8))
        self.assertTrue(np.all(affinity >= 0.0))
        # Zero diagonal (no self-loops)
        self.assertTrue(np.allclose(np.diag(affinity), 0.0))

        laplacian, deg = self.engine.compute_laplacian(affinity)
        self.assertEqual(laplacian.shape, (8, 8))
        self.assertEqual(len(deg), 8)

    def test_partitioning(self):
        res = self.engine.partition(self.dummy_embeddings, n_clusters=3)
        self.assertIn("labels", res)
        self.assertIn("spectral_coords", res)
        self.assertIn("centrality", res)
        self.assertEqual(len(res["labels"]), 8)
        self.assertEqual(res["spectral_coords"].shape, (8, 3))

    def test_edge_cases(self):
        single_emb = np.random.randn(1, 16)
        res_single = self.engine.partition(single_emb, n_clusters=2)
        self.assertEqual(len(res_single["labels"]), 1)


class TestSubmodularSelector(unittest.TestCase):
    def setUp(self):
        self.selector = SubmodularSelector(redundancy_weight=0.3)
        np.random.seed(42)
        self.dummy_embeddings = np.random.randn(6, 16)

    def test_greedy_selection(self):
        budget = 3
        selected = self.selector.select(self.dummy_embeddings, budget=budget)
        self.assertEqual(len(selected), budget)
        self.assertEqual(len(set(selected)), budget)  # Distinct elements

    def test_budget_exceeds_n(self):
        selected = self.selector.select(self.dummy_embeddings, budget=10)
        self.assertEqual(len(selected), 6)

    def test_zero_budget(self):
        selected = self.selector.select(self.dummy_embeddings, budget=0)
        self.assertEqual(len(selected), 0)


class TestHardwareProfiler(unittest.TestCase):
    def test_telemetry_capture(self):
        with HardwareProfiler(precision_mode="int8") as prof:
            dummy_work = [i ** 2 for i in range(100_000)]
            prof.record_tokens(50)

        metrics = prof.get_metrics()
        d = metrics.to_dict()
        self.assertGreater(d["wall_time_sec"], 0.0)
        self.assertEqual(d["tokens_generated"], 50)
        self.assertGreater(d["throughput_tokens_per_sec"], 0.0)
        self.assertEqual(d["precision_mode"], "int8")


class TestSummarizationMetrics(unittest.TestCase):
    def test_rouge_and_faithfulness(self):
        hyp = "Artificial intelligence models summarize large text documents."
        ref = "AI models can summarize long text documents efficiently."

        scores = SummarizationMetrics.compute_rouge(hyp, ref)
        self.assertIn("rouge-1", scores)
        self.assertIn("rouge-2", scores)
        self.assertIn("rouge-l", scores)
        self.assertGreater(scores["rouge-1"]["f1"], 0.0)

        faith = SummarizationMetrics.compute_faithfulness_and_alignment(hyp, ref)
        self.assertIn("faithfulness_score", faith)
        self.assertIn("compression_ratio", faith)


class TestSpectraSumPipeline(unittest.TestCase):
    def test_pipeline_execution(self):
        text = (
            "Natural language processing uses neural networks to extract document structure. "
            "Transformers suffer from attention bottlenecks on documents exceeding 1024 tokens. "
            "Spectral graph algorithms group similar sentences via harmonic eigenvectors. "
            "Hierarchical multi-scale summarization enables long-form document synthesis."
        )
        pipeline = SpectraSumPipeline(model_name="distill-bart", precision="fp32")
        res = pipeline.process_document(text, max_summary_length=60, min_summary_length=20)

        self.assertIn("final_summary", res)
        self.assertIn("hardware_metrics", res)
        self.assertIn("evaluation_metrics", res)
        self.assertTrue(len(res["final_summary"]) > 0)


if __name__ == "__main__":
    unittest.main()
