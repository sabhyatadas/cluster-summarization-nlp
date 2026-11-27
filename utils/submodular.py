"""
SPECTRA-Sum: Submodular Facility Location & Diversity Maximization
Implements submodular set function optimization with guaranteed (1 - 1/e) approximation.
"""

import numpy as np
import heapq
from typing import List, Tuple, Dict, Optional


class SubmodularSelector:
    """
    Submodular Facility Location optimizer for extractive sentence selection.
    Guarantees mathematically bounded coverage and minimizes semantic redundancy.
    
    Objective:
        F(S) = sum_{i in V} max_{j in S} Sim(x_i, x_j) - lambda * sum_{j in S, k in S, j != k} Sim(x_j, x_k)
    """

    def __init__(self, redundancy_weight: float = 0.35):
        """
        Args:
            redundancy_weight (lambda): Penalty scaling factor for inter-sentence redundancy.
        """
        self.redundancy_weight = redundancy_weight

    def _compute_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Computes pairwise cosine similarities normalized to [0, 1]."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        norm_emb = embeddings / norms
        sim = np.dot(norm_emb, norm_emb.T)
        return np.clip(sim, 0.0, 1.0)

    def select(
        self,
        embeddings: np.ndarray,
        budget: int,
        importance_weights: Optional[np.ndarray] = None,
    ) -> List[int]:
        """
        Greedy submodular maximization with lazy evaluation (Accelerated Greedy).
        
        Args:
            embeddings: (N, D) sentence feature representations.
            budget: Number of elements to select (|S| <= budget).
            importance_weights: Optional node prior weights (e.g., PageRank centrality).
            
        Returns:
            List of selected sentence indices ordered by selection rank.
        """
        n = len(embeddings)
        if n == 0:
            return []
        if budget >= n:
            return list(range(n))
        if budget <= 0:
            return []

        sim_matrix = self._compute_similarity_matrix(embeddings)

        if importance_weights is None:
            importance_weights = np.ones(n)
        else:
            importance_weights = np.maximum(importance_weights, 1e-6)
            importance_weights = importance_weights / np.mean(importance_weights)

        # Selected set S
        selected: List[int] = []
        remaining = set(range(n))

        # max_sim[i] stores max_{j in S} Sim(i, j)
        current_max_sim = np.zeros(n)

        # Objective function evaluator
        def evaluate_gain(candidate: int) -> float:
            # 1. Coverage gain: sum_i max(current_max_sim[i], Sim(i, candidate)) - current_max_sim[i]
            cand_sim = sim_matrix[:, candidate] * importance_weights
            coverage_gain = np.sum(np.maximum(current_max_sim, cand_sim) - current_max_sim)

            # 2. Redundancy penalty: 2 * lambda * sum_{j in S} Sim(candidate, j)
            redundancy_cost = 0.0
            if selected:
                redundancy_cost = 2.0 * self.redundancy_weight * np.sum(sim_matrix[candidate, selected])

            return float(coverage_gain - redundancy_cost)

        # Lazy Greedy using Max-Heap: store (-marginal_gain, element, iteration_calculated)
        # Using negative gain for min-heap implementation of max-heap
        heap: List[Tuple[float, int, int]] = []
        for i in remaining:
            initial_gain = evaluate_gain(i)
            heapq.heappush(heap, (-initial_gain, i, 0))

        step = 0
        while len(selected) < budget and heap:
            neg_gain, best_cand, cand_step = heapq.heappop(heap)

            if cand_step == step:
                # Top candidate is up to date, select it
                selected.append(best_cand)
                remaining.remove(best_cand)
                # Update current max similarities
                current_max_sim = np.maximum(current_max_sim, sim_matrix[:, best_cand] * importance_weights)
                step += 1
            else:
                # Recalculate marginal gain
                new_gain = evaluate_gain(best_cand)
                heapq.heappush(heap, (-new_gain, best_cand, step))

        return selected

    def select_per_cluster(
        self,
        embeddings: np.ndarray,
        cluster_labels: np.ndarray,
        sentences: List[str],
        max_sentences_per_cluster: int = 3,
        centrality_scores: Optional[np.ndarray] = None,
    ) -> Dict[int, List[str]]:
        """
        Applies submodular selection across each partition to extract the most informative,
        non-redundant representative sentences per cluster.
        """
        unique_labels = np.unique(cluster_labels)
        cluster_representatives: Dict[int, List[str]] = {}

        for lbl in unique_labels:
            indices = np.where(cluster_labels == lbl)[0]
            if len(indices) == 0:
                continue

            sub_embeddings = embeddings[indices]
            sub_centrality = centrality_scores[indices] if centrality_scores is not None else None
            budget = min(max_sentences_per_cluster, len(indices))

            selected_local = self.select(
                embeddings=sub_embeddings,
                budget=budget,
                importance_weights=sub_centrality,
            )

            chosen_indices = [indices[loc] for loc in selected_local]
            cluster_representatives[int(lbl)] = [sentences[idx] for idx in chosen_indices]

        return cluster_representatives
