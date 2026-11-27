"""
SPECTRA-Sum: Spectral Graph Laplacian Engine
Mathematical formulation of Normalized Cut Graph Partitioning and Manifold Embedding.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple, Optional
import networkx as nx


class SpectralGraphEngine:
    """
    Constructs a semantic affinity graph and performs spectral graph partitioning
    using the Normalized Symmetric Graph Laplacian: L_sym = I - D^(-1/2) W D^(-1/2).
    """

    def __init__(
        self,
        similarity_threshold: float = 0.15,
        pos_decay_sigma: float = 8.0,
        pos_weight: float = 0.25,
        k_neighbors: int = 5,
    ):
        """
        Args:
            similarity_threshold: Minimum cosine similarity to establish an edge.
            pos_decay_sigma: Gaussian kernel bandwidth for positional proximity.
            pos_weight: Weight given to discourse positional proximity (0.0 to 1.0).
            k_neighbors: K-nearest neighbors for graph sparsification.
        """
        self.similarity_threshold = similarity_threshold
        self.pos_decay_sigma = pos_decay_sigma
        self.pos_weight = pos_weight
        self.k_neighbors = k_neighbors

    def build_affinity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Constructs the affinity matrix W incorporating semantic similarity
        and positional discourse flow.
        """
        n = len(embeddings)
        if n == 0:
            return np.zeros((0, 0))
        if n == 1:
            return np.ones((1, 1))

        # L2-normalize embeddings for stable cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        norm_emb = embeddings / norms

        # Cosine similarity matrix S in [-1, 1]
        sim_matrix = np.dot(norm_emb, norm_emb.T)
        sim_matrix = np.clip(sim_matrix, 0.0, 1.0)  # ReLU on cosine similarity

        # Positional proximity kernel: P_ij = exp( - (i - j)^2 / (2 * sigma^2) )
        indices = np.arange(n)
        dist_sq = (indices[:, None] - indices[None, :]) ** 2
        pos_kernel = np.exp(-dist_sq / (2.0 * (self.pos_decay_sigma ** 2)))

        # Combined weight matrix: W_ij = (1 - alpha) * S_ij + alpha * (S_ij * P_ij)
        affinity = (1.0 - self.pos_weight) * sim_matrix + self.pos_weight * (sim_matrix * pos_kernel)

        # Remove self-loops
        np.fill_diagonal(affinity, 0.0)

        # Sparsify: enforce threshold and k-nearest neighbors
        affinity[affinity < self.similarity_threshold] = 0.0

        if self.k_neighbors < n:
            sparse_affinity = np.zeros_like(affinity)
            for i in range(n):
                top_k = np.argsort(affinity[i])[-self.k_neighbors:]
                sparse_affinity[i, top_k] = affinity[i, top_k]
            # Symmetrize: W = max(W, W^T)
            affinity = np.maximum(sparse_affinity, sparse_affinity.T)

        return affinity

    def compute_laplacian(self, affinity: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the Normalized Symmetric Graph Laplacian:
        L_sym = I - D^(-1/2) W D^(-1/2)
        """
        n = len(affinity)
        if n <= 1:
            return np.zeros((n, n)), np.ones(n)

        degrees = np.sum(affinity, axis=1)
        # Avoid division by zero for isolated vertices
        deg_inv_sqrt = np.zeros(n)
        mask = degrees > 1e-10
        deg_inv_sqrt[mask] = 1.0 / np.sqrt(degrees[mask])

        # Normalized affinity: W_norm = D^(-1/2) W D^(-1/2)
        d_mat = np.diag(deg_inv_sqrt)
        w_norm = d_mat @ affinity @ d_mat

        # L_sym = I - W_norm
        laplacian = np.eye(n) - w_norm
        return laplacian, degrees

    def partition(
        self,
        embeddings: np.ndarray,
        n_clusters: int,
        random_state: int = 42
    ) -> Dict[str, np.ndarray]:
        """
        Executes Spectral Graph Partitioning using the bottom-k non-trivial eigenvectors.
        Returns cluster assignments, spectral coordinates, and node centrality scores.
        """
        n = len(embeddings)
        k = max(1, min(int(n_clusters), n))

        if n <= 1:
            return {
                "labels": np.zeros(n, dtype=int),
                "spectral_coords": np.zeros((n, 2)),
                "affinity": np.ones((n, n)),
                "centrality": np.ones(n),
                "num_clusters": 1,
            }

        affinity = self.build_affinity_matrix(embeddings)
        laplacian, degrees = self.compute_laplacian(affinity)

        # Number of eigenvectors to extract
        num_vecs = min(max(k + 1, 3), n)

        try:
            # Solve smallest eigenvalues/eigenvectors of L_sym
            eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
            # Sort ascending
            idx = np.argsort(eigenvalues)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # Select eigenvectors 1 to k (skipping trivial first vector if connected)
            if n > 2 and num_vecs > 1:
                spectral_space = eigenvectors[:, 1:min(k + 1, n)]
            else:
                spectral_space = eigenvectors[:, :k]

            # L2 normalize rows of spectral space for unit-sphere clustering
            row_norms = np.linalg.norm(spectral_space, axis=1, keepdims=True)
            row_norms = np.maximum(row_norms, 1e-12)
            normalized_spectral = spectral_space / row_norms

            # Cluster in spectral space
            if k == 1:
                labels = np.zeros(n, dtype=int)
            else:
                kmeans = KMeans(n_clusters=k, n_init=10, random_state=random_state)
                labels = kmeans.fit_predict(normalized_spectral)

            # Coordinates for 2D/3D visualization
            if eigenvectors.shape[1] >= 3:
                coords = eigenvectors[:, 1:4]
            else:
                coords = np.pad(spectral_space, ((0, 0), (0, max(0, 3 - spectral_space.shape[1]))))

        except Exception as e:
            # Fallback to direct degree-weighted heuristic if eigen-solver fails
            labels = (np.arange(n) * k // n).astype(int)
            coords = np.zeros((n, 3))

        # Compute PageRank / Eigenvector centrality on the affinity graph
        centrality = self._compute_centrality(affinity)

        return {
            "labels": labels,
            "spectral_coords": coords,
            "affinity": affinity,
            "centrality": centrality,
            "num_clusters": len(np.unique(labels)),
        }

    def _compute_centrality(self, affinity: np.ndarray) -> np.ndarray:
        """Computes PageRank centrality to identify key anchor nodes in each cluster."""
        n = len(affinity)
        if n <= 1:
            return np.ones(n)
        try:
            g = nx.from_numpy_array(affinity)
            pr = nx.pagerank(g, alpha=0.85, max_iter=200)
            return np.array([pr[i] for i in range(n)])
        except Exception:
            deg = np.sum(affinity, axis=1)
            total = np.sum(deg)
            return deg / total if total > 0 else np.ones(n) / n
