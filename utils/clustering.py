import torch
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt


class ClusterEmbeddings():
	def __init__(
		self,
		cluster_estimate,
		cluster_fn,
		embeddings,
		sentences,
		words
	):
		self.embeddings = np.array(embeddings)
		self.sentences = np.array(sentences)
		self.words = np.array(words)
		
		n_samples = len(self.embeddings)
		self.num_clusters = max(1, min(int(cluster_estimate), n_samples))
		self.cluster_estimate = self.num_clusters
		self.cluster_fn = cluster_fn

		if n_samples <= 1 or self.num_clusters <= 1:
			self.labels = np.zeros(n_samples, dtype=int)
			self.clustering_algo = None
		elif self.cluster_fn == "agglo":
			self.clustering_algo = AgglomerativeClustering(n_clusters=self.num_clusters)
			self.cluster = self.clustering_algo.fit(self.embeddings)
			self.labels = self.cluster.labels_
		elif self.cluster_fn == "kmeans":
			self.clustering_algo = KMeans(n_clusters=self.num_clusters, n_init='auto', random_state=42)
			self.cluster = self.clustering_algo.fit(self.embeddings)
			self.labels = self.cluster.labels_
		else:
			self.clustering_algo = AgglomerativeClustering(n_clusters=self.num_clusters)
			self.cluster = self.clustering_algo.fit(self.embeddings)
			self.labels = self.cluster.labels_

	def get_sentence_clusters(self):
		sent_clusters = []
		for lbl in range(self.num_clusters):
			single_cluster = self.sentences[self.labels == lbl]
			chunk = " ".join([sent.strip() for sent in single_cluster if sent.strip()])
			if chunk:
				sent_clusters.append(chunk)
		if not sent_clusters and len(self.sentences) > 0:
			sent_clusters = [" ".join(self.sentences)]
		return np.array(sent_clusters)

	def make_plot(self):
		n_samples = len(self.embeddings)
		if n_samples < 2:
			print("Not enough samples to plot TSNE (needs at least 2).")
			return

		perplexity = min(30, max(1, n_samples - 1))
		projector = TSNE(
			n_components=2,
			perplexity=perplexity,
			learning_rate="auto",
			init="random",
			random_state=42
		)
		proj_embeddings = np.array(
			projector.fit_transform(self.embeddings)
		)

		for lbl in range(self.num_clusters):
			xs = proj_embeddings[self.labels == lbl]
			if len(xs) > 0:
				plt.scatter(xs[:, 0], xs[:, 1], label=f"Cluster {lbl}")

		plt.legend()
		plt.xlabel("x1")
		plt.ylabel("x2")
		plt.show()

if __name__ == '__main__':
	print(f"PyTorch version: {torch.__version__}")
	# Quick test
	dummy_emb = np.random.randn(4, 16)
	dummy_sents = ["Sentence 1.", "Sentence 2.", "Sentence 3.", "Sentence 4."]
	dummy_words = ["Sentence", "1", "Sentence", "2"]
	clemb = ClusterEmbeddings(2, "agglo", dummy_emb, dummy_sents, dummy_words)
	print("Sentence clusters:", clemb.get_sentence_clusters())