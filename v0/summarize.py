import os
import sys

# Ensure v0 directory is on sys.path
v0_dir = os.path.dirname(os.path.abspath(__file__))
if v0_dir not in sys.path:
    sys.path.insert(0, v0_dir)

from utils import make_data, sentence2embedding, ClusterSentenceEmbeddings, bart_summarize
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import warnings
from transformers import BartTokenizer, BartForConditionalGeneration
from sentence_transformers import SentenceTransformer
warnings.filterwarnings("ignore")


def get_summary(articles, threshold=7.0):
	"""
	articles: (list) a list of strings which are the articles to summarize
	threshold: (float) the distance threshold for the Agglomerative Clustering of sentences
	"""
	EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
	SUMMARIZATION_TOKENIZER = BartTokenizer.from_pretrained(
	    'facebook/bart-large-cnn')
	SUMMARIZATION_MODEL = BartForConditionalGeneration.from_pretrained(
	    'facebook/bart-large-cnn')
	sentences, words = make_data(articles)
	embeddings = sentence2embedding(sentences, EMBEDDING_MODEL)

	CUSTOM_CLEMB_KWARGS = {
		'clustering_algorithm': AgglomerativeClustering(
			n_clusters=None,
			distance_threshold=threshold
		),
		'embeddings': embeddings,
		'sentences': sentences,
		'words': words,
		'num_articles': len(articles)
	}

	clemb = ClusterSentenceEmbeddings(**CUSTOM_CLEMB_KWARGS)
	sentence_clusters = clemb.get_sentence_clusters()

	final_summary = ""
	for cluster in sentence_clusters:
		summary = bart_summarize(cluster, SUMMARIZATION_MODEL, SUMMARIZATION_TOKENIZER)
		final_summary += summary + " "

	return final_summary.strip()

if __name__ == '__main__':
	test_articles = [
		"Deep learning models require large datasets and significant computing resources for training. "
		"Inference optimization techniques help reduce latency when serving models in production."
	]
	print("Testing v0 get_summary...")
	res = get_summary(test_articles, threshold=7.0)
	print("Summary result:", res)
