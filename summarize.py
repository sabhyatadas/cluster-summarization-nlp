import os
import sys

# Ensure root directory is on sys.path
root_dir = os.path.abspath(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import nltk
# Add workspace nltk_data if available
workspace_nltk = os.path.abspath(os.path.join(root_dir, '..', '..', '.venv', 'nltk_data'))
if os.path.exists(workspace_nltk) and workspace_nltk not in nltk.data.path:
    nltk.data.path.insert(0, workspace_nltk)
local_nltk = os.path.abspath(os.path.join(root_dir, '.venv', 'nltk_data'))
if os.path.exists(local_nltk) and local_nltk not in nltk.data.path:
    nltk.data.path.insert(0, local_nltk)

from utils.sentence_embedding import make_embeddings, mean_pooling
from utils.clustering import ClusterEmbeddings
from models.summarizers import load_summarizer, summarize_input
from nltk.tokenize import sent_tokenize, word_tokenize
import math
from time import perf_counter
import numpy as np


def get_summary(model_name, article, max_length=250, min_length=50, increment=None):
	if increment is None:
		increment = [0]

	start_time = perf_counter()
	summarization_model, summarization_tokenizer = load_summarizer(model_name)
	raw_limit = getattr(summarization_tokenizer, 'model_max_length', 1024)
	summarizer_token_limit = min(raw_limit or 1024, 1024)
	print("Summarizer token limit:", summarizer_token_limit)

	input_word_toks = word_tokenize(article)
	num_words = len(input_word_toks)

	if num_words <= summarizer_token_limit:
		pred_summary = summarize_input(
			article, 
			summarization_model, 
			summarization_tokenizer,
			max_length=max_length,
			min_length=min_length
		)
		increment[0] = 100
		end_time = perf_counter()
		time_taken = end_time - start_time
		print("Direct summarization time taken:", time_taken)
	
	else:
		input_sent_toks = sent_tokenize(article)
		embeddings = make_embeddings(input_sent_toks, mean_pooling)
		if hasattr(embeddings, 'cpu'):
			embeddings = embeddings.cpu().numpy()
		else:
			embeddings = np.array(embeddings)

		increment[0] = 20

		n_clusters_estimate = max(1, min(math.ceil(num_words / summarizer_token_limit), len(input_sent_toks)))

		clemb = ClusterEmbeddings(
			cluster_estimate=n_clusters_estimate,
			cluster_fn="agglo",
			embeddings=embeddings,
			sentences=np.array(input_sent_toks),
			words=np.array(input_word_toks)
		)

		increment[0] = 50

		sentence_clusters = clemb.get_sentence_clusters()	

		n = len(sentence_clusters)
		summs = ""
		for cluster in sentence_clusters:
			cluster_summary = summarize_input(
				cluster, 
				summarization_model, 
				summarization_tokenizer,
				max_length=250,
				min_length=50,
			)
			if isinstance(cluster_summary, list):
				cluster_summary = cluster_summary[0]
			summs += cluster_summary + " "
			
			if n > 0:
				increment[0] += 40 / n

		pred_summary = summarize_input(
			summs, 
			summarization_model, 
			summarization_tokenizer,
			max_length=max_length,
			min_length=min_length,
		)

		increment[0] = 100

		end_time = perf_counter()
		time_taken = end_time - start_time
	
	return pred_summary, time_taken

def test():
	article = (
		"Artificial intelligence systems have advanced significantly in natural language processing and understanding. "
		"Pretrained transformer architectures demonstrate high quality in generating abstractive summaries of long documents. "
		"However, sequence length restrictions in transformer attention mechanisms create compute bottlenecks for lengthy articles. "
		"By partitioning text into coherent semantic clusters and summarizing each cluster with generative models, "
		"systems can scale beyond input length bounds while preserving factual coverage and context."
	)
	model_name = "BART"
	summ, time_taken = get_summary(model_name, article, 250, 40)
	print("\n--- Summary Result ---")
	print(summ)
	print(f"Time taken: {time_taken:.2f}s")

if __name__ == '__main__':
	test()

