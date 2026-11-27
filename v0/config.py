import os
import sys

# Ensure v0 directory is on sys.path
v0_dir = os.path.dirname(os.path.abspath(__file__))
if v0_dir not in sys.path:
    sys.path.insert(0, v0_dir)

from load_examples import example_set
from utils import make_data, sentence2embedding
from models import EMBEDDING_MODEL
from sklearn.cluster import AgglomerativeClustering


def build_config(set_num=1, distance_threshold=7):
    articles = example_set(set_num)
    sentences, words = make_data(articles)
    embeddings = sentence2embedding(sentences, EMBEDDING_MODEL)
    clemb_kwargs = {
        'clustering_algorithm': AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold
        ),
        'embeddings': embeddings,
        'sentences': sentences,
        'words': words,
        'num_articles': len(articles)
    }
    return articles, sentences, words, embeddings, clemb_kwargs

if __name__ == '__main__':
    print("Building example clustering configuration...")
    articles, sentences, words, embeddings, CLEMB_KWARGS = build_config()
    print(f"Loaded {len(articles)} articles, {len(sentences)} sentences, {len(words)} words.")
    print("CLEMB_KWARGS ready:", list(CLEMB_KWARGS.keys()))
