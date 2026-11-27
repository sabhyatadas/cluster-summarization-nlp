import os
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import nltk

# Add workspace nltk_data if available
v0_dir = os.path.dirname(os.path.abspath(__file__))
ws_nltk = os.path.abspath(os.path.join(v0_dir, '..', '..', '..', '.venv', 'nltk_data'))
if os.path.exists(ws_nltk) and ws_nltk not in nltk.data.path:
    nltk.data.path.insert(0, ws_nltk)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
from sklearn.cluster import AgglomerativeClustering
from transformers import BartTokenizer, BartForConditionalGeneration
from sentence_transformers import SentenceTransformer
warnings.filterwarnings("ignore")


def make_data(articles):
    sentences = []
    words = []

    for item in articles:
        tokenized_article = sent_tokenize(item)
        for sentence in tokenized_article:
            word_tokens = word_tokenize(sentence)
            for token in word_tokens:
                words.append(token)
            sentences.append(sentence)
    sentences, words = np.array(sentences), np.array(words)
    return sentences, words


def sentence2embedding(sentences_array, model):
    return model.encode(sentences_array)


def bart_summarize(
    text,
    model,
    tokenizer,
    device='cpu',
    num_beams=3,
    length_penalty=2.0,
    max_length=60,
    min_length=10,
    no_repeat_ngram_size=3,
):
    text = text.replace('\n', ' ').strip()
    if not text:
        return ""

    encoded = tokenizer(
        [text], return_tensors='pt', max_length=1024, truncation=True
    )
    text_input_ids = encoded['input_ids'].to(device)
    attn_mask = encoded.get('attention_mask', None)
    if attn_mask is not None:
        attn_mask = attn_mask.to(device)

    summary_ids = model.generate(
        input_ids=text_input_ids,
        attention_mask=attn_mask,
        num_beams=int(num_beams),
        length_penalty=float(length_penalty),
        max_length=int(max_length),
        min_length=min(int(min_length), text_input_ids.shape[1]),
        no_repeat_ngram_size=int(no_repeat_ngram_size)
    )
    summary_txt = tokenizer.decode(
        summary_ids.squeeze(), skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return summary_txt.strip()


class ClusterSentenceEmbeddings():
    def __init__(self, clustering_algorithm, embeddings, sentences, words, num_articles):
        self.num_articles = num_articles
        self.embeddings = np.array(embeddings)
        self.sentences = np.array(sentences)
        self.words = np.array(words)
        self.clustering_algorithm = clustering_algorithm
        self.clustering = clustering_algorithm.fit(self.embeddings)

    def cluster_outputs(self):
        n_clusters = getattr(self.clustering, 'n_clusters_', getattr(self.clustering, 'n_clusters', 1)) or 1
        labels = self.clustering.labels_
        outputs = {
            'num_clusters': n_clusters,
            'labels': labels,
            'components': getattr(self.clustering, 'n_connected_components_', 1),
            'distances': getattr(self.clustering, 'distances_', None),
            'exp_sentences_per_cluster': math.ceil(len(self.sentences) / n_clusters) if n_clusters else 0,
            'exp_words_per_cluster': math.ceil(len(self.words) / n_clusters) if n_clusters else 0,
        }
        return outputs

    def visualize_outputs(self, method='tsne', dims=2):
        cluster_outputs = self.cluster_outputs()
        n_clusters = cluster_outputs['num_clusters']
        if len(self.embeddings) < 2:
            print("Not enough samples to visualize.")
            return

        if method == 'tsne':
            perp = min(30, max(1, len(self.embeddings) - 1))
            tSNE = TSNE(n_components=dims, perplexity=perp, learning_rate='auto', init='random', random_state=42)
            projected_embeddings = np.array(tSNE.fit_transform(self.embeddings))

            indiv_clusters = []
            for label in range(n_clusters):
                indiv_clusters.append(projected_embeddings[cluster_outputs['labels'] == label])

            plt.figure(figsize=(10, 8))
            cmap = matplotlib.colormaps.get_cmap('hsv')
            for i, cluster in enumerate(indiv_clusters):
                if len(cluster) > 0:
                    plt.scatter(
                        cluster[:, :1],
                        cluster[:, 1:],
                        color=cmap(i / max(1, n_clusters)),
                        alpha=0.6,
                        label=f"Cluster {i}",
                        s=100,
                    )
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            plt.title('tSNE Results: Sentence Embeddings')
            plt.legend(title='Clustering Labels')
            plt.show()

        elif method == 'pca':
            pca = PCA(n_components=min(dims, len(self.embeddings), self.embeddings.shape[1]))
            projected_embeddings = np.array(pca.fit_transform(self.embeddings))

            indiv_clusters = []
            for label in range(n_clusters):
                indiv_clusters.append(projected_embeddings[cluster_outputs['labels'] == label])

            plt.figure(figsize=(10, 8))
            cmap = matplotlib.colormaps.get_cmap('hsv')
            for i, cluster in enumerate(indiv_clusters):
                if len(cluster) > 0:
                    plt.scatter(
                        cluster[:, :1],
                        cluster[:, 1:],
                        color=cmap(i / max(1, n_clusters)),
                        alpha=0.6,
                        label=f"Cluster {i}",
                        s=100,
                    )
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            plt.title('PCA Results: Sentence Embeddings')
            plt.legend(title='Clustering Labels')
            plt.show()

    def get_sentence_clusters(self):
        cluster_outputs = self.cluster_outputs()
        indiv_clusters = []
        for label in range(cluster_outputs['num_clusters']):
            single_cluster = self.sentences[cluster_outputs['labels'] == label]
            chunk = " ".join([item.strip() for item in single_cluster if item.strip()])
            if chunk:
                indiv_clusters.append(chunk)

        if not indiv_clusters and len(self.sentences) > 0:
            indiv_clusters = [" ".join(self.sentences)]

        return np.array(indiv_clusters)

def get_summary(articles, threshold=7.0):
    EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    SUMMARIZATION_TOKENIZER = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    SUMMARIZATION_MODEL = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
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
    sample_text = ["This is sentence one. This is sentence two.", "Another document with sentence three."]
    sents, wrds = make_data(sample_text)
    print(f"make_data result: {len(sents)} sentences, {len(wrds)} words.")
