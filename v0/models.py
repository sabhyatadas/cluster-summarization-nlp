from transformers import BartTokenizer, BartForConditionalGeneration
from sentence_transformers import SentenceTransformer

_EMBEDDING_MODEL = None
_SUMMARIZATION_TOKENIZER = None
_SUMMARIZATION_MODEL = None

def get_embedding_model(model_name='all-MiniLM-L6-v2'):
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        _EMBEDDING_MODEL = SentenceTransformer(model_name)
    return _EMBEDDING_MODEL

def get_summarization_tokenizer(model_name='facebook/bart-large-cnn'):
    global _SUMMARIZATION_TOKENIZER
    if _SUMMARIZATION_TOKENIZER is None:
        _SUMMARIZATION_TOKENIZER = BartTokenizer.from_pretrained(model_name)
    return _SUMMARIZATION_TOKENIZER

def get_summarization_model(model_name='facebook/bart-large-cnn'):
    global _SUMMARIZATION_MODEL
    if _SUMMARIZATION_MODEL is None:
        _SUMMARIZATION_MODEL = BartForConditionalGeneration.from_pretrained(model_name)
    return _SUMMARIZATION_MODEL

class _LazyModel:
    def __init__(self, loader_fn):
        self._loader_fn = loader_fn
        self._obj = None

    def _get_obj(self):
        if self._obj is None:
            self._obj = self._loader_fn()
        return self._obj

    def __getattr__(self, name):
        return getattr(self._get_obj(), name)

    def __call__(self, *args, **kwargs):
        return self._get_obj()(*args, **kwargs)

EMBEDDING_MODEL = _LazyModel(get_embedding_model)
SUMMARIZATION_TOKENIZER = _LazyModel(get_summarization_tokenizer)
SUMMARIZATION_MODEL = _LazyModel(get_summarization_model)

if __name__ == '__main__':
    print("Testing lazy model loading...")
    emb = get_embedding_model()
    print("Embedding model loaded:", type(emb))
