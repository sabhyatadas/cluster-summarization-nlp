import os
import sys

# Ensure root directory is on sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from configs.model_config import cfg as model_configs
from transformers import AutoTokenizer, AutoModel
import torch


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] 
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def make_embeddings(sentence_list, pool_fn=mean_pooling):
    tokenizer = AutoTokenizer.from_pretrained(model_configs.sent_model_name)
    model = AutoModel.from_pretrained(model_configs.sent_model_name)

    encoded_input = tokenizer(
        sentence_list, 
        padding=True, 
        truncation=True, 
        max_length=model_configs.sent_model_seq_limit, 
        return_tensors='pt'
    )
    with torch.no_grad():
        embeddings = model(**encoded_input)
    
    attn_mask = encoded_input['attention_mask']
    sentence_embeddings = pool_fn(embeddings, attn_mask)
    return sentence_embeddings

def test_embedder():
    sentences = ['This framework generates embeddings for each input sentence',
             'Sentences are passed as a list of string.',
             'The quick brown fox jumps over the lazy dog.']
    
    embeddings = make_embeddings(sentences, mean_pooling)
    print("Embeddings shape:", embeddings.shape)

if __name__ == '__main__':
    test_embedder()
