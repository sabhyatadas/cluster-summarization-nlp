import os
import sys

# Ensure project root directory is on sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from data.large_sum import LargeArticleDataset

dataset = LargeArticleDataset()
sample = dataset.get_sample(0)
print("Dataset sample keys:", list(sample.keys()))
print("Article preview:", sample['article'][:100], "...")
print("Abstract preview:", sample['abstract'][:100], "...")