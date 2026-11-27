import os
import sys

# Ensure root directory is on sys.path
root_dir = os.path.abspath(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from models.summarizers import load_summarizer, summarize_input

def run_tests():
    print("Testing model loader and summarization pipeline...")
    # Test with distill-bart (lightweight and cached)
    try:
        print("\nLoading distill-bart model...")
        model, tokenizer = load_summarizer("distill-bart")
        print("distill-bart loaded successfully!")
        
        sample_article = (
            "Extractive summarization selects the most representative sentences from a document. "
            "Clustering algorithms group semantically similar sentences based on transformer embeddings. "
            "By summarizing each cluster individually and combining the results, the system handles documents "
            "longer than the transformer sequence limit without losing key information."
        )
        print("Generating test summary...")
        summary = summarize_input(sample_article, model, tokenizer, max_length=60, min_length=15)
        print("Generated summary:\n", summary)
        print("\nAll tests passed successfully!")
    except Exception as e:
        print(f"Error during test: {e}")
        raise

if __name__ == '__main__':
    run_tests()