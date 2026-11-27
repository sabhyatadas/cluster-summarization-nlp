import os
import numpy as np


def example_set(set=1):
    assert set == 1, f'Set{set} was not found among the provided example sets'

    # Check both relative to script directory and cwd
    base_dir = os.path.dirname(__file__)
    candidate_paths = [
        os.path.join(base_dir, "examples", f"set{set}"),
        os.path.join(os.getcwd(), "examples", f"set{set}"),
        os.path.join(base_dir, f"set{set}"),
    ]

    path = None
    for p in candidate_paths:
        if os.path.exists(p) and os.path.isdir(p):
            path = p
            break

    if path is None:
        # Fallback default example strings if directory was not found
        return np.array([
            "Pretrained transformer models provide strong contextual representations for text analysis.",
            "Clustering algorithms group semantically similar sentences to overcome context limits."
        ])

    articles = []
    for file in sorted(os.listdir(path)):
        if file.startswith('.'):
            continue
        file_path = os.path.join(path, file)
        if os.path.isfile(file_path):
            with open(file_path, "r", encoding='utf8') as f:
                data = f.read().replace("\n", " ")
                if data.strip():
                    articles.append(data)

    return np.array(articles)

if __name__ == '__main__':
    articles = example_set()
    print(f"Loaded {len(articles)} example articles:")
    for i, a in enumerate(articles):
        print(f"[{i}] {a[:80]}...")
