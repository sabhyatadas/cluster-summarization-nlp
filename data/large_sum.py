import warnings


class PubMedDataset:
	def __init__(self, dataset_name="ccdv/pubmed-summarization", split="train"):
		self.dataset_name = dataset_name
		self.split = split
		try:
			from datasets import load_dataset
			self.dataset = load_dataset(self.dataset_name, split=self.split)
			self.inputs = self.dataset['article']
			self.targets = self.dataset['abstract']
		except Exception as e:
			warnings.warn(f"Could not load online dataset '{dataset_name}': {e}. Using sample mock data.")
			self.inputs = [
				"Medical imaging techniques such as MRI and CT scans are essential in diagnosing complex diseases. "
				"Recent advancements in deep learning have enabled automated anomaly detection in radiology scans, "
				"improving both speed and precision in clinical workflows."
			]
			self.targets = [
				"Deep learning enhances automated anomaly detection in medical imaging workflows."
			]
	
	def __getitem__(self, idx):
		return self.inputs[idx].replace("\n", " "), self.targets[idx].replace("\n", " ")

	def get_sample(self, idx):
		article, abstract = self[idx]
		return {'article': article, 'abstract': abstract}

	def __len__(self):
		return len(self.inputs)


# Alias for legacy compatibility (v0/test.py)
LargeArticleDataset = PubMedDataset

if __name__ == '__main__':
	ds = PubMedDataset()
	print(f"Dataset length: {len(ds)}")
	sample = ds.get_sample(0)
	print("Sample keys:", list(sample.keys()))
