import warnings


class MultiSumDataset:
	def __init__(self, dataset_name="multi_news", split="train"):
		self.dataset_name = dataset_name or "multi_news"
		self.split = split
		try:
			from datasets import load_dataset
			self.dataset = load_dataset(self.dataset_name, split=self.split)
			self.inputs = self.dataset['document']
			self.targets = self.dataset['summary']
		except Exception as e:
			warnings.warn(f"Could not load online dataset '{self.dataset_name}': {e}. Using sample mock data.")
			self.inputs = [
				"Multiple news outlets reported on the international climate conference yesterday. "
				"Delegates from over 190 countries gathered to negotiate emission reduction targets, "
				"focusing on renewable energy transitions and green industrial subsidies."
			]
			self.targets = [
				"Over 190 countries convened to negotiate new emission reduction targets and green energy initiatives."
			]

	def __getitem__(self, idx):
		return self.inputs[idx], self.targets[idx]

	def get_sample(self, idx):
		doc, summ = self[idx]
		return {'document': doc, 'summary': summ}

	def __len__(self):
		return len(self.inputs)

if __name__ == '__main__':
	ds = MultiSumDataset()
	print(f"Dataset length: {len(ds)}")
	sample = ds.get_sample(0)
	print("Sample keys:", list(sample.keys()))