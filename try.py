from transformers import BartTokenizer, BartForConditionalGeneration
import time

# Initialize tokenizer and model with DistilBART
model_name = "sshleifer/distilbart-cnn-12-6"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Input article for summarization (sample technical document)
ARTICLE = """Natural language processing systems have witnessed rapid progress due to deep transformer architectures. 
Pretrained sequence-to-sequence networks excel at document synthesis, abstractive summarization, and domain transfer. 
However, deploying these models on extensive technical reports or lengthy narratives introduces notable inference challenges. 
Fixed positional embeddings restrict the number of tokens processed in a single forward pass, often forcing crude truncation. 
By combining sentence-level dense embeddings with semantic clustering algorithms, large documents can be divided into topical 
sub-groups. Each cluster is independently synthesized, allowing comprehensive semantic representation across long documents 
without surpassing hardware or architectural constraints."""

# Measure the time taken for summarization
start_time = time.time()
inputs = tokenizer([ARTICLE], max_length=1024, return_tensors="pt", truncation=True)
summary_ids = model.generate(inputs["input_ids"], max_length=250, min_length=40, do_sample=False)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
end_time = time.time()

# Print the summary
print("Summary:", summary.strip())

# Print the time taken
print(f"Time taken for summarization: {end_time - start_time:.2f} seconds")
