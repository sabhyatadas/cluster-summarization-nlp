import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def load_summarizer(model_code):
    name_dict = {
        "bart": "facebook/bart-large-cnn",
        "distill-bart": "sshleifer/distilbart-cnn-12-6",
        "roberta": "google/roberta2roberta_L-24_cnn_daily_mail",
        "pegasus": "google/pegasus-cnn_dailymail"
    }

    clean_code = model_code.lower().strip()
    if clean_code not in name_dict:
        raise ValueError(f"Model code {model_code} not recognized. Please choose from: {list(name_dict.keys())}")

    model_name = name_dict[clean_code]
    model, tokenizer = None, None

    # Check for available device (cuda, mps, or cpu)
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if "bart" in model_name:
        tokenizer = BartTokenizer.from_pretrained(model_name)
        model = BartForConditionalGeneration.from_pretrained(model_name).to(device)
    
    elif "pegasus" in model_name:
        tokenizer = PegasusTokenizer.from_pretrained(model_name)
        model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)

    elif "roberta" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    return model, tokenizer

def split_long_text(input_text, tokenizer, chunk_size=1024):
    """
    Splits long text into smaller chunks that fit within the model's max token limit.
    """
    if chunk_size is None or chunk_size > 1024:
        chunk_size = 1024

    encoded = tokenizer(input_text, return_tensors="pt", truncation=False, verbose=False)
    tokenized_text = encoded['input_ids']
    total_tokens = tokenized_text.size(1)
    
    # Split into chunks if the total length exceeds the model's limit
    if total_tokens > chunk_size:
        chunks = []
        for i in range(0, total_tokens, chunk_size):
            chunk = tokenized_text[:, i:i+chunk_size]
            chunks.append(chunk)
        return chunks
    else:
        return [tokenized_text]

def summarize_input(
        input_article, 
        model, 
        tokenizer,
        max_length=150,
        min_length=50,
        num_beams=3,
        length_penalty=0.5,
        no_repeat_ngram_size=3
    ):
    if not input_article or not input_article.strip():
        return ""

    device = next(model.parameters()).device

    # Cap tokenizer max length to prevent 1e30 overflow from unconfigured tokenizers
    raw_limit = getattr(tokenizer, 'model_max_length', 1024)
    max_input_length = min(raw_limit or 1024, 1024)

    # Split article if it's too long
    article_chunks = split_long_text(input_article, tokenizer, chunk_size=max_input_length)

    # Store the combined summary for all chunks
    summary_output = []

    for chunk in article_chunks:
        chunk = chunk.to(device)
        attn_mask = torch.ones_like(chunk, device=device)
        
        chunk_len = chunk.shape[1]
        chunk_max_len = max(10, min(int(max_length), max_input_length))
        chunk_min_len = max(5, min(int(min_length), chunk_max_len - 1, chunk_len))

        # Generate summary for the chunk
        summary_ids = model.generate(
            input_ids=chunk,
            attention_mask=attn_mask,
            num_beams=int(num_beams), 
            length_penalty=float(length_penalty), 
            max_length=chunk_max_len, 
            min_length=chunk_min_len, 
            no_repeat_ngram_size=int(no_repeat_ngram_size)
        )
        
        summary_txt = tokenizer.decode(
            summary_ids.squeeze(), 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
        cleaned = summary_txt.replace("<n>", " ").strip()
        if cleaned:
            summary_output.append(cleaned)

    # Combine all chunk summaries
    return " ".join(summary_output)

if __name__ == '__main__':
    print("models/summarizers.py loaded successfully.")
