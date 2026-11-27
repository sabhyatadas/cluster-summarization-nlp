"""
SPECTRA-Sum: Quantized Acceleration Engine
Provides low-precision INT8 dynamic quantization and hardware-accelerated generation.
"""

import os
import torch
import warnings
from typing import Tuple, Dict, Any, Optional
from transformers import BartTokenizer, BartForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM


class QuantizedSummarizer:
    """
    Hardware-accelerated sequence-to-sequence summarizer with support for
    FP32, FP16, dynamic INT8 quantization, and torch.compile.
    """

    MODEL_REGISTRY = {
        "distill-bart": "sshleifer/distilbart-cnn-12-6",
        "bart": "facebook/bart-large-cnn",
        "pegasus": "google/pegasus-cnn_dailymail",
        "roberta": "google/roberta2roberta_L-24_cnn_daily_mail",
    }

    def __init__(
        self,
        model_name_or_key: str = "distill-bart",
        precision: str = "fp32",
        device: Optional[str] = None,
        use_torch_compile: bool = False,
    ):
        """
        Args:
            model_name_or_key: Registered model key or HuggingFace repo path.
            precision: Precision mode: 'fp32', 'fp16', 'int8'.
            device: 'cuda', 'mps', 'cpu', or None for auto-detection.
            use_torch_compile: Whether to compile graph with torch.compile.
        """
        self.model_key = model_name_or_key.lower().strip()
        self.model_path = self.MODEL_REGISTRY.get(self.model_key, model_name_or_key)
        self.precision = precision.lower()
        self.use_torch_compile = use_torch_compile

        # Detect optimal device
        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Dynamic INT8 is natively executed on CPU in PyTorch
        if self.precision == "int8":
            self.device = torch.device("cpu")

        self.tokenizer, self.model = self._load_and_optimize()

    def _load_and_optimize(self) -> Tuple[Any, Any]:
        """Loads weights and applies quantization / compilation transformations."""
        # Load tokenizer
        if "bart" in self.model_path:
            tokenizer = BartTokenizer.from_pretrained(self.model_path)
            model = BartForConditionalGeneration.from_pretrained(self.model_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)

        # Apply precision optimizations
        if self.precision == "fp16":
            if self.device.type in ["cuda", "mps"]:
                model = model.half()
            else:
                warnings.warn("FP16 on CPU may have limited support; falling back to FP32.")

        elif self.precision == "int8":
            # Dynamic quantization converts Linear layers to qint8
            try:
                if "qnnpack" in torch.backends.quantized.supported_engines:
                    torch.backends.quantized.engine = "qnnpack"
                elif "fbgemm" in torch.backends.quantized.supported_engines:
                    torch.backends.quantized.engine = "fbgemm"

                model = torch.ao.quantization.quantize_dynamic(
                    model,
                    {torch.nn.Linear},
                    dtype=torch.qint8
                )
            except Exception as e:
                warnings.warn(f"Dynamic INT8 quantization failed: {e}. Keeping baseline precision.")

        model = model.to(self.device)
        model.eval()

        # Optional PyTorch 2.x graph compilation
        if self.use_torch_compile and hasattr(torch, "compile"):
            try:
                model = torch.compile(model, mode="reduce-overhead")
            except Exception as e:
                warnings.warn(f"torch.compile skipped: {e}")

        return tokenizer, model

    def get_parameter_count(self) -> int:
        """Returns total trainable + fixed parameter count."""
        return sum(p.numel() for p in self.model.parameters())

    def summarize(
        self,
        text: str,
        max_length: int = 150,
        min_length: int = 40,
        num_beams: int = 2,
        length_penalty: float = 1.0,
        no_repeat_ngram_size: int = 3,
    ) -> Tuple[str, int]:
        """
        Executes accelerated generation and returns (generated_summary, tokens_produced).
        """
        text = text.strip()
        if not text:
            return "", 0

        # Tokenize with input length clamp
        raw_limit = getattr(self.tokenizer, "model_max_length", 1024)
        max_input = min(raw_limit or 1024, 1024)

        inputs = self.tokenizer(
            text,
            max_length=max_input,
            truncation=True,
            return_tensors="pt"
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        input_len = input_ids.shape[1]
        clamped_max_len = max(10, min(int(max_length), max_input))
        clamped_min_len = max(5, min(int(min_length), clamped_max_len - 1, input_len))

        with torch.no_grad():
            summary_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=clamped_max_len,
                min_length=clamped_min_len,
                num_beams=int(num_beams),
                length_penalty=float(length_penalty),
                no_repeat_ngram_size=int(no_repeat_ngram_size),
                use_cache=True,
            )

        tokens_produced = summary_ids.shape[1]
        summary_text = self.tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        return summary_text.replace("<n>", " ").strip(), tokens_produced
