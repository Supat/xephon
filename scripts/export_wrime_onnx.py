#!/usr/bin/env python3
"""Export the WRIME-fine-tuned RoBERTa to ONNX + ship its tokenizer.

Run via:
    .venv/bin/python scripts/export_wrime_onnx.py

Outputs to Models/wrime-roberta/:
    model.onnx
    tokenizer.json
    tokenizer_config.json
    config.json
    special_tokens_map.json
"""
from pathlib import Path
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "MuneK/roberta-base-japanese-finetuned-wrime"
OUT_DIR = Path(__file__).parent.parent / "Models" / "wrime-roberta"
MAX_TOKENS = 128

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {MODEL_ID}…")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).eval()

    print("Saving tokenizer + config…")
    tokenizer.save_pretrained(OUT_DIR)
    model.config.save_pretrained(OUT_DIR)

    print("Tracing + exporting ONNX…")
    sample = tokenizer(
        "今日は本当に楽しい一日でした。",
        return_tensors="pt",
        padding="max_length",
        max_length=MAX_TOKENS,
        truncation=True,
    )
    input_ids = sample["input_ids"]
    attention_mask = sample["attention_mask"]

    onnx_path = OUT_DIR / "model.onnx"
    with torch.no_grad():
        torch.onnx.export(
            model,
            (input_ids, attention_mask),
            str(onnx_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids":      {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "logits":         {0: "batch"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
    size_mb = onnx_path.stat().st_size / 1024 / 1024
    print(f"Wrote {onnx_path} ({size_mb:.1f} MB)")
    print("Sanity-check the export by inspecting Models/wrime-roberta/")

if __name__ == "__main__":
    main()
