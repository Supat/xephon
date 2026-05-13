#!/usr/bin/env python3
"""
Export audeering/wav2vec2-large-robust-6-ft-age-gender to ONNX.

The Hugging Face repo only ships PyTorch weights — the model has a
custom head that isn't one of the standard `transformers` classes,
so we instantiate it from the class definition the upstream README
documents, load the local weights, and run `torch.onnx.export` on a
small wrapper that exposes (age_logits, gender_probs) as named
outputs.

Output schema (matches what an `OnnxAgeGenderSER` Swift adapter
would consume):
  inputs:
    speech       [batch, time]  Float32  16 kHz mono waveform, zero-
                                         mean / unit-variance normalized
                                         (the checkpoint's
                                         preprocessor_config.json sets
                                         `do_normalize: true`; the
                                         Swift adapter applies that
                                         normalization before invoking
                                         ORT, so the exported graph
                                         only sees the normalized
                                         tensor)
  outputs:
    logits_age   [batch, 1]     Float32  regression in [0, 1],
                                         multiply by 100 for years
    logits_gender [batch, 3]    Float32  softmax already applied
                                         upstream; order = [female,
                                         male, child]. Authoritative
                                         source: the audeering
                                         tutorial notebook
                                         (github.com/audeering/
                                         w2v2-age-gender-how-to,
                                         cell 5) which states
                                         "logits_gender expresses
                                         the confidence for being
                                         female, male or child",
                                         consistent with the shipped
                                         config.json id2label
                                         {0: female, 1: male,
                                         2: child}. The Hugging Face
                                         model card's body prose and
                                         example output column
                                         header ("child, female,
                                         male") contradict the
                                         actual training-time label
                                         encoding and should not be
                                         trusted.

The exported graph weighs ~330 MB — comfortably under the 2 GB
protobuf cap, so unlike emotion2vec we don't need the external-data
split. The downstream FP16 quantize step in `scripts/fetch_models.sh`
halves it to ~165 MB on disk.
"""
from __future__ import annotations

import sys
from pathlib import Path

import onnx
import torch
import torch.nn as nn
from safetensors.torch import load_file as load_safetensors
from transformers import Wav2Vec2Config
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "Models" / "w2v2-age-gender"
OUT_ONNX = MODEL_DIR / "model.onnx"


class ModelHead(nn.Module):
    """Two-layer regression / classification head — verbatim from the
    upstream README.
    """

    def __init__(self, config, num_labels: int):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return self.out_proj(x)


class AgeGenderModel(Wav2Vec2PreTrainedModel):
    """Wav2Vec2 backbone + two heads (age regression, gender softmax)."""

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = ModelHead(config, 1)
        self.gender = ModelHead(config, 3)
        # README calls `self.init_weights()` here; newer transformers
        # versions trip an `all_tied_weights_keys` lookup that this
        # custom subclass doesn't define. Skip it — we overwrite
        # every parameter with `load_state_dict` immediately after
        # construction anyway.

    def forward(self, input_values: torch.Tensor):
        # Pass an explicit all-ones attention mask. Newer
        # `transformers` versions auto-generate the mask via
        # `create_bidirectional_mask`, which under torch.onnx.export
        # traces shape ops in a way that ends up indexing a tuple
        # ("q_length.shape[0]"). Passing the mask up front bypasses
        # that path entirely.
        attention_mask = torch.ones(
            input_values.shape, dtype=torch.long, device=input_values.device
        )
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        hidden_states = outputs[0]
        # Mean-pool across the time dimension into a single utterance
        # vector before each head.
        pooled = torch.mean(hidden_states, dim=1)
        logits_age = self.age(pooled)
        # Upstream applies softmax to gender at the modeling layer —
        # mirror that so the Swift side gets probabilities directly
        # and can argmax without re-implementing the softmax.
        logits_gender = torch.softmax(self.gender(pooled), dim=1)
        return logits_age, logits_gender


def main() -> int:
    if OUT_ONNX.exists():
        print(f"[skip] {OUT_ONNX} already exists")
        return 0

    if not MODEL_DIR.exists():
        print(
            f"[error] {MODEL_DIR} not found — run scripts/fetch_models.sh first",
            file=sys.stderr,
        )
        return 2

    # `from_pretrained` on this custom subclass trips a newer-
    # `transformers` internal (`all_tied_weights_keys`) the upstream
    # README predates. Load the config + state_dict manually instead
    # — same result, no version coupling.
    print(f"[load] AgeGenderModel from {MODEL_DIR}")
    config = Wav2Vec2Config.from_pretrained(str(MODEL_DIR))
    # transformers 5.x defaults Wav2Vec2 to the `sdpa` attention
    # implementation, which routes through `create_bidirectional_mask`
    # → `sdpa_mask`. Under torch.onnx.export's tracer, `q_length`
    # ends up as a tuple of shape ops and `q_length.shape[0]`
    # raises IndexError. Force the eager attention path — same math,
    # exportable.
    config.attn_implementation = "eager"
    model = AgeGenderModel(config)
    state_path = MODEL_DIR / "model.safetensors"
    if state_path.exists():
        state = load_safetensors(str(state_path))
    else:
        state = torch.load(
            str(MODEL_DIR / "pytorch_model.bin"),
            map_location="cpu",
            weights_only=True,
        )
    missing, unexpected = model.load_state_dict(state, strict=False)
    # `position_ids` is a buffer the newer Wav2Vec2 registers but
    # the checkpoint doesn't carry — safe to leave at the default.
    nonbenign = [k for k in missing if "position_ids" not in k]
    if nonbenign:
        raise RuntimeError(f"missing weights: {nonbenign}")
    if unexpected:
        # Don't fail on extras — sometimes a checkpoint carries
        # auxiliary buffers (e.g., masked_spec_embed) that aren't
        # used at inference.
        print(f"[warn] unexpected weight keys ignored: {unexpected}")
    model = model.eval()

    # 1 s of dummy audio at the model's training sample rate.
    dummy = torch.randn(1, 16000, dtype=torch.float32)

    print(f"[export] torch.onnx.export → {OUT_ONNX}")
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy,),
            str(OUT_ONNX),
            input_names=["speech"],
            output_names=["logits_age", "logits_gender"],
            dynamic_axes={
                "speech": {0: "batch", 1: "time"},
                "logits_age": {0: "batch"},
                "logits_gender": {0: "batch"},
            },
            opset_version=17,
            do_constant_folding=True,
            # transformers 5.x's `sdpa_mask` indexes
            # `q_length.shape[0]`, which under the legacy jit
            # tracer turns into a tuple — fails before the export
            # even completes. The dynamo exporter symbolicizes
            # shape ops the right way and traces this graph clean.
            dynamo=True,
        )

    onnx.checker.check_model(str(OUT_ONNX))
    print(f"[wrote] {OUT_ONNX}")
    print_schema(OUT_ONNX)
    return 0


def print_schema(path: Path) -> None:
    m = onnx.load(str(path))

    def fmt(t: onnx.TypeProto.Tensor) -> str:
        elem = onnx.TensorProto.DataType.Name(t.elem_type)
        dims = [
            d.dim_value if d.dim_value > 0 else (d.dim_param or "?")
            for d in t.shape.dim
        ]
        return f"{elem}[{', '.join(map(str, dims))}]"

    print("\n[schema]")
    print("  inputs:")
    for i in m.graph.input:
        print(f"    {i.name}: {fmt(i.type.tensor_type)}")
    print("  outputs:")
    for o in m.graph.output:
        print(f"    {o.name}: {fmt(o.type.tensor_type)}")


if __name__ == "__main__":
    sys.exit(main())
