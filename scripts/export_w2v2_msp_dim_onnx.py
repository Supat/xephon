#!/usr/bin/env python3
"""
Export audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim to ONNX.

The audeering HF repo only ships PyTorch weights (safetensors +
pytorch_model.bin). The original ONNX export used to live on
Zenodo (doi:10.5281/zenodo.6221127) but the direct file URL has
404'd, and re-exporting from the local checkpoint gives us a
graph that doesn't carry the FP16-converter's duplicate-Cast bug
the historical FP16 file does (the bug ORT 1.20+ rejects with
"two nodes with same node name (graph_input_cast0)" /
"Duplicate definition of name").

Output schema (matches what `W2V2DimensionalSER.invoke` expects):
  inputs:
    signal   [batch, time]  Float32  16 kHz mono waveform, zero-mean
                                     / unit-variance normalized (the
                                     checkpoint's preprocessor_config.json
                                     sets `do_normalize: true`; the
                                     Swift adapter applies that
                                     normalization before invoking ORT)
  outputs:
    logits   [1, 3]         Float32  order = [arousal, dominance, valence]
                                     (see model.yaml's labels list)

We deliberately drop the `hidden_states` output the README's
forward returns — it isn't read on the Swift side and shaving it
keeps the exported graph smaller.

~660 MB FP32. CoreML EP computes in FP16 internally on M-series
anyway, so on-device runtime size is the same as the FP16-quantized
graph would be — we just skip the buggy converter pass.
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
MODEL_DIR = REPO_ROOT / "Models" / "w2v2-msp-dim"
OUT_ONNX = MODEL_DIR / "model.onnx"


class RegressionHead(nn.Module):
    """Verbatim from the upstream README — Linear → tanh → Linear."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return self.out_proj(x)


class EmotionModel(Wav2Vec2PreTrainedModel):
    """Wav2Vec2 backbone + regression head producing [arousal,
    dominance, valence].
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        # README calls `self.init_weights()` here; same incompat
        # with newer transformers as the age-gender export, same
        # workaround — load_state_dict overwrites every parameter
        # immediately after so the init is wasted work anyway.

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        # Pass an explicit all-ones attention mask. transformers 5.x
        # auto-generates the mask via `create_bidirectional_mask` →
        # `sdpa_mask`, which torch.onnx.export's tracer trips on
        # with "tuple index out of range" on q_length.shape[0].
        # Pre-passing the mask routes around the bug.
        attention_mask = torch.ones(
            input_values.shape, dtype=torch.long, device=input_values.device
        )
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        hidden_states = outputs[0]
        pooled = torch.mean(hidden_states, dim=1)
        return self.classifier(pooled)


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

    print(f"[load] EmotionModel from {MODEL_DIR}")
    config = Wav2Vec2Config.from_pretrained(str(MODEL_DIR))
    # Same eager attention pin as the age-gender export — see the
    # forward() comment for the dynamo-tracer rationale.
    config.attn_implementation = "eager"
    model = EmotionModel(config)
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
    nonbenign = [k for k in missing if "position_ids" not in k]
    if nonbenign:
        raise RuntimeError(f"missing weights: {nonbenign}")
    if unexpected:
        print(f"[warn] unexpected weight keys ignored: {unexpected}")
    model = model.eval()

    dummy = torch.randn(1, 16000, dtype=torch.float32)

    print(f"[export] torch.onnx.export → {OUT_ONNX}")
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy,),
            str(OUT_ONNX),
            input_names=["signal"],
            output_names=["logits"],
            dynamic_axes={
                "signal": {0: "batch", 1: "time"},
                "logits": {0: "batch"},
            },
            opset_version=17,
            do_constant_folding=True,
            dynamo=True,
        )

    # The dynamo exporter splits weights into a `model.onnx.data`
    # sidecar by default. ORT's optimizer (at .extended) hits an
    # `Initializer::Initializer() … model_path must not be empty`
    # crash when it tries to constant-fold over the sidecar — the
    # path isn't threaded through that pass for our graph. Re-saving
    # inline yields a single self-contained .onnx with weights
    # embedded; well under the 2 GB protobuf cap at ~660 MB.
    print(f"[inline] re-saving weights inline (dropping .data sidecar)")
    model = onnx.load(str(OUT_ONNX), load_external_data=True)
    for init in model.graph.initializer:
        init.ClearField("external_data")
        if init.data_location == onnx.TensorProto.EXTERNAL:
            init.data_location = onnx.TensorProto.DEFAULT
    onnx.save_model(model, str(OUT_ONNX), save_as_external_data=False)
    # Clean up the now-orphaned sidecar so a fresh fetch doesn't
    # see a stale partial layout.
    data_sidecar = OUT_ONNX.with_name(OUT_ONNX.name + ".data")
    if data_sidecar.exists():
        data_sidecar.unlink()

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
