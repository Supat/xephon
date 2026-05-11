#!/usr/bin/env python3
"""
Export emotion2vec_plus_large to ONNX with the full utterance-level
classification pipeline that FunASR's runtime applies in
`Emotion2vec.inference`:

    speech [batch, time]
      → utterance layer-norm
      → extract_features (encoder)        # [batch, time_frames, 1024]
      → mean over time                    # [batch, 1024]
      → self.proj (Linear 1024 → 9)       # [batch, 9]
      → logits (softmax applied in Swift)

FunASR's built-in `model.export(type="onnx")` only writes the encoder, so
we wrap the model in a small `nn.Module` that performs the full chain
above and run `torch.onnx.export` on the wrapper.

Output schema (matches what Emotion2VecCategoricalSER.invoke expects on
the Swift side, sans `speech_lengths` since FunASR's encoder doesn't take
one):
  inputs:
    speech    [batch, time]  Float32   raw 16 kHz mono waveform
  outputs:
    logits    [batch, 9]     Float32   order = LABEL_ORDER
                                       (angry, disgusted, fearful, happy,
                                        neutral, other, sad, surprised,
                                        unknown)

The graph + weights exceed 2 GB and are split into `model.onnx` (graph)
and `model.data` (external weights). Both must ship in the same
directory; we put them under `Models/emotion2vec-plus-large/emotion2vec_onnx/`
and `project.yml` folder-references the subdirectory so the app bundle
gets `Xephon.app/emotion2vec_onnx/model.onnx` + `model.data`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import onnx
import torch
import torch.nn as nn
from funasr import AutoModel

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "Models" / "emotion2vec-plus-large"
OUT_DIR = MODEL_DIR / "emotion2vec_onnx"
OUT_ONNX = OUT_DIR / "model.onnx"
OUT_DATA_NAME = "model.data"


class Emotion2VecClassifier(nn.Module):
    """Wrap FunASR's Emotion2vec model so a single forward pass produces
    9-class logits — encoder → mean-pool → linear head — matching what
    `Emotion2vec.inference` does at runtime.
    """

    def __init__(self, base):
        super().__init__()
        self.base = base
        self.normalize = bool(base.cfg.normalize)

    def forward(self, speech: torch.Tensor) -> torch.Tensor:
        # `speech`: [batch, time], 16 kHz mono Float32.
        if self.normalize:
            # FunASR runs `F.layer_norm(source, source.shape)` — normalize
            # over the entire tensor (batch+time) per call. Compute mean
            # and variance manually so the op is ONNX-exportable with
            # dynamic axes (LayerNorm needs a static normalized_shape).
            flat = speech.reshape(speech.shape[0], -1)
            mean = flat.mean(dim=1, keepdim=True)
            var = flat.var(dim=1, keepdim=True, unbiased=False)
            speech = (speech - mean) / (var + 1e-5).sqrt()

        feats = self.base.extract_features(speech, padding_mask=None)
        x = feats["x"]              # [batch, time_frames, 1024]
        x = x.mean(dim=1)           # [batch, 1024]
        return self.base.proj(x)    # [batch, 9]


def main() -> int:
    if OUT_ONNX.exists() and (OUT_DIR / OUT_DATA_NAME).exists():
        print(f"[skip] {OUT_ONNX} already exists")
        return 0

    if not MODEL_DIR.exists():
        print(f"[error] {MODEL_DIR} not found — run scripts/fetch_models.sh first", file=sys.stderr)
        return 2

    print(f"[load] FunASR Emotion2vec from {MODEL_DIR}")
    auto = AutoModel(model=str(MODEL_DIR), disable_update=True)
    base = auto.model.eval()
    if not hasattr(base, "proj") or base.proj is None:
        raise RuntimeError("model has no `proj` head — wrong checkpoint?")
    wrapper = Emotion2VecClassifier(base).eval()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 16000, dtype=torch.float32)

    print(f"[export] torch.onnx.export → {OUT_ONNX}")
    with torch.no_grad():
        # The pipeline contains ops that the legacy ONNX exporter handles
        # cleanly; the new dynamo exporter struggles with parts of
        # FunASR's masked-attention path. Use the legacy path explicitly.
        torch.onnx.export(
            wrapper,
            (dummy,),
            str(OUT_ONNX),
            input_names=["speech"],
            output_names=["logits"],
            dynamic_axes={
                "speech": {0: "batch", 1: "time"},
                "logits": {0: "batch"},
            },
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )

    # The exported file holds weights inline; if it exceeds the 2 GB
    # protobuf cap (it does for emotion2vec_plus_large), re-save with
    # external data so the runtime can load it. ONNX raises during
    # `save_model` only if the model is invalid, not if it's oversize, so
    # we always re-save through the external-data path for safety.
    onnx_model = onnx.load(str(OUT_ONNX))
    onnx.save_model(
        onnx_model,
        str(OUT_ONNX),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=OUT_DATA_NAME,
        size_threshold=1024,
        convert_attribute=False,
    )
    onnx.checker.check_model(str(OUT_ONNX))
    print(f"[wrote] {OUT_ONNX} (+ {OUT_DATA_NAME})")
    print_schema(OUT_ONNX)
    return 0


def print_schema(path: Path) -> None:
    m = onnx.load(str(path))

    def fmt(t: onnx.TypeProto.Tensor) -> str:
        elem = onnx.TensorProto.DataType.Name(t.elem_type)
        dims = [d.dim_value if d.dim_value > 0 else (d.dim_param or "?") for d in t.shape.dim]
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
