#!/usr/bin/env python3
"""Convert an ONNX model's FP32 weights to FP16 in place.

CoreML and the Apple Neural Engine compute in FP16 internally, so the
on-disk FP32 weights cost storage without buying accuracy. Halving
them shrinks the bundle by hundreds of MB across the three SER models.

Usage:
    python scripts/quantize_onnx_fp16.py <model.onnx>
    python scripts/quantize_onnx_fp16.py <model.onnx> --output <out.onnx>
    python scripts/quantize_onnx_fp16.py <model.onnx> --no-keep-io

`--keep-io-types` (default on) inserts Cast nodes at the graph
boundary so the external interface stays FP32 — Swift callers send
FP32 and read FP32 back unchanged. Without it, audio and logits I/O
would also become FP16 and the Swift inferencers would need rework.

External-data layouts (a `model.data` companion file) are detected
from the source directory and preserved on save.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import onnx
# `onnxruntime.transformers.float16` is a fork of onnxconverter_common's
# converter tuned for transformer encoders. It exposes
# `force_fp16_initializers=True`, which forces all weight initializers
# to convert in lockstep with the ops that consume them — without it,
# encoder LayerNorm gamma/beta and attention-score divisor constants
# leak through as FP32 and ORT rejects the model at load with
# "Type parameter (T) ... bound to different types."
from onnxruntime.transformers import float16


def has_external_data(src: Path) -> bool:
    """A `*.data` sibling means the source ONNX uses external-data layout."""
    if not src.parent.exists():
        return False
    return any(
        f.suffix == ".data" and f.is_file()
        for f in src.parent.iterdir()
    )


def quantize(src: Path, dst: Path, keep_io_types: bool) -> None:
    print(f"[fp16] loading {src}")
    # `load_external_data=True` resolves any *.data sibling into raw_data
    # on the loaded proto, so we can freely rewrite the file afterward.
    model = onnx.load(str(src), load_external_data=True)
    src_size = src.stat().st_size + sum(
        f.stat().st_size for f in src.parent.glob("*.data") if f.is_file()
    )

    print(f"[fp16] converting (keep_io_types={keep_io_types})")
    converted = float16.convert_float_to_float16(
        model,
        keep_io_types=keep_io_types,
        # Shape inference fails on some dynamic-axes graphs (FunASR's
        # masked-attention path in particular). Skip it; the conversion
        # itself doesn't need static shapes.
        disable_shape_infer=True,
        # Convert ALL weight initializers (LayerNorm gamma/beta, attention
        # divisors, etc.) along with the ops that consume them. Without
        # this flag the converter leaves some FP32 weights connected to
        # FP16-converted ops, and ORT rejects the resulting graph at
        # load time with a type-mismatch error.
        force_fp16_initializers=True,
    )

    use_external = has_external_data(src)
    if use_external:
        data_name = dst.stem + ".data"
        # Strip stale external-data refs from initializers so save_model
        # writes a fresh data file rather than appending to the old one.
        for init in converted.graph.initializer:
            init.ClearField("external_data")
            if init.data_location == onnx.TensorProto.EXTERNAL:
                init.data_location = onnx.TensorProto.DEFAULT
        # Remove the old data file before saving so we don't half-clobber it.
        old_data = dst.with_name(data_name)
        if old_data.exists():
            old_data.unlink()
        print(f"[fp16] saving with external data → {dst} (+ {data_name})")
        onnx.save_model(
            converted,
            str(dst),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=data_name,
            size_threshold=1024,
            convert_attribute=False,
        )
    else:
        print(f"[fp16] saving → {dst}")
        onnx.save_model(converted, str(dst))

    onnx.checker.check_model(str(dst))

    dst_size = dst.stat().st_size + sum(
        f.stat().st_size for f in dst.parent.glob("*.data") if f.is_file()
    )
    print(
        f"[fp16] {src_size / 1024 / 1024:.1f} MB → "
        f"{dst_size / 1024 / 1024:.1f} MB"
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--output")
    p.add_argument("--no-keep-io", dest="keep_io", action="store_false")
    p.set_defaults(keep_io=True)
    args = p.parse_args()

    src = Path(args.model)
    if not src.exists():
        print(f"[fp16] not found: {src}", file=sys.stderr)
        return 2

    dst = Path(args.output) if args.output else src
    quantize(src, dst, args.keep_io)
    return 0


if __name__ == "__main__":
    sys.exit(main())
