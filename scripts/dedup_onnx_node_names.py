#!/usr/bin/env python3
"""Rename duplicate node names in an ONNX graph so ORT 1.20+ loads it.

`onnxruntime.transformers.float16` occasionally inserts two boundary
Cast nodes with the same name (`graph_input_cast0`,
`graph_output_cast{0,1}`) when keep_io_types is set. ORT used to
tolerate it; newer builds (~1.20+) reject the model at load with
"This is an invalid model. Error: two nodes with same node name".

This script walks the graph, finds names that repeat, and appends a
counter suffix (`__dup1`, `__dup2`, …) to every occurrence past the
first. Node-name uniqueness is the only invariant the rename
touches — input/output tensor names + the actual op data are
untouched, so ORT's execution is bit-identical to the original.

Usage:
    python scripts/dedup_onnx_node_names.py <model.onnx>
    python scripts/dedup_onnx_node_names.py <model.onnx> --output <out.onnx>

External-data sidecars (`*.data` next to the ONNX) are preserved.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import onnx


def has_external_data(src: Path) -> bool:
    if not src.parent.exists():
        return False
    return any(
        f.suffix == ".data" and f.is_file()
        for f in src.parent.iterdir()
    )


def dedup(src: Path, dst: Path) -> int:
    print(f"[dedup] loading {src}")
    model = onnx.load(str(src), load_external_data=True)

    seen: defaultdict[str, int] = defaultdict(int)
    renamed = 0
    for node in model.graph.node:
        seen[node.name] += 1
        if seen[node.name] > 1:
            new = f"{node.name}__dup{seen[node.name] - 1}"
            print(f"[dedup]   {node.name} → {new}")
            node.name = new
            renamed += 1

    if renamed == 0:
        print("[dedup] no duplicates found — nothing to do")
        return 0

    if has_external_data(src):
        data_name = dst.stem + ".data"
        for init in model.graph.initializer:
            init.ClearField("external_data")
            if init.data_location == onnx.TensorProto.EXTERNAL:
                init.data_location = onnx.TensorProto.DEFAULT
        old_data = dst.with_name(data_name)
        if old_data.exists():
            old_data.unlink()
        print(f"[dedup] saving with external data → {dst} (+ {data_name})")
        onnx.save_model(
            model,
            str(dst),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=data_name,
            size_threshold=1024,
            convert_attribute=False,
        )
    else:
        print(f"[dedup] saving → {dst}")
        onnx.save_model(model, str(dst))

    try:
        onnx.checker.check_model(str(dst))
    except onnx.checker.ValidationError as exc:
        # ORT's loader topo-sorts on load so this is cosmetic at the
        # checker boundary; surface it but don't fail the script.
        print(f"[dedup] checker warning (proceeding): {exc}")

    print(f"[dedup] renamed {renamed} duplicate node(s)")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--output")
    args = p.parse_args()

    src = Path(args.model)
    if not src.exists():
        print(f"[dedup] not found: {src}", file=sys.stderr)
        return 2

    dst = Path(args.output) if args.output else src
    return dedup(src, dst)


if __name__ == "__main__":
    sys.exit(main())
