#!/usr/bin/env python3
"""
crosscheck_ort.py -- prove MiniOnnx agrees with the real ONNX ecosystem.     (optional)

    pip install onnx onnxruntime numpy
    python python/crosscheck_ort.py models

It does three things:
  1. builds two extra models with the OFFICIAL onnx Python library (ort_linear.onnx,
     ort_mlp.onnx) -- files written by real protobuf, read by MiniOnnx's hand-written reader;
  2. runs onnx.checker on EVERY .onnx in the folder -- including the ones MiniOnnx's
     writer produced (tools/make_models);
  3. runs every model in onnxruntime on a deterministic input and writes <model>.ref
     (input + expected output). tests/test_main.cpp group 14 and
     `minionnx_run model.onnx --check model.ref` compare MiniOnnx against these files.
"""
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort


def pattern(shape, phase=0.0):
    n = int(np.prod(shape))
    return np.sin(0.37 * np.arange(n, dtype=np.float32) + np.float32(phase)).astype(np.float32).reshape(shape)


def build_official_models(folder):
    rng = np.random.default_rng(42)

    # ort_linear: what torch.onnx.export produces for Linear -> ReLU -> Linear -> Sigmoid
    W1 = rng.standard_normal((12, 6)).astype(np.float32)   # [out, in] -> transB=1
    b1 = rng.standard_normal((12,)).astype(np.float32)
    W2 = rng.standard_normal((12, 3)).astype(np.float32)   # [in, out] -> MatMul
    b2 = rng.standard_normal((1, 3)).astype(np.float32)
    nodes = [
        helper.make_node("Gemm", ["x", "W1", "b1"], ["h"], name="fc1", transB=1),
        helper.make_node("Relu", ["h"], ["r"], name="relu"),
        helper.make_node("MatMul", ["r", "W2"], ["z"], name="fc2"),
        helper.make_node("Add", ["z", "b2"], ["a"], name="bias2"),
        helper.make_node("Sigmoid", ["a"], ["y"], name="sig"),
    ]
    g = helper.make_graph(
        nodes, "ort_linear",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 6])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", 3])],
        initializer=[numpy_helper.from_array(W1, "W1"), numpy_helper.from_array(b1, "b1"),
                     numpy_helper.from_array(W2, "W2"), numpy_helper.from_array(b2, "b2")])
    m = helper.make_model(g, producer_name="onnx-helper", opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    onnx.save(m, os.path.join(folder, "ort_linear.onnx"))

    # ort_mlp: Gemm with alpha/beta, Tanh, Mul by a scalar, Softmax, Identity
    W = rng.standard_normal((6, 10)).astype(np.float32)
    C = rng.standard_normal((1, 10)).astype(np.float32)
    s = np.array([0.5], dtype=np.float32)
    nodes = [
        helper.make_node("Gemm", ["x", "W", "C"], ["g"], name="gemm", alpha=0.75, beta=2.0),
        helper.make_node("Tanh", ["g"], ["t"], name="tanh"),
        helper.make_node("Mul", ["t", "s"], ["u"], name="scale"),
        helper.make_node("Identity", ["u"], ["v"], name="id"),
        helper.make_node("Softmax", ["v"], ["p"], name="softmax", axis=-1),
    ]
    g = helper.make_graph(
        nodes, "ort_mlp",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 6])],
        [helper.make_tensor_value_info("p", TensorProto.FLOAT, ["batch", 10])],
        initializer=[numpy_helper.from_array(W, "W"), numpy_helper.from_array(C, "C"), numpy_helper.from_array(s, "s")])
    m = helper.make_model(g, producer_name="onnx-helper", opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    onnx.save(m, os.path.join(folder, "ort_mlp.onnx"))


def write_ref(path, inputs, outputs):
    with open(path, "w") as f:
        for kind, d in (("input", inputs), ("output", outputs)):
            for name, arr in d.items():
                f.write(f"{kind} {name} {arr.ndim} {' '.join(str(s) for s in arr.shape)}\n")
                f.write(" ".join(repr(float(v)) for v in arr.ravel()) + "\n")


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "models"
    build_official_models(folder)
    for fn in sorted(os.listdir(folder)):
        if not fn.endswith(".onnx"):
            continue
        path = os.path.join(folder, fn)
        onnx.checker.check_model(path)                  # validates MiniOnnx's WRITER too
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        feeds = {}
        for i, inp in enumerate(sess.get_inputs()):
            shape = [d if isinstance(d, int) else 3 for d in inp.shape]
            feeds[inp.name] = pattern(shape, 0.1 * i)
        outs = sess.run(None, feeds)
        outputs = {o.name: v for o, v in zip(sess.get_outputs(), outs)}
        write_ref(path[:-5] + ".ref", feeds, outputs)
        print(f"{fn:28s} onnx.checker OK, onnxruntime {ort.__version__} -> {fn[:-5]}.ref")


if __name__ == "__main__":
    main()
