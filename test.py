"""
Verify L-infinity robustness of the trained FashionMNIST MLP with Marabou.

Workflow
--------
1. Load the ONNX model exported by `train.py`.
2. Pick one image from the FashionMNIST test set and obtain its
   predicted class c.
3. For a sweep of perturbation radii eps, ask Marabou whether there
   exists an input x' in the L-infinity ball of radius eps around the
   image such that some other class i != c gets a logit at least as
   large as class c.
       - SAT     -> adversarial example exists, model is NOT robust.
       - UNSAT   -> verified robust at this eps.
       - TIMEOUT -> verifier could not decide within the budget.
4. Report SAT/UNSAT and runtime per (eps, target) pair.

Marabou's `addInequality([y_i, y_c], [1, -1], -delta)` encodes
    1 * y_i - 1 * y_c <= -delta
i.e. y_c >= y_i + delta. We negate this and instead add
    -y_c + y_i >= 0   (encoded as y_c - y_i <= 0)
to assert "class i beats class c" -- if Marabou finds SAT, that input
is an adversarial example targeting class i.
"""

import os
import sys
import time
import json
import argparse
import pathlib

import numpy as np
import torch
from torchvision import datasets, transforms

# Marabou's Python bindings live next to the repo root. Ensure they are
# importable without requiring `pip install`.
MARABOU_ROOT = os.environ.get(
    "MARABOU_ROOT",
    str(pathlib.Path(__file__).resolve().parent.parent / "Marabou"),
)
if MARABOU_ROOT not in sys.path:
    sys.path.insert(0, MARABOU_ROOT)

from maraboupy import Marabou, MarabouCore  # noqa: E402


ONNX_PATH = "./models/fashion_mlp.onnx"
DATA_DIR = "./data"
NUM_CLASSES = 10

FASHION_LABELS = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


class _Tee:
    """Mirror writes to multiple streams (used to also write stdout to log.txt)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def load_test_image(index: int):
    """Return (flattened image in [0,1], integer label) for test[index]."""
    test_set = datasets.FashionMNIST(
        DATA_DIR, train=False, download=True,
        transform=transforms.ToTensor())
    x, y = test_set[index]
    return x.numpy().reshape(-1), int(y)


def predict_class(onnx_path: str, image_flat: np.ndarray) -> int:
    """Run the ONNX model on `image_flat` (784,) via Marabou's evaluator."""
    net = Marabou.read_onnx(onnx_path)
    # MarabouNetworkONNX.evaluate expects the input shape (1,1,28,28).
    out = net.evaluateWithoutMarabou([image_flat.reshape(1, 1, 28, 28)])
    return int(np.argmax(out[0]))


def build_query(onnx_path: str, image_flat: np.ndarray, epsilon: float,
                orig_class: int, target_class: int):
    """
    Build a fresh MarabouNetworkONNX with:
      - input bounds: per-pixel [x_i - eps, x_i + eps] clipped to [0,1]
      - output constraint: y_target - y_orig >= 0  (target wins)

    Returns the network; caller invokes net.solve(...) on it.
    """
    net = Marabou.read_onnx(onnx_path)
    in_vars = net.inputVars[0].flatten()       # 784 input variable IDs
    out_vars = net.outputVars[0].flatten()     # 10 output variable IDs

    # Per-pixel L-infinity ball, clipped to the valid pixel range.
    for i, v in enumerate(in_vars):
        lb = max(0.0, float(image_flat[i]) - epsilon)
        ub = min(1.0, float(image_flat[i]) + epsilon)
        net.setLowerBound(int(v), lb)
        net.setUpperBound(int(v), ub)

    # y_orig - y_target <= 0   <=>   y_target >= y_orig.
    # If Marabou returns SAT, the corresponding input is an adversarial
    # example that flips the prediction from orig_class to target_class.
    net.addInequality(
        [int(out_vars[orig_class]), int(out_vars[target_class])],
        [1.0, -1.0],
        0.0,
    )
    return net, in_vars, out_vars


def solve(net, timeout_sec: int):
    opts = Marabou.createOptions(timeoutInSeconds=timeout_sec, verbosity=0)
    t0 = time.time()
    exit_code, vals, _stats = net.solve(verbose=False, options=opts)
    elapsed = time.time() - t0
    return exit_code, vals, elapsed


def verify_robustness(onnx_path, image_flat, orig_class, epsilon,
                      timeout_sec=60):
    """
    Returns dict with overall verdict for this image at this eps.
        verdict: 'robust' | 'not_robust' | 'unknown'
        details: per-target results
    """
    per_target = []
    not_robust = False
    unknown = False
    for target in range(NUM_CLASSES):
        if target == orig_class:
            continue
        net, in_vars, out_vars = build_query(
            onnx_path, image_flat, epsilon, orig_class, target)
        code, vals, elapsed = solve(net, timeout_sec)
        per_target.append({
            "target": target,
            "exit_code": code,
            "time_sec": elapsed,
            "n_assignments": len(vals),
        })
        # Print as we go so the user sees progress.
        print(f"    target={target:2d} ({FASHION_LABELS[target]:>11s}) "
              f"-> {code:<7s} in {elapsed:6.2f}s")
        if code == "sat":
            not_robust = True
            # Save the adversarial example for inspection.
            adv = np.array([vals[int(v)] for v in in_vars], dtype=np.float32)
            return {
                "verdict": "not_robust",
                "first_sat_target": target,
                "adv_example": adv,
                "details": per_target,
            }
        if code in ("TIMEOUT", "UNKNOWN", "ERROR"):
            unknown = True
    if not_robust:
        return {"verdict": "not_robust", "details": per_target}
    if unknown:
        return {"verdict": "unknown", "details": per_target}
    return {"verdict": "robust", "details": per_target}


def save_adversarial(out_dir: pathlib.Path, eps: float,
                     image_flat: np.ndarray, adv_flat: np.ndarray,
                     orig_class: int, target_class: int):
    """Persist the adversarial input as .npy and a 3-panel .png visualisation.

    Returns dict of file paths (relative to out_dir's parent) for the summary.
    """
    tag = f"eps{eps:.4f}".rstrip("0").rstrip(".")
    npy_path = out_dir / f"adv_{tag}.npy"
    png_path = out_dir / f"adv_{tag}.png"

    np.save(npy_path, adv_flat)

    # Lazy import: matplotlib is only needed when a counterexample is found.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    orig_img = image_flat.reshape(28, 28)
    adv_img = adv_flat.reshape(28, 28)
    diff = adv_img - orig_img
    # Amplify for display: scale so the largest |diff| maps to 1.
    max_abs = max(float(np.max(np.abs(diff))), 1e-8)
    diff_amp = diff / max_abs

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.4))
    axes[0].imshow(orig_img, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"original\nclass = {orig_class} "
                      f"({FASHION_LABELS[orig_class]})")
    axes[1].imshow(adv_img, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"adversarial\nflips to {target_class} "
                      f"({FASHION_LABELS[target_class]})")
    im = axes[2].imshow(diff_amp, cmap="seismic", vmin=-1, vmax=1)
    axes[2].set_title(f"diff (scaled x{1.0/max_abs:.1f})\n"
                      f"max|delta| = {max_abs:.4f}, eps = {eps}")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    return {"adv_npy": str(npy_path), "adv_png": str(png_path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", default=ONNX_PATH)
    parser.add_argument("--index", type=int, default=0,
                        help="Index of the FashionMNIST test image to verify")
    parser.add_argument("--epsilons", type=float, nargs="+",
                        default=[0.001, 0.005, 0.01, 0.02])
    parser.add_argument("--timeout", type=int, default=60,
                        help="Per-target Marabou timeout (seconds)")
    parser.add_argument("--output-dir", default="results",
                        help="Directory for summary.json, log.txt, "
                             "adversarial .npy / .png")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.txt"
    log_file = open(log_path, "w")
    sys.stdout = _Tee(sys.__stdout__, log_file)

    image, label = load_test_image(args.index)
    pred = predict_class(args.onnx, image)
    print(f"image idx={args.index}  true={label} ({FASHION_LABELS[label]})  "
          f"pred={pred} ({FASHION_LABELS[pred]})")
    if pred != label:
        print("WARNING: model misclassifies this image; verification still "
              "operates on the predicted class.")

    print("\nL-infinity robustness sweep:")
    summary = []
    per_eps_records = []
    for eps in args.epsilons:
        print(f"\neps = {eps}")
        result = verify_robustness(
            args.onnx, image, pred, eps, timeout_sec=args.timeout)
        total_time = sum(d["time_sec"] for d in result["details"])
        summary.append((eps, result["verdict"], total_time))
        record = {
            "epsilon": eps,
            "verdict": result["verdict"],
            "total_time_sec": total_time,
            "details": result["details"],
        }
        print(f"  verdict: {result['verdict']}   "
              f"total time: {total_time:.2f}s")
        if result["verdict"] == "not_robust":
            tgt = result.get("first_sat_target")
            print(f"  -> adversarial flips prediction to "
                  f"{tgt} ({FASHION_LABELS[tgt]})")
            adv = result.get("adv_example")
            if adv is not None and tgt is not None:
                paths = save_adversarial(
                    out_dir, eps, image, adv, pred, tgt)
                record["first_sat_target"] = tgt
                record.update(paths)
                print(f"  -> saved {paths['adv_npy']} and {paths['adv_png']}")
        per_eps_records.append(record)

    print("\n=== summary ===")
    print(f"{'epsilon':>8s}  {'verdict':>10s}  {'total_time(s)':>14s}")
    for eps, verdict, t in summary:
        print(f"{eps:>8.4f}  {verdict:>10s}  {t:>14.2f}")

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "onnx": args.onnx,
            "image_index": args.index,
            "true_label": label,
            "predicted_class": pred,
            "timeout_sec": args.timeout,
            "results": per_eps_records,
        }, f, indent=2)
    print(f"\nwrote {summary_path} and {log_path}")

    sys.stdout = sys.__stdout__
    log_file.close()


if __name__ == "__main__":
    main()
