#!/usr/bin/env python3
"""
Q2.4.5.4i Step 3 — element-wise comparison of native vs CPU-reference
block-0 outputs.

Compares:
  $NATIVE_DIR/24_img_resid2.f32  vs  $CPU_DIR/cpu_24_img_resid2.f32
  $NATIVE_DIR/24_txt_resid2.f32  vs  $CPU_DIR/cpu_24_txt_resid2.f32

Both files are raw F32 binaries of shape [seq, H=3072].

Reports:
  - Per-row (per-token) cosine similarity: min, mean, p10, p50, p90, max
  - Per-channel max-absolute magnitude (top-32 channels by max-abs in each)
  - Mean / max absolute magnitude (whole tensor)
  - NaN / Inf counts
  - Direction: agree on sign per element fraction
  - Element-wise relative error histogram

Verdict:
  GREEN  : per-row mean cosine sim > 0.99   AND  ratio of magnitudes < 5x
           → native algorithm is correct; magnitudes are model-design
  YELLOW : per-row mean cosine sim > 0.5    BUT  magnitudes diverge > 5x
           → likely precision-class issue (F16 saturation, BF16 widening
             insufficient)
  RED    : per-row mean cosine sim <= 0.5
           → algorithm bug somewhere in modulation/attn/ffn
"""

import argparse
import os
import sys

import numpy as np


def load_f32(path: str, expected_n: int) -> np.ndarray:
    sz = os.path.getsize(path)
    if sz != expected_n * 4:
        raise ValueError(
            f"{path}: size {sz} != expected {expected_n}*4 = {expected_n * 4}"
        )
    arr = np.fromfile(path, dtype=np.float32, count=expected_n)
    if arr.size != expected_n:
        raise ValueError(f"{path}: short read {arr.size}/{expected_n}")
    return arr


def safe_cosine(a: np.ndarray, b: np.ndarray, axis: int) -> np.ndarray:
    na = np.linalg.norm(a, axis=axis)
    nb = np.linalg.norm(b, axis=axis)
    denom = na * nb
    out = np.where(denom > 0, np.sum(a * b, axis=axis) / np.maximum(denom, 1e-30), 0.0)
    return out


def report_pair(label: str, native: np.ndarray, cpu: np.ndarray, H: int) -> dict:
    seq = native.size // H
    nat = native.reshape(seq, H)
    ref = cpu.reshape(seq, H)

    # Replace NaN/Inf with 0 for stat computation, but track them separately.
    nat_nan = int(np.isnan(nat).sum())
    nat_inf = int(np.isinf(nat).sum())
    ref_nan = int(np.isnan(ref).sum())
    ref_inf = int(np.isinf(ref).sum())

    nat_clean = np.nan_to_num(nat, nan=0.0, posinf=0.0, neginf=0.0)
    ref_clean = np.nan_to_num(ref, nan=0.0, posinf=0.0, neginf=0.0)

    nat_abs = np.abs(nat_clean)
    ref_abs = np.abs(ref_clean)

    # Per-row cosine similarity (along H).
    cos_per_row = safe_cosine(nat_clean.astype(np.float64),
                                ref_clean.astype(np.float64), axis=1)
    cos_min = float(np.min(cos_per_row))
    cos_mean = float(np.mean(cos_per_row))
    cos_p10 = float(np.quantile(cos_per_row, 0.1))
    cos_p50 = float(np.quantile(cos_per_row, 0.5))
    cos_p90 = float(np.quantile(cos_per_row, 0.9))
    cos_max = float(np.max(cos_per_row))

    # Per-channel max-abs (compare the most volatile channels).
    nat_chan_max = np.max(nat_abs, axis=0)  # [H]
    ref_chan_max = np.max(ref_abs, axis=0)
    top_native_chans = np.argsort(-nat_chan_max)[:16]
    top_ref_chans = np.argsort(-ref_chan_max)[:16]

    # Sign agreement.
    sign_agree = float(np.mean(np.sign(nat_clean) == np.sign(ref_clean)))

    # Whole-tensor magnitudes.
    nat_mag_mean = float(np.mean(nat_abs))
    nat_mag_max = float(np.max(nat_abs))
    ref_mag_mean = float(np.mean(ref_abs))
    ref_mag_max = float(np.max(ref_abs))
    mag_ratio_mean = nat_mag_mean / (ref_mag_mean + 1e-30)
    mag_ratio_max = nat_mag_max / (ref_mag_max + 1e-30)

    # Element-wise relative error (clipped to denom > 1e-3 to avoid noise).
    denom = np.maximum(np.abs(ref_clean), 1e-3)
    rel_err = np.abs(nat_clean - ref_clean) / denom
    rel_err_p50 = float(np.quantile(rel_err, 0.5))
    rel_err_p90 = float(np.quantile(rel_err, 0.9))
    rel_err_p99 = float(np.quantile(rel_err, 0.99))

    print(f"\n=== {label}  shape=[seq={seq}, H={H}] ===")
    print(f"NaN  native={nat_nan}  cpu={ref_nan}     "
          f"Inf  native={nat_inf}  cpu={ref_inf}")
    print(f"native: mean_abs={nat_mag_mean:.4g}  max_abs={nat_mag_max:.4g}")
    print(f"cpu   : mean_abs={ref_mag_mean:.4g}  max_abs={ref_mag_max:.4g}")
    print(f"ratio : mean_abs (native/cpu)={mag_ratio_mean:.4g}  "
          f"max_abs={mag_ratio_max:.4g}")
    print(f"per-row cosine similarity: "
          f"min={cos_min:.4f}  p10={cos_p10:.4f}  p50={cos_p50:.4f}  "
          f"p90={cos_p90:.4f}  mean={cos_mean:.4f}  max={cos_max:.4f}")
    print(f"sign-agree fraction: {sign_agree:.4f}")
    print(f"rel-err quantiles: p50={rel_err_p50:.4g}  p90={rel_err_p90:.4g}  "
          f"p99={rel_err_p99:.4g}")
    print(f"top native max-abs channels: {top_native_chans.tolist()}")
    print(f"  native chan max-abs: "
          f"{[float(nat_chan_max[c]) for c in top_native_chans[:8]]}")
    print(f"  cpu    chan max-abs: "
          f"{[float(ref_chan_max[c]) for c in top_native_chans[:8]]}")
    print(f"top cpu    max-abs channels: {top_ref_chans.tolist()}")

    return {
        "cos_mean": cos_mean,
        "cos_min": cos_min,
        "cos_p10": cos_p10,
        "mag_ratio_mean": mag_ratio_mean,
        "mag_ratio_max": mag_ratio_max,
        "nat_max_abs": nat_mag_max,
        "ref_max_abs": ref_mag_max,
        "sign_agree": sign_agree,
        "rel_err_p90": rel_err_p90,
        "nat_nan": nat_nan,
        "nat_inf": nat_inf,
        "ref_nan": ref_nan,
        "ref_inf": ref_inf,
    }


def verdict(stats: dict) -> str:
    cos = stats["cos_mean"]
    ratio_max = stats["mag_ratio_max"]
    nat_finite = stats["nat_nan"] + stats["nat_inf"]
    ref_finite = stats["ref_nan"] + stats["ref_inf"]
    if nat_finite > 0 or ref_finite > 0:
        return f"YELLOW (native NaN/Inf={nat_finite}, cpu NaN/Inf={ref_finite})"
    if cos > 0.99 and ratio_max < 5.0:
        return "GREEN — native algorithm correct, magnitudes match"
    if cos > 0.99 and ratio_max >= 5.0:
        return ("YELLOW — directionally correct but magnitude blow-up "
                f"(native/cpu max ratio = {ratio_max:.2g}); precision class")
    if cos > 0.5:
        return ("YELLOW — partial direction agreement (cos_mean=%.3f) — "
                "examine intermediate substeps") % cos
    return f"RED — algorithm error (cos_mean={cos:.3f}, ratio_max={ratio_max:.2g})"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--native-dir", default="/tmp/qie_block0_inputs",
                    help="dir with native-engine dumps "
                         "(includes 24_img_resid2.f32 + 24_txt_resid2.f32)")
    p.add_argument("--cpu-dir", default="/tmp/qie_block0_outputs",
                    help="dir with CPU-reference dumps "
                         "(cpu_24_img_resid2.f32 + cpu_24_txt_resid2.f32)")
    p.add_argument("--img-seq", type=int,
                    default=int(os.environ.get("QIE_Q45_IMG_SEQ", "64")))
    p.add_argument("--txt-seq", type=int,
                    default=int(os.environ.get("QIE_Q45_TXT_SEQ", "32")))
    p.add_argument("--hidden", type=int, default=3072)
    args = p.parse_args()

    if os.environ.get("QIE_Q45_BIG", "0") not in ("0", ""):
        # match the smoke harness defaults
        args.img_seq = 256
        args.txt_seq = 64

    img_n = args.img_seq * args.hidden
    txt_n = args.txt_seq * args.hidden

    print(f"comparing native vs CPU reference, img_seq={args.img_seq} "
          f"txt_seq={args.txt_seq} H={args.hidden}")
    print(f"  native_dir={args.native_dir}")
    print(f"  cpu_dir={args.cpu_dir}")

    nat_img = load_f32(os.path.join(args.native_dir, "24_img_resid2.f32"), img_n)
    cpu_img = load_f32(os.path.join(args.cpu_dir,    "cpu_24_img_resid2.f32"), img_n)
    img_stats = report_pair("img_resid2", nat_img, cpu_img, args.hidden)

    nat_txt = load_f32(os.path.join(args.native_dir, "24_txt_resid2.f32"), txt_n)
    cpu_txt = load_f32(os.path.join(args.cpu_dir,    "cpu_24_txt_resid2.f32"), txt_n)
    txt_stats = report_pair("txt_resid2", nat_txt, cpu_txt, args.hidden)

    print("\n" + "=" * 60)
    print("FINAL VERDICT (img_resid2):", verdict(img_stats))
    print("FINAL VERDICT (txt_resid2):", verdict(txt_stats))
    print("=" * 60)

    # Exit code: 0 if both GREEN, 1 if any YELLOW (mag), 2 if any RED.
    bad = max(
        (0 if "GREEN" in verdict(s) else 1 if "YELLOW" in verdict(s) else 2)
        for s in (img_stats, txt_stats)
    )
    sys.exit(bad)


if __name__ == "__main__":
    main()
