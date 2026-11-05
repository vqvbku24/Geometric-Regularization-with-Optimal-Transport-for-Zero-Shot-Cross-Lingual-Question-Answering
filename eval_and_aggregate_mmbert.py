#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_and_aggregate_mmbert.py

Comprehensive Evaluation & Reporting for mmBERT Cross-Lingual Pipeline:
1. Evaluates M1 Zero-Shot Baseline (Stage 1 mmBERT SQuAD checkpoint) on XQuAD (VI, AR, HI).
2. Evaluates M5 Ours (Stage 2 Teacher-Student OT checkpoints) on XQuAD (VI, AR, HI).
3. Evaluates on MLQA (VI, AR, HI) if datasets exist.
4. Generates Markdown & JSON results comparing M1 vs M5 on mmBERT.
5. Uploads results to Hugging Face Hub if HF repo is configured.
"""

import os
import sys
import json
import re
import argparse
import subprocess
import datetime
from typing import Dict, Tuple, Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE_DIR, ".env"))
except ImportError:
    pass

DEFAULT_HF_REPO = "vinhvo1205/Sinkhorn_2_stages"


def log_msg(msg: str):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def get_hf_token() -> Optional[str]:
    token = os.environ.get("HF_TOKEN")
    if not token:
        tf = os.path.join(BASE_DIR, ".hf_token")
        if os.path.exists(tf):
            try:
                with open(tf, "r", encoding="utf-8") as f:
                    token = f.read().strip()
            except Exception:
                pass
    return token


def run_quick_eval(eval_file: str, ckpt_path: str, stage1_ckpt: Optional[str] = None, model_name: str = "jhu-clsp/mmBERT-base", n_samples: int = 0) -> Tuple[float, float]:
    """
    Runs phase4-evaluation/quick_eval_mmbert.py as a subprocess.
    Returns (EM, F1) as floats.
    """
    if not os.path.exists(eval_file):
        log_msg(f"  [Warning] Eval file not found: {eval_file}")
        return (-1.0, -1.0)
    if not os.path.exists(ckpt_path):
        log_msg(f"  [Warning] Checkpoint not found: {ckpt_path}")
        return (-1.0, -1.0)

    script_path = os.path.join(BASE_DIR, "phase4-evaluation", "quick_eval_mmbert.py")
    cmd = [
        sys.executable,
        script_path,
        "--ckpt", ckpt_path,
        "--eval_file", eval_file,
        "--model_name", model_name,
        "--n_samples", str(n_samples),
    ]
    if stage1_ckpt and os.path.exists(stage1_ckpt) and ckpt_path != stage1_ckpt:
        cmd.extend(["--stage1_ckpt", stage1_ckpt])

    log_msg(f"  [Eval] Running: {' '.join(cmd)}")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    output = proc.stdout

    em, f1 = -1.0, -1.0
    em_match = re.search(r"Exact Match \(EM\):\s*([\d\.]+)%", output)
    f1_match = re.search(r"F1 Score:\s*([\d\.]+)%", output)
    if em_match and f1_match:
        em = float(em_match.group(1))
        f1 = float(f1_match.group(1))
    else:
        log_msg(f"  [Warning] Could not parse EM/F1 from eval output:\n{output[-500:]}")

    return em, f1


def main():
    parser = argparse.ArgumentParser(description="Evaluate mmBERT Cross-Lingual Transfer on VI, AR, HI")
    parser.add_argument("--stage1_ckpt", type=str, default="checkpoints/stage1_squad_best_mmbert.pt")
    parser.add_argument("--vi_ckpt", type=str, default="checkpoint_stage2_vi/mmbert_m5_ours/stage2_best.pt")
    parser.add_argument("--ar_ckpt", type=str, default="checkpoint_stage2_ar/mmbert_m5_ours/stage2_ar_best.pt")
    parser.add_argument("--hi_ckpt", type=str, default="checkpoint_stage2_hi/mmbert_m5_ours/stage2_hi_best.pt")
    parser.add_argument("--model_name", type=str, default="jhu-clsp/mmBERT-base")
    parser.add_argument("--hf_repo_id", type=str, default=DEFAULT_HF_REPO)
    parser.add_argument("--n_samples", type=int, default=0, help="Number of samples to evaluate (0 for all)")
    parser.add_argument("--skip_mlqa", action="store_true", default=False)
    args = parser.parse_args()

    log_msg("=" * 70)
    log_msg("  UNIFIED mmBERT PIPELINE EVALUATION (VI, AR, HI)")
    log_msg(f"  Model       : {args.model_name}")
    log_msg(f"  Stage 1 Ckpt: {args.stage1_ckpt}")
    log_msg(f"  VI Ckpt     : {args.vi_ckpt}")
    log_msg(f"  AR Ckpt     : {args.ar_ckpt}")
    log_msg(f"  HI Ckpt     : {args.hi_ckpt}")
    log_msg("=" * 70)

    # ── Dataset paths ─────────────────────────────────────────────
    xquad_files = {
        "vi": os.path.join(BASE_DIR, "dataset", "xquad.vi.json"),
        "ar": os.path.join(BASE_DIR, "dataset", "xquad.ar.json"),
        "hi": os.path.join(BASE_DIR, "dataset", "xquad.hi.json"),
    }
    
    # Checkpoints per language
    m5_ckpts = {
        "vi": args.vi_ckpt,
        "ar": args.ar_ckpt,
        "hi": args.hi_ckpt,
    }

    results = {
        "model_name": args.model_name,
        "timestamp": datetime.datetime.now().isoformat(),
        "xquad": {},
    }

    # 1. Evaluate M1 (Zero-shot) on all 3 languages
    log_msg("\n[PHASE 1/2] Evaluating M1 (Zero-Shot mmBERT Baseline)...")
    for lang in ["vi", "ar", "hi"]:
        ef = xquad_files[lang]
        log_msg(f"Evaluating M1 on XQuAD-{lang.upper()}...")
        em, f1 = run_quick_eval(eval_file=ef, ckpt_path=args.stage1_ckpt, stage1_ckpt=None, model_name=args.model_name, n_samples=args.n_samples)
        results["xquad"][f"m1_{lang}"] = {"em": em, "f1": f1}
        log_msg(f"  -> M1 XQuAD-{lang.upper()}: EM={em:.2f}%, F1={f1:.2f}%")

    # 2. Evaluate M5 (Ours) on all 3 languages
    log_msg("\n[PHASE 2/2] Evaluating M5 (Ours - Full Coordinated Stage 2)...")
    for lang in ["vi", "ar", "hi"]:
        ef = xquad_files[lang]
        ckpt = m5_ckpts[lang]
        log_msg(f"Evaluating M5 on XQuAD-{lang.upper()} (Ckpt: {ckpt})...")
        em, f1 = run_quick_eval(eval_file=ef, ckpt_path=ckpt, stage1_ckpt=args.stage1_ckpt, model_name=args.model_name, n_samples=args.n_samples)
        results["xquad"][f"m5_{lang}"] = {"em": em, "f1": f1}
        log_msg(f"  -> M5 XQuAD-{lang.upper()}: EM={em:.2f}%, F1={f1:.2f}%")

    # Format Markdown Table
    m1_vi = results["xquad"].get("m1_vi", {"em": -1, "f1": -1})
    m5_vi = results["xquad"].get("m5_vi", {"em": -1, "f1": -1})
    m1_ar = results["xquad"].get("m1_ar", {"em": -1, "f1": -1})
    m5_ar = results["xquad"].get("m5_ar", {"em": -1, "f1": -1})
    m1_hi = results["xquad"].get("m1_hi", {"em": -1, "f1": -1})
    m5_hi = results["xquad"].get("m5_hi", {"em": -1, "f1": -1})

    def fmt_cell(m):
        if m["em"] < 0:
            return "N/A"
        return f"{m['em']:.2f}/{m['f1']:.2f}"

    def fmt_delta(m_base, m_ours):
        if m_base["em"] < 0 or m_ours["em"] < 0:
            return "N/A"
        d_em = m_ours["em"] - m_base["em"]
        d_f1 = m_ours["f1"] - m_base["f1"]
        return f"+{d_em:.2f}/+{d_f1:.2f}" if d_em >= 0 else f"{d_em:.2f}/{d_f1:.2f}"

    md_content = f"""# mmBERT Cross-Lingual Transfer Results (XQuAD EM/F1)

**Model:** `{args.model_name}`  
**Evaluated:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  

| Config | Vietnamese | Arabic | Hindi |
| :--- | :---: | :---: | :---: |
| **M1: Zero-shot (mmBERT)** | {fmt_cell(m1_vi)} | {fmt_cell(m1_ar)} | {fmt_cell(m1_hi)} |
| **M5: Ours (Full OT)** | **{fmt_cell(m5_vi)}** | **{fmt_cell(m5_ar)}** | **{fmt_cell(m5_hi)}** |
| **Gain (Δ Ours - M1)** | *{fmt_delta(m1_vi, m5_vi)}* | *{fmt_delta(m1_ar, m5_ar)}* | *{fmt_delta(m1_hi, m5_hi)}* |

---
*Generated automatically by eval_and_aggregate_mmbert.py*
"""

    print("\n" + "=" * 70)
    print("  FINAL mmBERT RESULTS SUMMARY")
    print("=" * 70)
    print(md_content)
    print("=" * 70)

    # Save to files
    out_md = os.path.join(BASE_DIR, "results_mmbert_pipeline.md")
    out_json = os.path.join(BASE_DIR, "results_mmbert_pipeline.json")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md_content)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log_msg(f"Saved results to: {out_md} and {out_json}")

    # Upload to HF if configured
    hf_token = get_hf_token()
    if args.hf_repo_id and hf_token:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=hf_token)
            log_msg(f"Uploading evaluation results to Hugging Face: {args.hf_repo_id}...")
            api.upload_file(
                path_or_fileobj=out_md,
                path_in_repo="results/results_mmbert_pipeline.md",
                repo_id=args.hf_repo_id,
                repo_type="model"
            )
            api.upload_file(
                path_or_fileobj=out_json,
                path_in_repo="results/results_mmbert_pipeline.json",
                repo_id=args.hf_repo_id,
                repo_type="model"
            )
            log_msg("✓ Evaluation results uploaded to Hugging Face successfully!")
        except Exception as e:
            log_msg(f"  [Warning] Hugging Face upload failed: {e}")


if __name__ == "__main__":
    main()

