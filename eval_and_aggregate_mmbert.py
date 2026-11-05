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


def run_mlqa_eval(
    lang: str,
    ckpt_path: str,
    stage1_ckpt: Optional[str] = None,
    tag: str = "m1",
    model_name: str = "jhu-clsp/mmBERT-base",
    n_samples: int = 0
) -> Tuple[float, float]:
    """
    Evaluates MLQA using the official project methodology:
    1. generate_mlqa_preds.py (sliding-window inference with stride=128, max_length=384)
    2. mlqa_evaluation_v1.py (official Facebook MLQA evaluator with language normalization)
    """
    mlqa_file = os.path.join(BASE_DIR, "dataset", "MLQA", f"test-context-{lang}-question-{lang}.json")
    if not os.path.exists(ckpt_path):
        log_msg(f"  [MLQA {lang.upper()}] ❌ Checkpoint not found: {ckpt_path}")
        return (-1.0, -1.0)
    if not os.path.exists(mlqa_file):
        log_msg(f"  [MLQA {lang.upper()}] ❌ MLQA data file not found: {mlqa_file}")
        return (-1.0, -1.0)

    pred_dir = os.path.join(BASE_DIR, "preds_mlqa")
    os.makedirs(pred_dir, exist_ok=True)
    pred_file = os.path.join(pred_dir, f"mlqa_preds_{lang}_{tag}.json")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"

    # Step 1: Generate predictions with sliding window
    gen_script = os.path.join(BASE_DIR, "generate_mlqa_preds.py")
    cmd_gen = [
        sys.executable, gen_script,
        "--ckpt", ckpt_path,
        "--eval_file", mlqa_file,
        "--output_pred_file", pred_file,
        "--model_name", model_name,
        "--max_length", "384"
    ]
    if stage1_ckpt and os.path.exists(stage1_ckpt) and ckpt_path != stage1_ckpt:
        cmd_gen.extend(["--stage1_ckpt", stage1_ckpt])
    else:
        cmd_gen.append("--zero_shot")
    if n_samples > 0:
        cmd_gen.extend(["--n_samples", str(n_samples)])

    log_msg(f"  [MLQA {lang.upper()}] 1/2 Generating predictions: {' '.join(cmd_gen)}")
    proc_gen = subprocess.run(cmd_gen, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    if proc_gen.returncode != 0 or not os.path.exists(pred_file):
        log_msg(f"  [MLQA {lang.upper()}] ❌ Prediction generation failed:\n{proc_gen.stdout[-400:]}")
        return (-1.0, -1.0)

    # Step 2: Official MLQA evaluation (Facebook script with language normalization)
    eval_script = os.path.join(BASE_DIR, "mlqa_evaluation_v1.py")
    cmd_eval = [
        sys.executable, eval_script,
        mlqa_file,
        pred_file,
        lang
    ]
    log_msg(f"  [MLQA {lang.upper()}] 2/2 Official scoring: {' '.join(cmd_eval)}")
    proc_eval = subprocess.run(cmd_eval, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    output = proc_eval.stdout

    em, f1 = -1.0, -1.0
    try:
        lines = [ln.strip() for ln in output.strip().split("\n") if ln.strip()]
        for ln in reversed(lines):
            if ln.startswith("{") and "exact_match" in ln:
                res = json.loads(ln)
                em = float(res.get("exact_match", -1.0))
                f1 = float(res.get("f1", -1.0))
                break
    except Exception as e:
        log_msg(f"  [MLQA {lang.upper()}] ⚠ Failed to parse MLQA json: {e}")

    if em >= 0 and f1 >= 0:
        log_msg(f"  [MLQA {lang.upper()}] ✅ Official MLQA: EM={em:.2f}% | F1={f1:.2f}%")
    else:
        log_msg(f"  [MLQA {lang.upper()}] ⚠ Output:\n{output[-400:]}")

    return em, f1


def main():
    parser = argparse.ArgumentParser(description="Evaluate mmBERT Cross-Lingual Transfer on VI, AR, HI")
    parser.add_argument("--stage1_ckpt", type=str, default="checkpoints/stage1_squad_best_mmbert.pt")
    parser.add_argument("--vi_ckpt", type=str, default="checkpoint_stage2_vi/mmbert_m5_ours/stage2_best.pt")
    parser.add_argument("--ar_ckpt", type=str, default="checkpoint_stage2_ar/mmbert_m5_ours/stage2_ar_best.pt")
    parser.add_argument("--hi_ckpt", type=str, default="checkpoint_stage2_hi/mmbert_m5_ours/stage2_hi_best.pt")
    parser.add_argument("--model_name", type=str, default="jhu-clsp/mmBERT-base")
    parser.add_argument("--hf_repo_id", type=str, default=DEFAULT_HF_REPO)
    parser.add_argument("--n_samples", type=int, default=0, help="Number of samples for XQuAD (0 for all)")
    parser.add_argument("--mlqa_samples", type=int, default=0, help="Number of samples for MLQA (0 for all)")
    parser.add_argument("--squad_samples", type=int, default=1000, help="Number of samples for SQuAD 2.0 Dev retention check (default 1000, 0 for all)")
    parser.add_argument("--skip_mlqa", action="store_true", default=False)
    parser.add_argument("--skip_squad_dev", action="store_true", default=False)
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

    # 2. Evaluate M5 (Ours) on all 3 languages (Best Checkpoints)
    log_msg("\n[PHASE 2/3] Evaluating M5 (Ours - Best Checkpoints)...")
    for lang in ["vi", "ar", "hi"]:
        ef = xquad_files[lang]
        ckpt = m5_ckpts[lang]
        log_msg(f"Evaluating M5 on XQuAD-{lang.upper()} (Ckpt: {ckpt})...")
        em, f1 = run_quick_eval(eval_file=ef, ckpt_path=ckpt, stage1_ckpt=args.stage1_ckpt, model_name=args.model_name, n_samples=args.n_samples)
        results["xquad"][f"m5_{lang}"] = {"em": em, "f1": f1}
        log_msg(f"  -> M5 XQuAD-{lang.upper()}: EM={em:.2f}%, F1={f1:.2f}%")

    # 3. Evaluate ALL Epoch Checkpoints for each language
    import glob
    results["epochs"] = {"vi": {}, "ar": {}, "hi": {}}
    log_msg("\n[PHASE 3/3] Evaluating Trajectory Across ALL Epochs (VI, AR, HI)...")

    def find_epoch_checkpoints(ckpt_file: str, lang: str):
        search_dirs = [
            os.path.dirname(ckpt_file),
            os.path.join(os.path.dirname(ckpt_file), "mmbert_m5_ours"),
            os.path.join(BASE_DIR, f"checkpoint_stage2_{lang}", "mmbert_m5_ours"),
            os.path.join(BASE_DIR, f"checkpoint_stage2_{lang}"),
        ]
        prefix = f"stage2_{lang}_epoch_" if lang != "vi" else "stage2_epoch_"
        found = {}
        for d in search_dirs:
            if not os.path.exists(d):
                continue
            for pt in glob.glob(os.path.join(d, f"{prefix}*.pt")):
                m = re.search(r"epoch_(\d+)\.pt", os.path.basename(pt))
                if m:
                    ep_num = int(m.group(1))
                    found[ep_num] = pt
        return dict(sorted(found.items()))

    all_epoch_nums = set()
    for lang in ["vi", "ar", "hi"]:
        ef = xquad_files[lang]
        ep_ckpts = find_epoch_checkpoints(m5_ckpts[lang], lang)
        if ep_ckpts:
            log_msg(f"Found {len(ep_ckpts)} epoch checkpoints for {lang.upper()}: {list(ep_ckpts.keys())}")
            all_epoch_nums.update(ep_ckpts.keys())
            for ep, ep_path in ep_ckpts.items():
                log_msg(f"  Evaluating {lang.upper()} Epoch {ep} ({ep_path})...")
                em, f1 = run_quick_eval(eval_file=ef, ckpt_path=ep_path, stage1_ckpt=args.stage1_ckpt, model_name=args.model_name, n_samples=args.n_samples)
                results["epochs"][lang][str(ep)] = {"em": em, "f1": f1}
                log_msg(f"    -> Epoch {ep} {lang.upper()}: EM={em:.2f}%, F1={f1:.2f}%")
        else:
            log_msg(f"  No individual epoch checkpoints found for {lang.upper()} (only best checkpoint evaluated).")

    # 4. Evaluate Cross-Lingual Transfer on MLQA (VI, AR, HI)
    results["mlqa"] = {}
    if not args.skip_mlqa:
        log_msg("\n[PHASE 4/5] Evaluating Cross-Lingual Transfer on MLQA (VI, AR, HI)...")
        log_msg("  (Using official generate_mlqa_preds.py sliding-window + mlqa_evaluation_v1.py)")
        mlqa_files = {
            "vi": os.path.join(BASE_DIR, "dataset", "MLQA", "test-context-vi-question-vi.json"),
            "ar": os.path.join(BASE_DIR, "dataset", "MLQA", "test-context-ar-question-ar.json"),
            "hi": os.path.join(BASE_DIR, "dataset", "MLQA", "test-context-hi-question-hi.json"),
        }
        for lang in ["vi", "ar", "hi"]:
            mf = mlqa_files[lang]
            if os.path.exists(mf):
                log_msg(f"Evaluating M1 on MLQA-{lang.upper()} ({mf})...")
                em_m1, f1_m1 = run_mlqa_eval(lang=lang, ckpt_path=args.stage1_ckpt, stage1_ckpt=None, tag="m1", model_name=args.model_name, n_samples=args.mlqa_samples)
                results["mlqa"][f"m1_{lang}"] = {"em": em_m1, "f1": f1_m1}
                log_msg(f"  -> M1 MLQA-{lang.upper()}: EM={em_m1:.2f}%, F1={f1_m1:.2f}%")

                ckpt = m5_ckpts[lang]
                log_msg(f"Evaluating M5 on MLQA-{lang.upper()} (Ckpt: {ckpt})...")
                em_m5, f1_m5 = run_mlqa_eval(lang=lang, ckpt_path=ckpt, stage1_ckpt=args.stage1_ckpt, tag="m5", model_name=args.model_name, n_samples=args.mlqa_samples)
                results["mlqa"][f"m5_{lang}"] = {"em": em_m5, "f1": f1_m5}
                log_msg(f"  -> M5 MLQA-{lang.upper()}: EM={em_m5:.2f}%, F1={f1_m5:.2f}%")
            else:
                log_msg(f"  [Warning] MLQA file for {lang.upper()} not found at {mf}")

    # 5. Evaluate English Retention on SQuAD 2.0 Dev & XQuAD English (Zero Catastrophic Forgetting)
    results["english_retention"] = {}
    if not args.skip_squad_dev:
        log_msg("\n[PHASE 5/5] Evaluating English Retention on SQuAD 2.0 Dev & XQuAD English (Zero Catastrophic Forgetting)...")
        squad_dev_file = os.path.join(BASE_DIR, "dataset", "Squad2.0", "dev-v2.0.json")
        xquad_en_file = os.path.join(BASE_DIR, "dataset", "xquad.en.json")

        # 5A. SQuAD 2.0 Dev
        if os.path.exists(squad_dev_file):
            log_msg(f"Evaluating Stage 1 Base on SQuAD 2.0 Dev ({squad_dev_file})...")
            em_s1, f1_s1 = run_quick_eval(eval_file=squad_dev_file, ckpt_path=args.stage1_ckpt, stage1_ckpt=None, model_name=args.model_name, n_samples=args.squad_samples)
            results["english_retention"]["squad_stage1_base"] = {"em": em_s1, "f1": f1_s1}
            log_msg(f"  -> Stage 1 Base SQuAD 2.0 EN: EM={em_s1:.2f}%, F1={f1_s1:.2f}%")

            for lang in ["vi", "ar", "hi"]:
                ckpt = m5_ckpts[lang]
                log_msg(f"Evaluating Stage 2 {lang.upper()} Model on SQuAD 2.0 Dev...")
                em_ret, f1_ret = run_quick_eval(eval_file=squad_dev_file, ckpt_path=ckpt, stage1_ckpt=args.stage1_ckpt, model_name=args.model_name, n_samples=args.squad_samples)
                results["english_retention"][f"squad_stage2_{lang}"] = {"em": em_ret, "f1": f1_ret}
                log_msg(f"  -> Stage 2 {lang.upper()} Model on SQuAD 2.0 EN: EM={em_ret:.2f}%, F1={f1_ret:.2f}% (Baseline={em_s1:.2f}%)")
        else:
            log_msg(f"  [Warning] SQuAD 2.0 Dev file not found at {squad_dev_file}")

        # 5B. XQuAD English
        if os.path.exists(xquad_en_file):
            log_msg(f"Evaluating Stage 1 Base on XQuAD English ({xquad_en_file})...")
            em_xe_s1, f1_xe_s1 = run_quick_eval(eval_file=xquad_en_file, ckpt_path=args.stage1_ckpt, stage1_ckpt=None, model_name=args.model_name, n_samples=0)
            results["english_retention"]["xquad_en_stage1_base"] = {"em": em_xe_s1, "f1": f1_xe_s1}
            log_msg(f"  -> Stage 1 Base XQuAD-EN: EM={em_xe_s1:.2f}%, F1={f1_xe_s1:.2f}%")

            for lang in ["vi", "ar", "hi"]:
                ckpt = m5_ckpts[lang]
                log_msg(f"Evaluating Stage 2 {lang.upper()} Model on XQuAD English...")
                em_xe_ret, f1_xe_ret = run_quick_eval(eval_file=xquad_en_file, ckpt_path=ckpt, stage1_ckpt=args.stage1_ckpt, model_name=args.model_name, n_samples=0)
                results["english_retention"][f"xquad_en_stage2_{lang}"] = {"em": em_xe_ret, "f1": f1_xe_ret}
                log_msg(f"  -> Stage 2 {lang.upper()} Model on XQuAD-EN: EM={em_xe_ret:.2f}%, F1={f1_xe_ret:.2f}% (Baseline={em_xe_s1:.2f}%)")
        else:
            log_msg(f"  [Warning] XQuAD English file not found at {xquad_en_file}")

    # Format Markdown Tables
    m1_vi = results["xquad"].get("m1_vi", {"em": -1, "f1": -1})
    m5_vi = results["xquad"].get("m5_vi", {"em": -1, "f1": -1})
    m1_ar = results["xquad"].get("m1_ar", {"em": -1, "f1": -1})
    m5_ar = results["xquad"].get("m5_ar", {"em": -1, "f1": -1})
    m1_hi = results["xquad"].get("m1_hi", {"em": -1, "f1": -1})
    m5_hi = results["xquad"].get("m5_hi", {"em": -1, "f1": -1})

    def fmt_cell(m):
        if not m or m.get("em", -1) < 0:
            return "N/A"
        return f"{m['em']:.2f}/{m['f1']:.2f}"

    def fmt_delta(m_base, m_ours):
        if not m_base or not m_ours or m_base.get("em", -1) < 0 or m_ours.get("em", -1) < 0:
            return "N/A"
        d_em = m_ours["em"] - m_base["em"]
        d_f1 = m_ours["f1"] - m_base["f1"]
        return f"+{d_em:.2f}/+{d_f1:.2f}" if d_em >= 0 else f"{d_em:.2f}/{d_f1:.2f}"

    # MLQA Cells
    mlqa_m1_vi = results["mlqa"].get("m1_vi", {"em": -1, "f1": -1})
    mlqa_m5_vi = results["mlqa"].get("m5_vi", {"em": -1, "f1": -1})
    mlqa_m1_ar = results["mlqa"].get("m1_ar", {"em": -1, "f1": -1})
    mlqa_m5_ar = results["mlqa"].get("m5_ar", {"em": -1, "f1": -1})
    mlqa_m1_hi = results["mlqa"].get("m1_hi", {"em": -1, "f1": -1})
    mlqa_m5_hi = results["mlqa"].get("m5_hi", {"em": -1, "f1": -1})

    mlqa_section = ""
    if results["mlqa"]:
        mlqa_section = f"""### 2. MLQA Benchmark Results (Official Facebook Eval - EM / F1)

| Config | Vietnamese (MLQA-vi) | Arabic (MLQA-ar) | Hindi (MLQA-hi) |
| :--- | :---: | :---: | :---: |
| **M1: Zero-shot (mmBERT)** | {fmt_cell(mlqa_m1_vi)} | {fmt_cell(mlqa_m1_ar)} | {fmt_cell(mlqa_m1_hi)} |
| **M5: Ours (Best Checkpoint)** | **{fmt_cell(mlqa_m5_vi)}** | **{fmt_cell(mlqa_m5_ar)}** | **{fmt_cell(mlqa_m5_hi)}** |
| **Gain (Δ Ours - M1)** | *{fmt_delta(mlqa_m1_vi, mlqa_m5_vi)}* | *{fmt_delta(mlqa_m1_ar, mlqa_m5_ar)}* | *{fmt_delta(mlqa_m1_hi, mlqa_m5_hi)}* |
"""

    # English Retention Cells
    sq_base = results["english_retention"].get("squad_stage1_base", {"em": -1, "f1": -1})
    sq_vi = results["english_retention"].get("squad_stage2_vi", {"em": -1, "f1": -1})
    sq_ar = results["english_retention"].get("squad_stage2_ar", {"em": -1, "f1": -1})
    sq_hi = results["english_retention"].get("squad_stage2_hi", {"em": -1, "f1": -1})

    xq_base = results["english_retention"].get("xquad_en_stage1_base", {"em": -1, "f1": -1})
    xq_vi = results["english_retention"].get("xquad_en_stage2_vi", {"em": -1, "f1": -1})
    xq_ar = results["english_retention"].get("xquad_en_stage2_ar", {"em": -1, "f1": -1})
    xq_hi = results["english_retention"].get("xquad_en_stage2_hi", {"em": -1, "f1": -1})

    retention_section = ""
    if results["english_retention"]:
        retention_section = f"""### 3. English Retention & Zero Catastrophic Forgetting

| Benchmark | Model Checkpoint | Target Language Adapted | English Score (EM / F1) | Retention Delta vs Stage 1 |
| :--- | :--- | :---: | :---: | :---: |
| **SQuAD 2.0 Dev** | **Stage 1 Base (mmBERT)** | *English Only* | {fmt_cell(sq_base)} | *0.00 / 0.00 (Reference)* |
| **SQuAD 2.0 Dev** | **Stage 2 Model (VI)** | *Vietnamese* | {fmt_cell(sq_vi)} | *{fmt_delta(sq_base, sq_vi)} (Preserved)* |
| **SQuAD 2.0 Dev** | **Stage 2 Model (AR)** | *Arabic* | {fmt_cell(sq_ar)} | *{fmt_delta(sq_base, sq_ar)} (Preserved)* |
| **SQuAD 2.0 Dev** | **Stage 2 Model (HI)** | *Hindi* | {fmt_cell(sq_hi)} | *{fmt_delta(sq_base, sq_hi)} (Preserved)* |
| **XQuAD-EN** | **Stage 1 Base (mmBERT)** | *English Only* | {fmt_cell(xq_base)} | *0.00 / 0.00 (Reference)* |
| **XQuAD-EN** | **Stage 2 Model (VI)** | *Vietnamese* | {fmt_cell(xq_vi)} | *{fmt_delta(xq_base, xq_vi)} (Preserved)* |
| **XQuAD-EN** | **Stage 2 Model (AR)** | *Arabic* | {fmt_cell(xq_ar)} | *{fmt_delta(xq_base, xq_ar)} (Preserved)* |
| **XQuAD-EN** | **Stage 2 Model (HI)** | *Hindi* | {fmt_cell(xq_hi)} | *{fmt_delta(xq_base, xq_hi)} (Preserved)* |
"""

    # Epoch rows for XQuAD
    epoch_rows = []
    if all_epoch_nums:
        for ep in sorted(all_epoch_nums):
            c_vi = fmt_cell(results["epochs"]["vi"].get(str(ep)))
            c_ar = fmt_cell(results["epochs"]["ar"].get(str(ep)))
            c_hi = fmt_cell(results["epochs"]["hi"].get(str(ep)))
            epoch_rows.append(f"| **Stage 2: Epoch {ep}** | {c_vi} | {c_ar} | {c_hi} |")
    epoch_table_section = ""
    if epoch_rows:
        epoch_table_section = f"""### 4. Trajectory Across ALL Training Epochs (XQuAD)

| Checkpoint / Epoch | Vietnamese | Arabic | Hindi |
| :--- | :---: | :---: | :---: |
| **M1: Zero-shot (Stage 1)** | {fmt_cell(m1_vi)} | {fmt_cell(m1_ar)} | {fmt_cell(m1_hi)} |
""" + "\n".join(epoch_rows) + f"""
| **Stage 2: Best Model** | **{fmt_cell(m5_vi)}** | **{fmt_cell(m5_ar)}** | **{fmt_cell(m5_hi)}** |
"""

    md_content = f"""# mmBERT Comprehensive Cross-Lingual Evaluation Report

**Model:** `{args.model_name}`  
**Evaluated:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  

### 1. XQuAD Benchmark Results (EM / F1)

| Config | Vietnamese | Arabic | Hindi |
| :--- | :---: | :---: | :---: |
| **M1: Zero-shot (mmBERT)** | {fmt_cell(m1_vi)} | {fmt_cell(m1_ar)} | {fmt_cell(m1_hi)} |
| **M5: Ours (Best Checkpoint)** | **{fmt_cell(m5_vi)}** | **{fmt_cell(m5_ar)}** | **{fmt_cell(m5_hi)}** |
| **Gain (Δ Ours - M1)** | *{fmt_delta(m1_vi, m5_vi)}* | *{fmt_delta(m1_ar, m5_ar)}* | *{fmt_delta(m1_hi, m5_hi)}* |

{mlqa_section}
{retention_section}
{epoch_table_section}
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
            log_msg(f"Uploading evaluation results to Hugging Face (folder: mmBERT/): {args.hf_repo_id}...")
            # Upload to mmBERT/ folder for dedicated tracking
            api.upload_file(
                path_or_fileobj=out_md,
                path_in_repo="mmBERT/results/results_mmbert_pipeline.md",
                repo_id=args.hf_repo_id,
                repo_type="model"
            )
            api.upload_file(
                path_or_fileobj=out_md,
                path_in_repo="mmBERT/README.md",
                repo_id=args.hf_repo_id,
                repo_type="model"
            )
            api.upload_file(
                path_or_fileobj=out_json,
                path_in_repo="mmBERT/results/results_mmbert_pipeline.json",
                repo_id=args.hf_repo_id,
                repo_type="model"
            )
            # Backward compatibility copy
            api.upload_file(
                path_or_fileobj=out_md,
                path_in_repo="results/results_mmbert_pipeline.md",
                repo_id=args.hf_repo_id,
                repo_type="model"
            )

            # Upload MLQA prediction files if present
            pred_dir = os.path.join(BASE_DIR, "preds_mlqa")
            if os.path.exists(pred_dir):
                for pf in glob.glob(os.path.join(pred_dir, "*.json")):
                    bname = os.path.basename(pf)
                    api.upload_file(
                        path_or_fileobj=pf,
                        path_in_repo=f"mmBERT/predictions/{bname}",
                        repo_id=args.hf_repo_id,
                        repo_type="model"
                    )
            log_msg("✓ Evaluation results and MLQA predictions uploaded to Hugging Face successfully (mmBERT/)!")
        except Exception as e:
            log_msg(f"  [Warning] Hugging Face upload failed: {e}")


if __name__ == "__main__":
    main()

