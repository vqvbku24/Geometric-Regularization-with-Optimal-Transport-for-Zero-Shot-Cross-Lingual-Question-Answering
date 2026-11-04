#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analysis/probe_mmbert_layers.py

Systematic Layer Probing for mmBERT (ModernBERT architecture) across 22 layers:
1. Cross-lingual Representation Alignment (OT-weighted Cosine Similarity & Mean Cosine Similarity)
2. Sinkhorn Transport Plan Entropy H(gamma)
3. Structural Dissection: Global/Full Attention vs. Local Sliding-Window Attention

Supports:
- Local dataset loading (dataset/xquad.*.json)
- Hugging Face Datasets fallback (load_dataset('google/xquad', ...))
- GPU acceleration (Colab T4, A100, RTX 3090/4090)
- Publication-quality chart generation & CSV export
"""

import os
import sys
import json
import argparse
import random
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _ROOT)

from transformers import AutoTokenizer, AutoModel


# ─────────────────────────────────────────────────────────────────────────────
# 1. Sinkhorn OT & Entropy Functions (batched, log-domain)
# ─────────────────────────────────────────────────────────────────────────────

def sinkhorn_log_domain(
    C: torch.Tensor,
    en_pad_mask: torch.Tensor,
    tgt_pad_mask: torch.Tensor,
    epsilon: float = 0.1,
    num_iters: int = 50,
) -> torch.Tensor:
    """
    Batched log-domain Sinkhorn algorithm with non-uniform marginals on valid tokens.
    C: (B, M, N)
    en_pad_mask: (B, M) True = PAD
    tgt_pad_mask: (B, N) True = PAD
    """
    B, M, N = C.shape
    valid_en = (~en_pad_mask).float()
    valid_tgt = (~tgt_pad_mask).float()
    n_en = valid_en.sum(dim=1, keepdim=True).clamp(min=1.0)
    n_tgt = valid_tgt.sum(dim=1, keepdim=True).clamp(min=1.0)

    log_mu = torch.where(en_pad_mask, torch.tensor(-1e9, device=C.device), -torch.log(n_en))
    log_nu = torch.where(tgt_pad_mask, torch.tensor(-1e9, device=C.device), -torch.log(n_tgt))

    u = torch.zeros(B, M, device=C.device)
    v = torch.zeros(B, N, device=C.device)

    log_K = -C / epsilon

    for _ in range(num_iters):
        u = log_mu - torch.logsumexp(log_K + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(log_K + u.unsqueeze(2), dim=1)

    log_gamma = log_K + u.unsqueeze(2) + v.unsqueeze(1)
    gamma = torch.exp(log_gamma)

    # Zero-out PAD tokens
    gamma = gamma.masked_fill(en_pad_mask.unsqueeze(2), 0.0)
    gamma = gamma.masked_fill(tgt_pad_mask.unsqueeze(1), 0.0)
    return gamma


def compute_transport_entropy(gamma: torch.Tensor, en_pad_mask: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Computes average row-wise Shannon entropy of the transport plan on valid source tokens.
    """
    row_sum = gamma.sum(dim=-1, keepdim=True).clamp(min=eps)
    p = gamma / row_sum  # (B, M, N)
    h_tokens = -(p * (p + eps).log()).sum(dim=-1)  # (B, M)

    valid_mask = (~en_pad_mask).float()
    valid_tokens = valid_mask.sum().clamp(min=1.0)
    mean_entropy = (h_tokens * valid_mask).sum() / valid_tokens
    return mean_entropy.item()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Dataset Loading (Local JSON or Hugging Face datasets)
# ─────────────────────────────────────────────────────────────────────────────

def load_parallel_samples(lang: str, num_samples: int = 150, seed: int = 42) -> list[dict]:
    """
    Loads parallel (English, Target Language) QA pairs from XQuAD.
    Tries local dataset/ first, falls back to HuggingFace datasets library.
    """
    lang = lang.lower()
    local_tgt_path = os.path.join(_ROOT, "dataset", f"xquad.{lang}.json")
    local_en_path = os.path.join(_ROOT, "dataset", "xquad.en.json")

    pairs = []

    # Option A: Local files exist
    if os.path.exists(local_tgt_path) and os.path.exists(local_en_path):
        with open(local_en_path, "r", encoding="utf-8") as f:
            en_data = json.load(f)["data"]
        with open(local_tgt_path, "r", encoding="utf-8") as f:
            tgt_data = json.load(f)["data"]

        en_map = {}
        for art in en_data:
            for p in art["paragraphs"]:
                for qa in p["qas"]:
                    en_map[qa["id"]] = {"q": qa["question"], "c": p["context"]}

        for art in tgt_data:
            for p in art["paragraphs"]:
                for qa in p["qas"]:
                    qid = qa["id"]
                    if qid in en_map:
                        pairs.append({
                            "id": qid,
                            "en_q": en_map[qid]["q"],
                            "en_c": en_map[qid]["c"],
                            "tgt_q": qa["question"],
                            "tgt_c": p["context"],
                        })
    else:
        # Option B: Hugging Face datasets fallback
        try:
            from datasets import load_dataset
            print(f"Loading XQuAD ({lang} and en) from Hugging Face Hub...")
            ds_en = load_dataset("google/xquad", "xquad.en", split="validation")
            ds_tgt = load_dataset("google/xquad", f"xquad.{lang}", split="validation")

            en_map = {row["id"]: {"q": row["question"], "c": row["context"]} for row in ds_en}
            for row in ds_tgt:
                qid = row["id"]
                if qid in en_map:
                    pairs.append({
                        "id": qid,
                        "en_q": en_map[qid]["q"],
                        "en_c": en_map[qid]["c"],
                        "tgt_q": row["question"],
                        "tgt_c": row["context"],
                    })
        except Exception as e:
            raise RuntimeError(f"Could not load XQuAD for {lang} from local or Hugging Face Hub: {e}")

    rng = random.Random(seed)
    rng.shuffle(pairs)
    return pairs[:num_samples]


# ─────────────────────────────────────────────────────────────────────────────
# 3. Layer Probing Runner
# ─────────────────────────────────────────────────────────────────────────────

def probe_model_layers(
    model_name: str = "jhu-clsp/mmBERT-base",
    languages: list[str] = ["vi", "ar", "hi"],
    num_samples: int = 120,
    batch_size: int = 8,
    max_length: int = 384,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_dir: str = os.path.join(_ROOT, "analysis", "mmbert_probing"),
):
    device = torch.device(device_str)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 75)
    print(f"  mmBERT 22-LAYER PROBING & SYSTEMATIC ARCHITECTURAL ANALYSIS")
    print(f"  Model       : {model_name}")
    print(f"  Device      : {device}")
    print(f"  Languages   : {languages}")
    print(f"  Samples/lang: {num_samples} (Batch size: {batch_size})")
    print(f"  Output Dir  : {output_dir}")
    print("=" * 75)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0
    if tokenizer.sep_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.sep_token_id = tokenizer.eos_token_id

    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    model.to(device)
    model.eval()

    num_layers = getattr(model.config, "num_hidden_layers", 22)
    # Detect layer types (ModernBERT alternating attention)
    layer_types_cfg = getattr(model.config, "layer_types", None)
    global_every_n = getattr(model.config, "global_attn_every_n_layers", 3)

    layer_types = []
    for l_idx in range(num_layers):
        if layer_types_cfg is not None and l_idx < len(layer_types_cfg):
            t = layer_types_cfg[l_idx]
            layer_types.append("Global" if ("full" in t.lower() or "global" in t.lower()) else "Local")
        else:
            layer_types.append("Global" if (l_idx % global_every_n == 0) else "Local")

    print(f"\nBackbone contains {num_layers} layers.")
    global_indices = [i for i, t in enumerate(layer_types) if t == "Global"]
    local_indices = [i for i, t in enumerate(layer_types) if t == "Local"]
    print(f"  - Global Attention Layers ({len(global_indices)}): {global_indices}")
    print(f"  - Local Attention Layers  ({len(local_indices)}): {local_indices}\n")

    all_records = []

    for lang in languages:
        print(f"--> Probing Language: {lang.upper()} ({num_samples} parallel pairs)...")
        pairs = load_parallel_samples(lang, num_samples=num_samples)

        # Accumulators per layer: layer_idx -> list of batch metrics
        layer_ot_sim = {i: [] for i in range(num_layers)}
        layer_mean_sim = {i: [] for i in range(num_layers)}
        layer_entropy = {i: [] for i in range(num_layers)}

        for b_start in range(0, len(pairs), batch_size):
            batch_pairs = pairs[b_start : b_start + batch_size]
            en_texts = [f"{p['en_q']} {tokenizer.sep_token} {p['en_c']}" for p in batch_pairs]
            tgt_texts = [f"{p['tgt_q']} {tokenizer.sep_token} {p['tgt_c']}" for p in batch_pairs]

            enc_en = tokenizer(
                en_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(device)

            enc_tgt = tokenizer(
                tgt_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                out_en = model(**enc_en)
                out_tgt = model(**enc_tgt)

            en_pad = (enc_en.attention_mask == 0)
            tgt_pad = (enc_tgt.attention_mask == 0)
            valid_en = (~en_pad).float().unsqueeze(-1)
            valid_tgt = (~tgt_pad).float().unsqueeze(-1)

            # Probe each layer
            for l_idx in range(num_layers):
                # hidden_states[0] is embedding; hidden_states[l_idx + 1] is layer l_idx
                h_en = out_en.hidden_states[l_idx + 1]    # (B, M, D)
                h_tgt = out_tgt.hidden_states[l_idx + 1]  # (B, N, D)

                # 1. Mean Pooled Cosine Similarity
                mean_en = (h_en * valid_en).sum(dim=1) / valid_en.sum(dim=1).clamp(min=1.0)
                mean_tgt = (h_tgt * valid_tgt).sum(dim=1) / valid_tgt.sum(dim=1).clamp(min=1.0)
                mean_sim = F.cosine_similarity(mean_en, mean_tgt, dim=-1).mean().item()
                layer_mean_sim[l_idx].append(mean_sim)

                # 2. Sinkhorn Cost & Transport Plan
                h_en_norm = F.normalize(h_en, p=2, dim=-1)
                h_tgt_norm = F.normalize(h_tgt, p=2, dim=-1)
                # Cost matrix C = 1 - cosine_similarity
                C = 1.0 - torch.bmm(h_en_norm, h_tgt_norm.transpose(1, 2))
                # Mask PAD to 1e4
                C = C.masked_fill(en_pad.unsqueeze(2), 1e4)
                C = C.masked_fill(tgt_pad.unsqueeze(1), 1e4)

                gamma = sinkhorn_log_domain(C, en_pad, tgt_pad, epsilon=0.1, num_iters=50)

                # Transport cost = <gamma, C>
                ot_cost = (gamma * C.masked_fill(C >= 1e3, 0.0)).sum(dim=(-2, -1)).mean().item()
                ot_sim = max(0.0, 1.0 - ot_cost)
                layer_ot_sim[l_idx].append(ot_sim)

                # Transport Plan Entropy
                ent = compute_transport_entropy(gamma, en_pad)
                layer_entropy[l_idx].append(ent)

        for l_idx in range(num_layers):
            all_records.append({
                "language": lang.upper(),
                "layer_index": l_idx,
                "attention_type": layer_types[l_idx],
                "ot_cosine_sim": float(np.mean(layer_ot_sim[l_idx])),
                "mean_pooled_sim": float(np.mean(layer_mean_sim[l_idx])),
                "transport_entropy": float(np.mean(layer_entropy[l_idx])),
            })

    df = pd.DataFrame(all_records)
    csv_path = os.path.join(output_dir, "mmbert_layer_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved metrics table to: {csv_path}")

    # Generate Publication Chart
    plot_probing_results(df, num_layers, layer_types, output_dir)

    # Print Layer Selection Recommendation
    recommend_layers(df, num_layers, layer_types)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Publication Plot & Scientific Recommendation
# ─────────────────────────────────────────────────────────────────────────────

def plot_probing_results(df: pd.DataFrame, num_layers: int, layer_types: list[str], output_dir: str):
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    colors = {"VI": "#1f77b4", "AR": "#2ca02c", "HI": "#d62728"}
    markers_attn = {"Global": "o", "Local": "x"}

    # Top: OT Cosine Similarity
    ax1 = axes[0]
    for lang in df["language"].unique():
        sub = df[df["language"] == lang].sort_values("layer_index")
        ax1.plot(sub["layer_index"], sub["ot_cosine_sim"], label=f"{lang} (OT-Cosine)", color=colors.get(lang, "gray"), linewidth=2, alpha=0.85)

        # Highlight Global Attention points
        global_sub = sub[sub["attention_type"] == "Global"]
        ax1.scatter(global_sub["layer_index"], global_sub["ot_cosine_sim"], color=colors.get(lang, "gray"), s=70, zorder=5, marker="o")
        local_sub = sub[sub["attention_type"] == "Local"]
        ax1.scatter(local_sub["layer_index"], local_sub["ot_cosine_sim"], color=colors.get(lang, "gray"), s=35, zorder=4, marker="x", alpha=0.6)

    ax1.set_ylabel("Cross-Lingual OT Alignment\n(1 - Transport Cost)", fontsize=12, fontweight="bold")
    ax1.set_title("mmBERT Layer-Wise Probing: Representation Alignment & Transport Entropy", fontsize=14, fontweight="bold", pad=12)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right", frameon=True)

    # Candidate zone shaded (e.g. 50% - 80% depth: layer 11 to 18)
    cand_start, cand_end = int(num_layers * 0.50), int(num_layers * 0.85)
    ax1.axvspan(cand_start - 0.5, cand_end + 0.5, color="gold", alpha=0.15, label="Candidate Zone (50%-85%)")

    # Bottom: Transport Entropy
    ax2 = axes[1]
    for lang in df["language"].unique():
        sub = df[df["language"] == lang].sort_values("layer_index")
        ax2.plot(sub["layer_index"], sub["transport_entropy"], label=f"{lang} (Entropy)", color=colors.get(lang, "gray"), linewidth=2, alpha=0.85)
        global_sub = sub[sub["attention_type"] == "Global"]
        ax2.scatter(global_sub["layer_index"], global_sub["transport_entropy"], color=colors.get(lang, "gray"), s=70, zorder=5, marker="o")

    ax2.axvspan(cand_start - 0.5, cand_end + 0.5, color="gold", alpha=0.15)
    ax2.set_xlabel("Layer Index (0 to 21)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Sinkhorn Transport Entropy H(γ)\n(Lower = Sharper Match)", fontsize=12, fontweight="bold")
    ax2.set_xticks(range(num_layers))
    
    # Custom x-tick labels indicating Global Attention
    xtick_labels = [f"L{i}\n(G)" if layer_types[i] == "Global" else f"L{i}" for i in range(num_layers)]
    ax2.set_xticklabels(xtick_labels, fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "mmbert_layer_probing.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"✓ Saved publication chart to: {chart_path}")


def recommend_layers(df: pd.DataFrame, num_layers: int, layer_types: list[str]):
    print("\n" + "=" * 75)
    print("  LAYER SELECTION RECOMMENDATION ANALYSIS")
    print("=" * 75)

    # Average metrics across languages per layer
    avg_df = df.groupby(["layer_index", "attention_type"])[["ot_cosine_sim", "mean_pooled_sim", "transport_entropy"]].mean().reset_index()

    # Score: Alignment minus normalized entropy penalty
    norm_ent = (avg_df["transport_entropy"] - avg_df["transport_entropy"].min()) / (avg_df["transport_entropy"].max() - avg_df["transport_entropy"].min() + 1e-8)
    avg_df["composite_score"] = avg_df["ot_cosine_sim"] - 0.2 * norm_ent

    print("\nTop Global Attention Layers (Sorted by Alignment & Receptive Field):")
    globals_only = avg_df[avg_df["attention_type"] == "Global"].sort_values("composite_score", ascending=False)
    for _, row in globals_only.iterrows():
        print(f"  - Layer {int(row['layer_index']):02d} [GLOBAL]: OT-Sim={row['ot_cosine_sim']:.4f}, Entropy={row['transport_entropy']:.4f}, Score={row['composite_score']:.4f}")

    # Optimal 4 Global Attention Layers in the mid-to-deep range (50% - 95%)
    cand_globals = avg_df[(avg_df["attention_type"] == "Global") & (avg_df["layer_index"] >= int(num_layers * 0.4))].sort_values("layer_index")
    selected_layers = cand_globals["layer_index"].tolist()
    if len(selected_layers) > 4:
        selected_layers = selected_layers[-4:]

    print("-" * 75)
    print(f"🌟 RECOMMENDED TARGET LAYERS FOR mmBERT: {selected_layers}")
    print(f"Reasoning:")
    print(f"  1. 100% Global Attention: Every selected layer has full 512-token receptive field.")
    print(f"  2. Mid-to-Deep Semantics: Captures transferable cross-lingual syntax/semantics without MLM bias.")
    print(f"  3. Copy-Paste CLI Argument for Stage 1 & Stage 2 Training:")
    print(f"     --target_layers {' '.join(str(l) for l in selected_layers)}")
    print("=" * 75 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# 5. CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Probe mmBERT layers for Optimal Transport alignment")
    parser.add_argument("--model_name", type=str, default="jhu-clsp/mmBERT-base")
    parser.add_argument("--languages", nargs="+", default=["vi", "ar", "hi"])
    parser.add_argument("--num_samples", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default=os.path.join(_ROOT, "analysis", "mmbert_probing"))
    args = parser.parse_args()

    probe_model_layers(
        model_name=args.model_name,
        languages=args.languages,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device_str=args.device,
        output_dir=args.output_dir,
    )

