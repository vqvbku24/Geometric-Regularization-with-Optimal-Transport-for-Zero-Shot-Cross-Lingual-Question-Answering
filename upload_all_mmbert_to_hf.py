#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
upload_all_mmbert_to_hf.py

Đồng bộ toàn bộ checkpoints, logs và báo cáo kết quả của mmBERT pipeline
lên một thư mục chuyên biệt 'mmBERT/' trên Hugging Face Hub (vinhvo1205/Sinkhorn_2_stages).
"""

import os
import sys
import glob
import logging

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE_DIR, ".env"))
except ImportError:
    pass

logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO, datefmt="%H:%M:%S")
log = logging.getLogger("hf_sync_mmbert")


def get_token():
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


def sync_all(repo_id: str = "vinhvo1205/Sinkhorn_2_stages", prefix: str = "mmBERT"):
    token = get_token()
    if not token:
        log.error("❌ Không tìm thấy HF_TOKEN trong .env hoặc biến môi trường! Không thể upload lên HF.")
        return False

    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
    except ImportError:
        log.error("❌ Thư viện 'huggingface_hub' chưa được cài đặt!")
        return False

    log.info("=" * 70)
    log.info(f"  ĐỒNG BỘ TOÀN BỘ ASSETS mmBERT LÊN HUGGING FACE HUB")
    log.info(f"  Repository : {repo_id}")
    log.info(f"  Thư mục Hub: {prefix}/")
    log.info("=" * 70)

    success_count = 0
    fail_count = 0

    def upload_file_safe(local_path: str, repo_path: str):
        nonlocal success_count, fail_count
        if not os.path.exists(local_path):
            return
        sz_mb = os.path.getsize(local_path) / (1024 * 1024)
        log.info(f"  ⬆ Uploading: {local_path} ({sz_mb:.1f} MB) -> {repo_path}...")
        try:
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=repo_path,
                repo_id=repo_id,
                repo_type="model",
            )
            log.info(f"    ✓ Thành công: {repo_path}")
            success_count += 1
        except Exception as e:
            log.warning(f"    ⚠ Lỗi khi upload {repo_path}: {e}")
            fail_count += 1

    def upload_folder_safe(local_folder: str, repo_folder: str):
        nonlocal success_count, fail_count
        if not os.path.isdir(local_folder):
            return
        log.info(f"  📁 Uploading folder: {local_folder} -> {repo_folder}...")
        try:
            api.upload_folder(
                folder_path=local_folder,
                path_in_repo=repo_folder,
                repo_id=repo_id,
                repo_type="model",
            )
            log.info(f"    ✓ Folder thành công: {repo_folder}")
            success_count += 1
        except Exception as e:
            log.warning(f"    ⚠ Lỗi khi upload folder {repo_folder}: {e}")
            fail_count += 1

    # 1. Stage 1 Checkpoints (Best and all epochs)
    s1_dir = os.path.join(BASE_DIR, "checkpoints")
    if os.path.exists(s1_dir):
        s1_files = set(glob.glob(os.path.join(s1_dir, "*stage1*mmbert*.pt")) + glob.glob(os.path.join(s1_dir, "stage1_squad_epoch_*.pt")))
        for s1_file in sorted(s1_files):
            upload_file_safe(s1_file, f"{prefix}/checkpoints/{os.path.basename(s1_file)}")

    # 2. Stage 2 Checkpoints (VI, AR, HI)
    for lang, dir_name, best_name in [
        ("vi", "checkpoint_stage2_vi", "stage2_best.pt"),
        ("ar", "checkpoint_stage2_ar", "stage2_ar_best.pt"),
        ("hi", "checkpoint_stage2_hi", "stage2_hi_best.pt"),
    ]:
        # Search in mmbert_m5_ours or root of dir_name
        search_dirs = [
            os.path.join(BASE_DIR, dir_name, "mmbert_m5_ours"),
            os.path.join(BASE_DIR, dir_name),
        ]
        for s_dir in search_dirs:
            if not os.path.exists(s_dir):
                continue
            for pt_file in glob.glob(os.path.join(s_dir, "*.pt")):
                rel_fname = os.path.basename(pt_file)
                upload_file_safe(pt_file, f"{prefix}/{dir_name}/{rel_fname}")

            # Tensorboard
            for tb_cand in glob.glob(os.path.join(s_dir, "*tensorboard*")):
                if os.path.isdir(tb_cand):
                    upload_folder_safe(tb_cand, f"{prefix}/logs/{os.path.basename(tb_cand)}")

    # 3. Results Reports
    res_md = os.path.join(BASE_DIR, "results_mmbert_pipeline.md")
    res_json = os.path.join(BASE_DIR, "results_mmbert_pipeline.json")
    if os.path.exists(res_md):
        upload_file_safe(res_md, f"{prefix}/results/results_mmbert_pipeline.md")
        upload_file_safe(res_md, f"{prefix}/README.md")
    if os.path.exists(res_json):
        upload_file_safe(res_json, f"{prefix}/results/results_mmbert_pipeline.json")

    log.info("=" * 70)
    log.info(f"  HOÀN TẤT ĐỒNG BỘ mmBERT: {success_count} thành công, {fail_count} lỗi/bỏ qua")
    log.info(f"  Đường dẫn xem trên Hub: https://huggingface.co/{repo_id}/tree/main/{prefix}")
    log.info("=" * 70)
    return fail_count == 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, default="vinhvo1205/Sinkhorn_2_stages")
    parser.add_argument("--prefix", type=str, default="mmBERT")
    args = parser.parse_args()

    repo = os.environ.get("HF_REPO_ID", args.repo_id)
    sync_all(repo_id=repo, prefix=args.prefix)
