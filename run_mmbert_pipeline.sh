#!/usr/bin/env bash
# ==============================================================================
# run_mmbert_pipeline.sh
#
# FULL END-TO-END PIPELINE FOR mmBERT (ModernBERT architecture, 22 layers):
# 1. Multi-GPU auto-detection & DDP orchestration via torchrun
# 2. Stage 1: Fine-tune mmBERT on SQuAD (Batch Size 32, Target Layers: 10 13 16 19)
# 3. Stage 2 VI: Teacher-Student Sinkhorn OT for Vietnamese (M5 Ours)
# 4. Stage 2 AR: Teacher-Student Sinkhorn OT for Arabic (M5 Ours)
# 5. Stage 2 HI: Teacher-Student Sinkhorn OT for Hindi (M5 Ours)
# 6. Evaluation & Reporting: XQuAD evaluation across VI, AR, HI & auto HF sync
# ==============================================================================

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# Load token from .env or .hf_token
if [ -f "$DIR/.env" ]; then
    set -a
    source "$DIR/.env" 2>/dev/null || true
    set +a
fi
if [ -f "$DIR/.hf_token" ]; then
    export HF_TOKEN=$(cat "$DIR/.hf_token" | tr -d '\r\n')
fi

# ── Cấu hình mặc định ────────────────────────────────────────────────────────
MODEL_NAME="jhu-clsp/mmBERT-base"
TARGET_LAYERS="10 13 16 19"
BATCH_SIZE=32
STAGE1_EPOCHS=6
STAGE2_EPOCHS=6
LR_STAGE1=2e-5
SEED=42
HF_REPO="${HF_REPO_ID:-vinhvo1205/Sinkhorn_2_stages}"

# Flags
SKIP_STAGE1=0
EVAL_ONLY=0
FORCE_TRAIN=0
SKIP_VI=0
SKIP_AR=0
SKIP_HI=0

# Parse CLI arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --stage1_epochs)
            STAGE1_EPOCHS="$2"
            shift 2
            ;;
        --stage2_epochs)
            STAGE2_EPOCHS="$2"
            shift 2
            ;;
        --target_layers)
            # Custom target layers passed as string e.g. "10 13 16 19"
            TARGET_LAYERS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --hf_repo_id)
            HF_REPO="$2"
            shift 2
            ;;
        --skip_stage1)
            SKIP_STAGE1=1
            shift
            ;;
        --eval_only)
            EVAL_ONLY=1
            shift
            ;;
        --force_train)
            FORCE_TRAIN=1
            shift
            ;;
        --skip_vi)
            SKIP_VI=1
            shift
            ;;
        --skip_ar)
            SKIP_AR=1
            shift
            ;;
        --skip_hi)
            SKIP_HI=1
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            shift
            ;;
    esac
done

echo "======================================================================"
echo "  mmBERT FULL END-TO-END PIPELINE (STAGE 1 + STAGE 2 VI, AR, HI)"
echo "  Backbone Model:  $MODEL_NAME"
echo "  Target Layers:   $TARGET_LAYERS (Global Attention Sweet Spot)"
echo "  Batch Size:      $BATCH_SIZE"
echo "  Stage 1 Epochs:  $STAGE1_EPOCHS (lr=$LR_STAGE1)"
echo "  Stage 2 Epochs:  $STAGE2_EPOCHS"
echo "  Random Seed:     $SEED"
echo "  HuggingFace Hub: $HF_REPO"
echo "  Eval Only Mode:  $EVAL_ONLY"
echo "  Date & Time:     $(date)"
echo "======================================================================"

# ───────────────────────────────────────────────────────────────────
# BƯỚC 1: Kiểm tra & Tải Prerequisites
# ───────────────────────────────────────────────────────────────────
echo ""
echo "[BƯỚC 1/6] Kiểm tra các file tiền đề cần thiết..."
if [ -f "download_prerequisites.py" ]; then
    python download_prerequisites.py || true
fi

# ───────────────────────────────────────────────────────────────────
# BƯỚC 2: Nhận diện phần cứng GPU cho Multi-GPU DDP
# ───────────────────────────────────────────────────────────────────
echo ""
echo "[BƯỚC 2/6] Nhận diện cấu hình GPU khả dụng cho torchrun DDP..."
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)" 2>/dev/null || echo 1)

if [ "$NUM_GPUS" -lt 1 ]; then
    echo "  ⚠ Không tìm thấy GPU, chạy ở chế độ CPU (nproc=1)..."
    NUM_GPUS=1
    TORCHRUN_CMD="torchrun --nproc_per_node=1"
else
    echo "  ✓ Phát hiện $NUM_GPUS GPU khả dụng!"
    echo "  ✓ Khởi tạo torchrun với --nproc_per_node=$NUM_GPUS"
    TORCHRUN_CMD="torchrun --nproc_per_node=$NUM_GPUS"
fi

STAGE1_CKPT="checkpoints/stage1_squad_best_mmbert.pt"
VI_OUT_DIR="checkpoint_stage2_vi/mmbert_m5_ours"
AR_OUT_DIR="checkpoint_stage2_ar/mmbert_m5_ours"
HI_OUT_DIR="checkpoint_stage2_hi/mmbert_m5_ours"

VI_BEST_CKPT="$VI_OUT_DIR/stage2_best.pt"
AR_BEST_CKPT="$AR_OUT_DIR/stage2_ar_best.pt"
HI_BEST_CKPT="$HI_OUT_DIR/stage2_hi_best.pt"

# ───────────────────────────────────────────────────────────────────
# BƯỚC 3: Huấn luyện Stage 1 trên SQuAD (mmBERT Backbone)
# ───────────────────────────────────────────────────────────────────
if [ "$EVAL_ONLY" -eq 0 ]; then
    echo ""
    echo "[BƯỚC 3/6] Huấn luyện Stage 1 trên SQuAD v2.0 (mmBERT-base)..."
    
    if [ "$FORCE_TRAIN" -eq 0 ] && [ "$SKIP_STAGE1" -eq 0 ] && [ -f "$STAGE1_CKPT" ]; then
        echo "  ✓ Đã tìm thấy Stage 1 mmBERT checkpoint: $STAGE1_CKPT (Bỏ qua training)"
    elif [ "$SKIP_STAGE1" -eq 1 ]; then
        echo "  ℹ Cờ --skip_stage1 được bật: Bỏ qua Stage 1."
    else
        echo "  ▶ Khởi động Stage 1 training với DDP (port 29500)..."
        $TORCHRUN_CMD --master_port=29500 \
            train_stage1.py \
            --model_name "$MODEL_NAME" \
            --target_layers $TARGET_LAYERS \
            --checkpoint_name "stage1_squad_best_mmbert.pt" \
            --batch_size "$BATCH_SIZE" \
            --epochs "$STAGE1_EPOCHS" \
            --lr "$LR_STAGE1" \
            --seed "$SEED" \
            --hf_repo_id "$HF_REPO"
        echo "  ✓ Hoàn tất Stage 1 mmBERT!"
    fi
fi

if [ ! -f "$STAGE1_CKPT" ]; then
    echo "❌ LỖI: Không tìm thấy Stage 1 checkpoint tại $STAGE1_CKPT"
    exit 1
fi

# ───────────────────────────────────────────────────────────────────
# BƯỚC 4: Huấn luyện Stage 2 cho từng ngôn ngữ (VI, AR, HI)
# ───────────────────────────────────────────────────────────────────
if [ "$EVAL_ONLY" -eq 0 ]; then
    echo ""
    echo "[BƯỚC 4/6] Huấn luyện Stage 2 (Sinkhorn OT + Span + Margin, M5 static schedule)..."

    # --- 4a. Vietnamese (VI) ---
    if [ "$SKIP_VI" -eq 1 ]; then
        echo "  ℹ Bỏ qua Vietnamese theo yêu cầu (--skip_vi)."
    elif [ "$FORCE_TRAIN" -eq 0 ] && [ -f "$VI_BEST_CKPT" ]; then
        echo "  ✓ Đã có checkpoint Stage 2 VI: $VI_BEST_CKPT (Bỏ qua training)"
    else
        echo ""
        echo "  [4a/3] Bắt đầu huấn luyện Stage 2 Vietnamese (port 29600)..."
        $TORCHRUN_CMD --master_port=29600 \
            train_stage2.py \
            --stage1_ckpt "$STAGE1_CKPT" \
            --model_name "$MODEL_NAME" \
            --target_layers $TARGET_LAYERS \
            --output_dir "$VI_OUT_DIR" \
            --lambda_ot 0.5 \
            --lambda_span 1.0 \
            --lambda_margin 1.0 \
            --lambda_reg 50.0 \
            --lambda_kd 0.0 \
            --batch_size "$BATCH_SIZE" \
            --max_epochs "$STAGE2_EPOCHS" \
            --seed "$SEED" \
            --hf_repo_id "$HF_REPO"
        echo "  ✓ Hoàn tất huấn luyện Stage 2 Vietnamese!"
    fi

    # --- 4b. Arabic (AR) ---
    if [ "$SKIP_AR" -eq 1 ]; then
        echo "  ℹ Bỏ qua Arabic theo yêu cầu (--skip_ar)."
    elif [ "$FORCE_TRAIN" -eq 0 ] && [ -f "$AR_BEST_CKPT" ]; then
        echo "  ✓ Đã có checkpoint Stage 2 AR: $AR_BEST_CKPT (Bỏ qua training)"
    else
        echo ""
        echo "  [4b/3] Bắt đầu huấn luyện Stage 2 Arabic (port 29700)..."
        $TORCHRUN_CMD --master_port=29700 \
            arabic/train_stage2_ar.py \
            --stage1_ckpt "$STAGE1_CKPT" \
            --model_name "$MODEL_NAME" \
            --target_layers $TARGET_LAYERS \
            --output_dir "$AR_OUT_DIR" \
            --lambda_ot 0.5 \
            --lambda_span 1.0 \
            --lambda_margin 1.0 \
            --lambda_reg 50.0 \
            --lambda_kd 0.0 \
            --batch_size "$BATCH_SIZE" \
            --max_epochs "$STAGE2_EPOCHS" \
            --seed "$SEED" \
            --hf_repo_id "$HF_REPO"
        echo "  ✓ Hoàn tất huấn luyện Stage 2 Arabic!"
    fi

    # --- 4c. Hindi (HI) ---
    if [ "$SKIP_HI" -eq 1 ]; then
        echo "  ℹ Bỏ qua Hindi theo yêu cầu (--skip_hi)."
    elif [ "$FORCE_TRAIN" -eq 0 ] && [ -f "$HI_BEST_CKPT" ]; then
        echo "  ✓ Đã có checkpoint Stage 2 HI: $HI_BEST_CKPT (Bỏ qua training)"
    else
        echo ""
        echo "  [4c/3] Bắt đầu huấn luyện Stage 2 Hindi (port 29800)..."
        $TORCHRUN_CMD --master_port=29800 \
            hindi/train_stage2_hi.py \
            --stage1_ckpt "$STAGE1_CKPT" \
            --model_name "$MODEL_NAME" \
            --target_layers $TARGET_LAYERS \
            --output_dir "$HI_OUT_DIR" \
            --lambda_ot 0.5 \
            --lambda_span 1.0 \
            --lambda_margin 1.0 \
            --lambda_reg 50.0 \
            --lambda_kd 0.0 \
            --batch_size "$BATCH_SIZE" \
            --max_epochs "$STAGE2_EPOCHS" \
            --seed "$SEED" \
            --hf_repo_id "$HF_REPO"
        echo "  ✓ Hoàn tất huấn luyện Stage 2 Hindi!"
    fi
else
    echo ""
    echo "[BƯỚC 4/6] Chế độ --eval_only được bật: Bỏ qua toàn bộ bước huấn luyện."
fi

# ───────────────────────────────────────────────────────────────────
# BƯỚC 5: Đánh giá Toàn Diện XQuAD & MLQA (M1 Baseline vs M5 Ours)
# ───────────────────────────────────────────────────────────────────
echo ""
echo "[BƯỚC 5/6] Đánh giá toàn diện trên XQuAD & MLQA, xuất bảng báo cáo..."
python eval_and_aggregate_mmbert.py \
    --stage1_ckpt "$STAGE1_CKPT" \
    --vi_ckpt "$VI_BEST_CKPT" \
    --ar_ckpt "$AR_BEST_CKPT" \
    --hi_ckpt "$HI_BEST_CKPT" \
    --model_name "$MODEL_NAME" \
    --hf_repo_id "$HF_REPO"

# ───────────────────────────────────────────────────────────────────
# BƯỚC 6: Đồng bộ toàn bộ Assets lên Hugging Face Hub (mmBERT/) & Tổng kết
# ───────────────────────────────────────────────────────────────────
echo ""
echo "[BƯỚC 6/6] Đồng bộ toàn bộ kết quả, checkpoints & logs lên Hugging Face Hub (thư mục: mmBERT/)..."
python upload_all_mmbert_to_hf.py --repo_id "$HF_REPO" --prefix "mmBERT" || true

echo ""
echo "======================================================================"
echo "  🎉 PIPELINE mmBERT HOÀN TẤT THÀNH CÔNG!"
echo "  Bảng kết quả đã được lưu tại: results_mmbert_pipeline.md"
echo "  File JSON số liệu:            results_mmbert_pipeline.json"
echo "  Đường dẫn xem trên Hub:       https://huggingface.co/$HF_REPO/tree/main/mmBERT"
echo "  Thời gian kết thúc:           $(date)"
echo "======================================================================"

