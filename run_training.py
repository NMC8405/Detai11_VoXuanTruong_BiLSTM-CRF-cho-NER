"""
run_training.py
────────────────
Huấn luyện cả hai mô hình BiLSTM+Softmax và BiLSTM+CRF.
Hỗ trợ chế độ ablation study để so sánh các biến thể.

Cách dùng:
    python run_training.py                     # Train cả 2 model (5 epochs)
    python run_training.py --model crf         # Chỉ train CRF
    python run_training.py --model softmax     # Chỉ train Softmax
    python run_training.py --epochs 10         # 10 epochs
    python run_training.py --ablation          # Chạy ablation study đầy đủ
"""

import subprocess
import sys
import os
import argparse
import time
import json

# Reconfigure stdout/stderr to UTF-8 to prevent Windows UnicodeEncodeError
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass
if sys.stderr.encoding != 'utf-8':
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# ── Đường dẫn ──────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR  = os.path.join(BASE_DIR, 'src')
DATA_DIR = os.path.join(SRC_DIR, 'data')
MODEL_OUTPUT = os.path.join(BASE_DIR, 'models')


def check_data():
    """Kiểm tra dữ liệu CoNLL đã tồn tại chưa."""
    required = ['eng.train', 'eng.testa', 'eng.testb']
    missing = [f for f in required if not os.path.exists(os.path.join(DATA_DIR, f))]
    if missing:
        print(f"\n⚠  Thiếu dữ liệu: {missing}")
        print("   Chạy prepare_data.py trước:")
        print("   python prepare_data.py\n")
        return False
    sizes = {f: os.path.getsize(os.path.join(DATA_DIR, f)) // 1024 for f in required}
    print("   Dữ liệu tìm thấy:")
    for f, kb in sizes.items():
        print(f"   ✓ {f} ({kb:,} KB)")
    return True


def check_gpu():
    """Kiểm tra GPU CUDA có sẵn không."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"   ✓ GPU: {gpu_name} ({gpu_mem:.1f} GB)")
            return True
        else:
            print("   ⚠ CUDA không khả dụng — dùng CPU")
            return False
    except Exception as e:
        print(f"   ⚠ Không kiểm tra được GPU ({e}) — dùng CPU")
        return False


def train_model(use_crf: bool, name: str, epochs: int, use_gpu: bool,
                word_dim=100, hidden_dim=200, char_dim=25, char_mode='CNN'):
    model_type = "BiLSTM + CRF" if use_crf else "BiLSTM + Softmax"
    print(f"\n{'='*60}")
    print(f"  Training: {model_type}  →  '{name}'")
    print(f"  Epochs={epochs}, word_dim={word_dim}, hidden_dim={hidden_dim}")
    print(f"  char_dim={char_dim}, char_mode={char_mode}, GPU={'✓' if use_gpu else '✗'}")
    print(f"{'='*60}")

    cmd = [
        sys.executable, 'train.py',
        '--pre_emb', '',
        '--all_emb', '0',
        '--word_dim', str(word_dim),
        '--char_dim', str(char_dim),
        '--char_lstm_dim', str(char_dim),
        '--hidden_dim', str(hidden_dim),
        '--epoch', str(epochs),
        '--crf', '1' if use_crf else '0',
        '--name', name,
        '--use_gpu', '1' if use_gpu else '0',
        '--models_path', os.path.abspath(MODEL_OUTPUT),
        '--char_mode', char_mode,
    ]

    os.makedirs(MODEL_OUTPUT, exist_ok=True)
    t0 = time.time()

    result = subprocess.run(cmd, cwd=SRC_DIR)

    elapsed = time.time() - t0
    if result.returncode == 0:
        print(f"\n  ✓ Hoàn thành {model_type} trong {elapsed:.1f}s")
    else:
        print(f"\n  ✗ Training thất bại (exit code {result.returncode})")
    return result.returncode == 0


def run_ablation(epochs: int, use_gpu: bool, word_dim: int, hidden_dim: int, char_dim: int):
    """
    Chạy ablation study — so sánh các biến thể model:
    1. BiLSTM + Softmax (no char) — Baseline
    2. BiLSTM + Softmax (char CNN)
    3. BiLSTM + CRF (no char)
    4. BiLSTM + CRF (char CNN) — Full model
    """
    print("\n╔═══════════════════════════════════════════════════╗")
    print("║       Ablation Study — So sánh các biến thể       ║")
    print("╚═══════════════════════════════════════════════════╝")

    variants = [
        {"name": "ablation_softmax_nochar", "crf": False, "char_dim": 0, "char_mode": "CNN",
         "label": "BiLSTM+Softmax (no char)"},
        {"name": "ablation_softmax_char",   "crf": False, "char_dim": char_dim, "char_mode": "CNN",
         "label": "BiLSTM+Softmax (char CNN)"},
        {"name": "ablation_crf_nochar",     "crf": True,  "char_dim": 0, "char_mode": "CNN",
         "label": "BiLSTM+CRF (no char)"},
        {"name": "ablation_crf_char",       "crf": True,  "char_dim": char_dim, "char_mode": "CNN",
         "label": "BiLSTM+CRF (char CNN)"},
    ]

    results = []
    for i, v in enumerate(variants):
        print(f"\n{'─'*50}")
        print(f"  Ablation [{i+1}/{len(variants)}]: {v['label']}")
        print(f"{'─'*50}")

        ok = train_model(
            use_crf=v["crf"],
            name=v["name"],
            epochs=epochs,
            use_gpu=use_gpu,
            word_dim=word_dim,
            hidden_dim=hidden_dim,
            char_dim=v["char_dim"],
            char_mode=v["char_mode"],
        )

        # Đọc kết quả từ training history
        hist_path = os.path.join(MODEL_OUTPUT, f'training_history_{v["name"]}.json')
        result_entry = {
            "name": v["name"],
            "label": v["label"],
            "crf": v["crf"],
            "char_dim": v["char_dim"],
            "char_mode": v["char_mode"],
            "success": ok,
        }

        if os.path.exists(hist_path):
            with open(hist_path, 'r', encoding='utf-8') as f:
                hist = json.load(f)
            if hist.get("epochs"):
                last = hist["epochs"][-1]
                result_entry["best_dev_f1"] = hist.get("best_dev_f1", 0)
                result_entry["final_test_f1"] = last.get("test_f1", 0)
                result_entry["final_loss"] = last.get("avg_loss", 0)
                result_entry["training_time"] = hist.get("total_training_time", 0)
                result_entry["epochs_data"] = hist["epochs"]

        results.append(result_entry)

    # Lưu kết quả ablation
    ablation_path = os.path.join(MODEL_OUTPUT, 'ablation_results.json')
    with open(ablation_path, 'w', encoding='utf-8') as f:
        json.dump({"variants": results, "epochs": epochs}, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  ✓ Ablation results saved → {ablation_path}")

    # In bảng kết quả
    print(f"\n{'═'*60}")
    print(f"  📊 Ablation Study Results")
    print(f"{'═'*60}")
    print(f"  {'Variant':<35} {'Dev F1':>8} {'Test F1':>8} {'Time':>8}")
    print(f"  {'─'*35} {'─'*8} {'─'*8} {'─'*8}")
    for r in results:
        dev_f1 = r.get("best_dev_f1", "N/A")
        test_f1 = r.get("final_test_f1", "N/A")
        t = r.get("training_time", "N/A")
        dev_str = f"{dev_f1:.1f}%" if isinstance(dev_f1, (int, float)) else dev_f1
        test_str = f"{test_f1:.1f}%" if isinstance(test_f1, (int, float)) else test_f1
        t_str = f"{t:.0f}s" if isinstance(t, (int, float)) else t
        print(f"  {r['label']:<35} {dev_str:>8} {test_str:>8} {t_str:>8}")
    print(f"{'═'*60}\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện BiLSTM-NER models")
    parser.add_argument('--model',   choices=['crf','softmax','both'], default='both',
                        help='Model cần train')
    parser.add_argument('--epochs',  type=int, default=5,
                        help='Số epoch (mặc định 5 cho kết quả tốt)')
    parser.add_argument('--gpu',     action='store_true', default=True,
                        help='Dùng GPU (CUDA) nếu có — mặc định BẬT')
    parser.add_argument('--no-gpu',  action='store_true',
                        help='Tắt GPU, dùng CPU')
    parser.add_argument('--word-dim',   type=int, default=100)
    parser.add_argument('--hidden-dim', type=int, default=200)
    parser.add_argument('--char-dim',   type=int, default=25)
    parser.add_argument('--ablation',   action='store_true',
                        help='Chạy ablation study (4 biến thể)')
    args = parser.parse_args()

    if args.no_gpu:
        args.gpu = False

    print("\n╔═══════════════════════════════════════════════════╗")
    print("║     🏷️  NER BiLSTM-CRF — Training Pipeline       ║")
    print("╚═══════════════════════════════════════════════════╝")

    # Kiểm tra dữ liệu
    print("\n[Bước 1] Kiểm tra dữ liệu...")
    if not check_data():
        sys.exit(1)

    # Kiểm tra GPU
    print(f"\n[Bước 2] Kiểm tra thiết bị...")
    gpu_available = check_gpu() if args.gpu else False
    use_gpu = args.gpu and gpu_available

    print(f"\n[Bước 3] Cấu hình training:")
    print(f"   Model: {args.model}  |  Epochs: {args.epochs}")
    print(f"   GPU: {'✓ ' + ('Có' if use_gpu else 'Không') }  |  word_dim={args.word_dim}")
    print(f"   Ablation: {'✓ Có' if args.ablation else '✗ Không'}")

    ok = True

    if args.ablation:
        # Chạy ablation study
        run_ablation(args.epochs, use_gpu, args.word_dim, args.hidden_dim, args.char_dim)
    else:
        # Train bình thường
        if args.model in ('softmax', 'both'):
            ok &= train_model(False, 'bilstm_softmax', args.epochs, use_gpu,
                              args.word_dim, args.hidden_dim, args.char_dim)
        if args.model in ('crf', 'both'):
            ok &= train_model(True,  'bilstm_crf',     args.epochs, use_gpu,
                              args.word_dim, args.hidden_dim, args.char_dim)

    print("\n" + "="*60)
    if ok:
        print("  ✅ Training hoàn tất! Chạy web app:")
        print("     python -m uvicorn web_app.app:app --reload --port 8000")
        print("  Hoặc double-click: 2_Chay_Giao_Dien_Web.bat")
    else:
        print("  ❌ Có lỗi trong quá trình training. Xem log trên.")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
