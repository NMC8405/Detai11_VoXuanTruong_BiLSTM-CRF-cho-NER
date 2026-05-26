"""
error_analysis.py
──────────────────
Phân tích lỗi boundary, F1-score, confusion matrix chi tiết trên tập test.
So sánh BiLSTM+Softmax và BiLSTM+CRF.
Xuất kết quả JSON cho web app.

Cách dùng:
    python error_analysis.py
    python error_analysis.py --report evaluation_report.md
"""

import os
import sys
import pickle
import argparse
import json
import torch
import numpy as np
from collections import defaultdict, Counter

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

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
sys.path.insert(0, SRC_DIR)

try:
    from seqeval.metrics import classification_report, f1_score
    HAS_SEQEVAL = True
except ImportError:
    HAS_SEQEVAL = False
    print("⚠  seqeval không tìm thấy. Cài: pip install seqeval")

from model import BiLSTM_CRF
from loader import load_sentences, prepare_dataset, update_tag_scheme


MODELS_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
MAPPING_PATH = os.path.join(MODELS_DIR, 'mapping.pkl')
DATA_DIR     = os.path.join(SRC_DIR, 'data')


# ── Kiểm tra lỗi BIO boundary ──────────────────────────────────────
def check_boundary_errors(tags):
    """
    Đếm số lỗi vi phạm quy tắc BIO:
    - I-X xuất hiện mà không có B-X hoặc I-X cùng loại trước đó.
    """
    errors = []
    for i, tag in enumerate(tags):
        if not tag.startswith('I-'):
            continue
        ent_type = tag[2:]
        if i == 0:
            errors.append((i, tag, 'O', 'Bắt đầu câu với I-'))
        else:
            prev = tags[i - 1]
            if prev == 'O':
                errors.append((i, tag, prev, 'I- sau O'))
            elif prev[2:] != ent_type:
                errors.append((i, tag, prev, f'I-{ent_type} sau {prev}'))
    return errors


# ── Tính confusion matrix ──────────────────────────────────────────
def compute_confusion_matrix(y_true_flat, y_pred_flat, tag_list):
    """Tính confusion matrix cho các tag."""
    tag_to_idx = {t: i for i, t in enumerate(tag_list)}
    n = len(tag_list)
    matrix = np.zeros((n, n), dtype=int)

    for true, pred in zip(y_true_flat, y_pred_flat):
        if true in tag_to_idx and pred in tag_to_idx:
            matrix[tag_to_idx[true]][tag_to_idx[pred]] += 1

    return matrix.tolist()


# ── Phân tích entity-level errors ──────────────────────────────────
def analyze_entity_errors(y_true, y_pred, words_list):
    """Phân tích lỗi chi tiết theo từng loại entity."""
    error_examples = []

    for sent_true, sent_pred, sent_words in zip(y_true, y_pred, words_list):
        for i, (t, p) in enumerate(zip(sent_true, sent_pred)):
            if t != p:
                context_start = max(0, i - 2)
                context_end = min(len(sent_words), i + 3)
                context = ' '.join(sent_words[context_start:context_end])
                error_examples.append({
                    "word": sent_words[i] if i < len(sent_words) else "?",
                    "true_tag": t,
                    "pred_tag": p,
                    "context": context,
                    "position": i,
                })

    # Nhóm lỗi theo loại
    error_types = Counter()
    for e in error_examples:
        error_types[f"{e['true_tag']} → {e['pred_tag']}"] += 1

    return {
        "total_errors": len(error_examples),
        "error_type_counts": dict(error_types.most_common(20)),
        "examples": error_examples[:50],  # Giữ 50 ví dụ
    }


# ── Dự đoán với một model ────────────────────────────────────────────
def predict_model(model, test_data, id_to_tag, device):
    y_true, y_pred = [], []
    y_true_flat, y_pred_flat = [], []
    words_list = []
    all_errors = defaultdict(int)

    for data in test_data:
        gt_ids = data['tags']
        sentence_in = torch.LongTensor(data['words']).to(device)
        chars = data['chars']
        chars_length = [max(len(c), 1) for c in chars]
        char_maxl = max(chars_length)
        chars_mask = np.zeros((len(chars_length), char_maxl), dtype='int')
        for i, c in enumerate(chars):
            chars_mask[i, :chars_length[i]] = c[:chars_length[i]]
        chars_mask = torch.LongTensor(chars_mask).to(device)
        caps = torch.LongTensor(data['caps']).to(device)

        with torch.no_grad():
            _, tag_seq = model(sentence_in, chars_mask, caps, chars_length, {})

        true_tags = [id_to_tag[t] for t in gt_ids]
        pred_tags = [id_to_tag[t] for t in tag_seq]

        y_true.append(true_tags)
        y_pred.append(pred_tags)
        y_true_flat.extend(true_tags)
        y_pred_flat.extend(pred_tags)
        words_list.append(data['str_words'])

        for err in check_boundary_errors(pred_tags):
            all_errors[err[3]] += 1

    return y_true, y_pred, all_errors, y_true_flat, y_pred_flat, words_list


# ── In bảng kết quả ─────────────────────────────────────────────────
def print_section(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print('─'*60)


def run_analysis(args):
    if not os.path.exists(MAPPING_PATH):
        print(f"✗ Không tìm thấy mapping: {MAPPING_PATH}")
        print("  Hãy chạy training trước: python run_training.py")
        sys.exit(1)

    print("\n╔═══════════════════════════════════════════════════╗")
    print("║     🔍 NER Error Analysis & Evaluation            ║")
    print("╚═══════════════════════════════════════════════════╝")

    # Load mappings
    print("\n[1/5] Load mappings...")
    with open(MAPPING_PATH, 'rb') as f:
        mappings = pickle.load(f)
    tag_to_id = mappings['tag_to_id']
    id_to_tag = {v: k for k, v in tag_to_id.items()}
    params    = mappings['parameters']
    print(f"      Số nhãn: {len(tag_to_id)}  |  Tag scheme: {params.get('tag_scheme','iob')}")

    # Load test data
    print("[2/5] Load tập test...")
    test_path = os.path.join(DATA_DIR, 'eng.testb')
    if not os.path.exists(test_path):
        print(f"      ✗ Không tìm thấy: {test_path}")
        sys.exit(1)
    test_sents = load_sentences(test_path, params['lower'], params['zeros'])
    update_tag_scheme(test_sents, params['tag_scheme'])
    test_data = prepare_dataset(test_sents, mappings['word_to_id'],
                                mappings['char_to_id'], tag_to_id, params['lower'])
    print(f"      Số câu: {len(test_data):,}  |  Tổng token: {sum(len(d['words']) for d in test_data):,}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    report_lines = []
    all_results = {}

    # Tag list (bỏ START/STOP)
    tag_list = sorted([t for t in tag_to_id.keys() if t not in ('<START>', '<STOP>')])

    for model_name in ['bilstm_softmax', 'bilstm_crf']:
        model_path = os.path.join(MODELS_DIR, model_name)
        if not os.path.exists(model_path):
            print(f"\n⚠  Không tìm thấy model: {model_name} — bỏ qua")
            continue

        # Load model
        print(f"\n[3/5] Evaluate: {model_name} on {device}")
        model = torch.load(model_path, map_location=device, weights_only=False)
        model.to(device)
        model.eval()

        y_true, y_pred, errors, y_true_flat, y_pred_flat, words_list = \
            predict_model(model, test_data, id_to_tag, device)

        total_errors = sum(errors.values())
        is_crf = 'crf' in model_name

        print_section(f"{'BiLSTM + CRF' if is_crf else 'BiLSTM + Softmax'}")

        # F1 report
        report_dict = {}
        if HAS_SEQEVAL:
            report = classification_report(y_true, y_pred, digits=4)
            report_dict = classification_report(y_true, y_pred, digits=4, output_dict=True)
            print(report)
        else:
            print("  (Cài seqeval để xem F1 report chi tiết)")

        # Boundary errors
        print(f"\n  Lỗi Boundary (BIO violations):")
        print(f"  Tổng số lỗi: {total_errors}")
        if errors:
            for err_type, count in sorted(errors.items(), key=lambda x: -x[1])[:5]:
                print(f"    - {err_type}: {count} lần")
        else:
            print("  ✓ Không có lỗi boundary!")

        # Confusion matrix
        print(f"\n[4/5] Tính confusion matrix cho {model_name}...")
        cm = compute_confusion_matrix(y_true_flat, y_pred_flat, tag_list)

        # Entity-level error analysis
        print(f"[5/5] Phân tích entity-level errors cho {model_name}...")
        entity_errors = analyze_entity_errors(y_true, y_pred, words_list)
        print(f"      Tổng lỗi dự đoán: {entity_errors['total_errors']:,}")
        print(f"      Top lỗi:")
        for err_type, count in list(entity_errors['error_type_counts'].items())[:5]:
            print(f"        {err_type}: {count}")

        # Ghi report
        report_lines.append(f"\n## Model: {model_name}\n")
        if HAS_SEQEVAL:
            report_lines.append(f"```\n{report}\n```\n")
        report_lines.append(f"**Boundary Errors:** {total_errors}\n")

        # Lưu kết quả JSON
        model_result = {
            "model_name": model_name,
            "is_crf": is_crf,
            "boundary_errors": {
                "total": total_errors,
                "by_type": dict(errors),
            },
            "f1_report": report_dict,
            "confusion_matrix": {
                "tags": tag_list,
                "matrix": cm,
            },
            "entity_errors": entity_errors,
        }
        all_results[model_name] = model_result

    # Lưu tất cả kết quả vào JSON
    json_path = os.path.join(MODELS_DIR, 'evaluation_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  ✓ Evaluation results JSON → {json_path}")

    # Lưu report Markdown
    if args.report:
        os.makedirs(os.path.dirname(args.report) or '.', exist_ok=True)
        with open(args.report, 'w', encoding='utf-8') as f:
            f.write("# Báo cáo Đánh giá NER — BiLSTM+Softmax vs BiLSTM+CRF\n")
            f.writelines(report_lines)
        print(f"  ✓ Đã lưu báo cáo: {args.report}")

    print("\n" + "="*60)
    print("  ✅ Phân tích hoàn tất!")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Phân tích lỗi NER models")
    parser.add_argument('--report', default='evaluation_report.md',
                        help='Lưu báo cáo ra file Markdown')
    args = parser.parse_args()
    run_analysis(args)


if __name__ == '__main__':
    main()
