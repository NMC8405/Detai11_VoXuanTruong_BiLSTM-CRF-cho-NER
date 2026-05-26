"""
prepare_data.py
────────────────
Chuyển đổi dữ liệu CSV (Data_theo_tung_tu.csv) sang định dạng CoNLL
để phục vụ huấn luyện BiLSTM-CRF.

Cách dùng:
    python prepare_data.py
    python prepare_data.py --csv data/Data_theo_tung_tu.csv --out src/data
"""

import os
import sys
import random
import argparse
import pandas as pd
from collections import Counter

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

random.seed(42)


def process_data(csv_path: str, output_dir: str, train_ratio=0.8, dev_ratio=0.1):
    print(f"\n{'='*60}")
    print(f"  🏷️  Chuẩn bị dữ liệu NER từ CSV")
    print(f"{'='*60}")

    if not os.path.exists(csv_path):
        print(f"[LỖI] Không tìm thấy file: {csv_path}")
        sys.exit(1)

    print(f"[1/5] Đọc file: {csv_path}")
    try:
        df = pd.read_csv(csv_path, encoding='latin-1')
    except Exception:
        df = pd.read_csv(csv_path, encoding='utf-8')

    # Forward-fill Sentence #
    df['Sentence #'] = df['Sentence #'].ffill()

    print(f"      Tổng số token: {len(df):,}")
    print(f"      Các cột: {list(df.columns)}")

    # Xác định tên cột tự động
    col_word = next((c for c in df.columns if c.lower() in ['word','token','text']), df.columns[1])
    col_pos  = next((c for c in df.columns if c.lower() == 'pos'), None)
    col_tag  = next((c for c in df.columns if c.lower() in ['tag','ner','label']), df.columns[-1])
    col_sent = df.columns[0]

    print(f"      Cột word='{col_word}', pos='{col_pos}', tag='{col_tag}'")

    # ── Thống kê phân phối entity ────────────────────────────────────
    print(f"\n[2/5] Thống kê phân phối entity...")
    tag_counts = Counter(df[col_tag].dropna().astype(str))
    total_tokens = sum(tag_counts.values())

    entity_tags = {k: v for k, v in tag_counts.items() if k != 'O'}
    total_entity = sum(entity_tags.values())
    o_count = tag_counts.get('O', 0)

    print(f"      ┌─────────────────────────────────────────")
    print(f"      │ Tag              Count       %")
    print(f"      ├─────────────────────────────────────────")
    print(f"      │ O                {o_count:>8,}    {100*o_count/total_tokens:.1f}%")
    for tag, count in sorted(entity_tags.items(), key=lambda x: -x[1]):
        print(f"      │ {tag:<16} {count:>8,}    {100*count/total_tokens:.1f}%")
    print(f"      ├─────────────────────────────────────────")
    print(f"      │ Entity tokens    {total_entity:>8,}    {100*total_entity/total_tokens:.1f}%")
    print(f"      │ Tổng cộng        {total_tokens:>8,}    100.0%")
    print(f"      └─────────────────────────────────────────")

    # Đếm unique entity types (loại bỏ B-/I- prefix)
    entity_types = set()
    for tag in entity_tags:
        if '-' in tag:
            entity_types.add(tag.split('-', 1)[1].upper())
    print(f"      Loại entity: {sorted(entity_types)}")

    print(f"\n[3/5] Nhóm câu...")
    sentences = (
        df.groupby(col_sent, sort=False)
        .apply(lambda s: [
            (str(row[col_word]),
             str(row[col_pos]) if col_pos else 'NN',
             str(row[col_tag]))
            for _, row in s.iterrows()
        ])
        .tolist()
    )

    total = len(sentences)
    print(f"      Tổng số câu: {total:,}")

    # Hiển thị mẫu dữ liệu
    print(f"\n[3.5] Mẫu dữ liệu (câu đầu tiên):")
    if sentences:
        sample = sentences[0][:10]
        for word, pos, tag in sample:
            tag_color = "  " if tag == "O" else " *"
            print(f"      {tag_color} {word:<20} {pos:<6} {tag}")
        if len(sentences[0]) > 10:
            print(f"       ... và {len(sentences[0]) - 10} token nữa")

    random.shuffle(sentences)

    train_end = int(total * train_ratio)
    dev_end   = int(total * (train_ratio + dev_ratio))

    splits = {
        'eng.train':    sentences[:train_end],
        'eng.testa':    sentences[train_end:dev_end],
        'eng.testb':    sentences[dev_end:],
        'eng.train50000': sentences[:min(5000, train_end)],
    }

    for name, sents in splits.items():
        token_count = sum(len(s) for s in sents)
        print(f"      {name}: {len(sents):,} câu ({token_count:,} tokens)")

    print(f"\n[4/5] Ghi file CoNLL...")
    os.makedirs(output_dir, exist_ok=True)

    def write_conll(sents, path):
        with open(path, 'w', encoding='utf-8') as f:
            for sent in sents:
                for word, pos, tag in sent:
                    f.write(f"{word} {pos} {pos} {tag}\n")
                f.write("\n")

    for filename, sents in splits.items():
        out_path = os.path.join(output_dir, filename)
        write_conll(sents, out_path)
        size_kb = os.path.getsize(out_path) // 1024
        print(f"      ✓ {out_path} ({size_kb:,} KB)")

    print(f"\n[5/5] ✅ Hoàn thành! Dữ liệu sẵn sàng để huấn luyện.")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Chuẩn bị dữ liệu CoNLL cho NER")
    parser.add_argument('--csv', default=os.path.join('data', 'Data_theo_tung_tu.csv'),
                        help='Đường dẫn file CSV đầu vào')
    parser.add_argument('--out', default=os.path.join('src', 'data'),
                        help='Thư mục output')
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--dev-ratio',   type=float, default=0.1)
    args = parser.parse_args()

    # Tìm file CSV nếu không ở thư mục hiện tại
    csv_path = args.csv
    if not os.path.exists(csv_path):
        candidates = [
            'Data_theo_tung_tu.csv',
            os.path.join('data', 'Data_theo_tung_tu.csv'),
            os.path.join('..', 'data', 'Data_theo_tung_tu.csv'),
            os.path.join('..', args.csv),
        ]
        for c in candidates:
            if os.path.exists(c):
                csv_path = c
                print(f"  Tìm thấy CSV tại: {csv_path}")
                break

    process_data(csv_path, args.out, args.train_ratio, args.dev_ratio)


if __name__ == '__main__':
    main()
