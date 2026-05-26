# 🏷️ Gán nhãn chuỗi bằng BiLSTM-CRF cho NER

> **Đề tài 11** — So sánh BiLSTM+Softmax và BiLSTM+CRF cho bài toán Nhận dạng Thực thể Tên (NER)

---

## 📋 Mục tiêu

| Yêu cầu | Mô tả |
|---------|-------|
| BiLSTM + Softmax | Phân loại độc lập từng từ — dễ sai logic BIO |
| BiLSTM + CRF | Tối ưu toàn chuỗi — đảm bảo ràng buộc BIO |
| Phân tích lỗi boundary | Đếm lỗi vi phạm quy tắc IOB |
| F1-score theo entity | So sánh PER / GEO / ORG / GPE / TIM |
| Giao diện web | Demo tương tác + biểu đồ kết quả |

---

## 📁 Cấu trúc dự án

```
NER-BiLSTM-CRF/
├── 📄 README.md
├── 📄 requirements.txt
├── 🔧 prepare_data.py        ← Chuyển CSV → CoNLL format
├── 🔧 run_training.py        ← Huấn luyện cả 2 model
├── 🔧 error_analysis.py      ← Phân tích lỗi & đánh giá
├── 🪟 1_Huan_Luyen_Mo_Hinh.bat   ← Double-click để train (Windows)
├── 🪟 2_Chay_Giao_Dien_Web.bat   ← Double-click để chạy web (Windows)
│
├── 📂 src/                   ← Core model code
│   ├── model.py              ← BiLSTM-CRF architecture (PyTorch)
│   ├── loader.py             ← Đọc & chuẩn bị dữ liệu CoNLL
│   ├── utils.py              ← Hàm tiện ích (IOB, mapping, ...)
│   ├── train.py              ← Vòng lặp huấn luyện
│   ├── eval.py               ← Đánh giá bằng conlleval
│   └── data/                 ← Dữ liệu CoNLL (tạo sau bước 1)
│       ├── eng.train
│       ├── eng.testa
│       ├── eng.testb
│       └── eng.train50000
│
├── 📂 models/                ← Model đã train (tạo sau bước 2)
│   ├── bilstm_crf
│   ├── bilstm_softmax
│   └── mapping.pkl
│
└── 📂 web_app/               ← Giao diện web
    ├── app.py                ← FastAPI backend
    └── static/
        └── index.html        ← Frontend (single-page app)
```

---

## ⚡ Hướng dẫn chạy nhanh (Windows)

### Bước 1 — Cài đặt thư viện
```batch
pip install -r requirements.txt
```

### Bước 2 — Chuẩn bị dữ liệu
```batch
python prepare_data.py
```
> Đặt file `Data_theo_tung_tu.csv` vào cùng thư mục trước khi chạy.

### Bước 3 — Huấn luyện model
```batch
1_Huan_Luyen_Mo_Hinh.bat
```
hoặc:
```batch
python run_training.py --epochs 1
```

### Bước 4 — Chạy giao diện web
```batch
2_Chay_Giao_Dien_Web.bat
```
hoặc:
```batch
python -m uvicorn web_app.app:app --reload --port 8000
```
Mở trình duyệt: **http://localhost:8000**

### Bước 5 — Phân tích lỗi (tuỳ chọn)
```batch
python error_analysis.py --report evaluation_report.md
```

---

## 🌐 Tính năng giao diện

| Tab | Nội dung |
|-----|----------|
| **Demo tương tác** | Nhập câu → so sánh kết quả Softmax vs CRF với highlight màu |
| **F1-Score** | Biểu đồ cột theo entity (Precision / Recall / F1) |
| **Lỗi Boundary** | Thống kê + ví dụ cụ thể lỗi vi phạm BIO |
| **BiLSTM là gì?** | Giải thích kiến trúc, gates, bidirectional |
| **BiLSTM + Softmax** | Lý thuyết, hạn chế, công thức |
| **BiLSTM + CRF** | Transition matrix, Viterbi decode |
| **Vai trò CRF** | Kết luận từ thực nghiệm |

---

## 🧠 Kiến trúc Model

```
Input sentence
     │
     ▼
[Char CNN] + [Word Embedding] + [Cap Feature]
     │
     ▼
   BiLSTM (Bidirectional)
     │
     ├─── Softmax → Argmax(local)    → BiLSTM + Softmax
     │
     └─── CRF → Viterbi(global)     → BiLSTM + CRF  ✓
```

### Tham số mặc định

| Tham số | Giá trị |
|---------|---------|
| Word embedding dim | 100 |
| Char CNN dim | 25 |
| BiLSTM hidden dim | 200 |
| Dropout | 0.5 |
| Optimizer | SGD với momentum |
| Tag scheme | IOB |

---

## 📊 Kết quả thực nghiệm (1 epoch)

### F1-Score theo entity

| Entity | Softmax F1 | CRF F1 | Cải thiện |
|--------|-----------|--------|-----------|
| GEO    | 0.40 | 0.70 | +75% |
| GPE    | 0.00 | 0.59 | +∞ |
| ORG    | 0.02 | 0.49 | +2350% |
| PER    | 0.07 | 0.72 | +929% |
| TIM    | 0.08 | 0.67 | +738% |
| **Micro avg** | **0.24** | **0.65** | **+170%** |

### Lỗi Boundary (BIO violations)

| Model | Lỗi logic |
|-------|-----------|
| Ground Truth | 0 |
| BiLSTM + CRF | **0** ✅ |
| BiLSTM + Softmax | **403** ❌ |

---

## 🔗 Tài liệu tham khảo

- Lample et al. (2016) — *Neural Architectures for Named Entity Recognition* ([arXiv:1603.01360](https://arxiv.org/abs/1603.01360))
- Ma & Hovy (2016) — *End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF* ([arXiv:1603.01354](https://arxiv.org/abs/1603.01354))
- Code tham khảo: [ZubinGou/NER-BiLSTM-CRF-PyTorch](https://github.com/ZubinGou/NER-BiLSTM-CRF-PyTorch)
