import os
import sys
import pickle
import json
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional

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

# Add src to path
src_dir = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.insert(0, os.path.abspath(src_dir))

from model import BiLSTM_CRF
from loader import prepare_sentence

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_resources()
    load_real_f1_scores()
    yield

app = FastAPI(title="NER BiLSTM-CRF API", version="3.0.0", lifespan=lifespan)

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
mapping_path = os.path.join(models_dir, "mapping.pkl")

model_crf = None
model_softmax = None
mappings = None
id_to_tag = None

# ── F1-score results (default, will be overridden by real data if available) ──
F1_SCORES = {
    "softmax": {
        "geo":  {"precision": 0.28, "recall": 0.70, "f1": 0.40, "support": 3756},
        "gpe":  {"precision": 0.04, "recall": 0.00, "f1": 0.00, "support": 1561},
        "org":  {"precision": 0.03, "recall": 0.01, "f1": 0.02, "support": 1990},
        "per":  {"precision": 0.08, "recall": 0.06, "f1": 0.07, "support": 1789},
        "tim":  {"precision": 0.19, "recall": 0.05, "f1": 0.08, "support": 1989},
        "micro_avg": {"precision": 0.23, "recall": 0.26, "f1": 0.24, "support": 11085},
    },
    "crf": {
        "geo":  {"precision": 0.68, "recall": 0.72, "f1": 0.70, "support": 3756},
        "gpe":  {"precision": 0.61, "recall": 0.58, "f1": 0.59, "support": 1561},
        "org":  {"precision": 0.51, "recall": 0.48, "f1": 0.49, "support": 1990},
        "per":  {"precision": 0.74, "recall": 0.70, "f1": 0.72, "support": 1789},
        "tim":  {"precision": 0.70, "recall": 0.65, "f1": 0.67, "support": 1989},
        "micro_avg": {"precision": 0.66, "recall": 0.64, "f1": 0.65, "support": 11085},
    }
}

BOUNDARY_ERRORS = {
    "ground_truth": 0,
    "softmax": 403,
    "crf": 0,
    "total_sentences": 4800,
    "total_tokens": 94000
}

# ── Demo sentences for quick-test buttons ─────────────────────────────────────
DEMO_SENTENCES = [
    "Barack Obama was born in Hawaii and served as president of the United States.",
    "Apple Inc. was founded by Steve Jobs in Cupertino, California in 1976.",
    "The United Nations General Assembly met in New York on Tuesday.",
    "Elon Musk announced Tesla will open a new factory in Berlin next January.",
    "Google and Microsoft are competing for dominance in the artificial intelligence market.",
]


def load_real_f1_scores():
    """Load F1 scores thật từ evaluation_results.json nếu có."""
    global F1_SCORES, BOUNDARY_ERRORS
    eval_path = os.path.join(models_dir, 'evaluation_results.json')
    if not os.path.exists(eval_path):
        return

    try:
        with open(eval_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for model_key, json_key in [('softmax', 'bilstm_softmax'), ('crf', 'bilstm_crf')]:
            if json_key in data and 'f1_report' in data[json_key]:
                report = data[json_key]['f1_report']
                new_scores = {}
                for entity_key in ['geo', 'gpe', 'org', 'per', 'tim', 'art', 'eve', 'nat']:
                    for report_key in report:
                        if entity_key.upper() in report_key.upper():
                            r = report[report_key]
                            new_scores[entity_key] = {
                                "precision": round(r.get("precision", 0), 4),
                                "recall": round(r.get("recall", 0), 4),
                                "f1": round(r.get("f1-score", r.get("f1", 0)), 4),
                                "support": int(r.get("support", 0)),
                            }
                            break

                # Micro avg
                if "micro avg" in report:
                    ma = report["micro avg"]
                    new_scores["micro_avg"] = {
                        "precision": round(ma.get("precision", 0), 4),
                        "recall": round(ma.get("recall", 0), 4),
                        "f1": round(ma.get("f1-score", ma.get("f1", 0)), 4),
                        "support": int(ma.get("support", 0)),
                    }

                if new_scores:
                    F1_SCORES[model_key] = new_scores

            # Update boundary errors
            if json_key in data and 'boundary_errors' in data[json_key]:
                be = data[json_key]['boundary_errors']
                BOUNDARY_ERRORS[model_key] = be.get("total", 0)

        print("✓ Loaded real F1 scores from evaluation_results.json")
    except Exception as e:
        print(f"⚠ Could not load evaluation results: {e}")


def load_resources():
    global model_crf, model_softmax, mappings, id_to_tag
    if not os.path.exists(mapping_path):
        print("⚠  Mappings not found. Run training first (1_Huan_Luyen_Mo_Hinh.bat)")
        return
    try:
        with open(mapping_path, "rb") as f:
            mappings = pickle.load(f)
        tag_to_id = mappings["tag_to_id"]
        id_to_tag = {v: k for k, v in tag_to_id.items()}

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        crf_path = os.path.join(models_dir, "bilstm_crf")
        softmax_path = os.path.join(models_dir, "bilstm_softmax")

        if os.path.exists(crf_path):
            model_crf = torch.load(crf_path, map_location=device, weights_only=False)
            model_crf.device = device
            model_crf.to(device)
            model_crf.eval()
            print(f"✓ BiLSTM-CRF model loaded on {device}")

        if os.path.exists(softmax_path):
            model_softmax = torch.load(softmax_path, map_location=device, weights_only=False)
            model_softmax.device = device
            model_softmax.to(device)
            model_softmax.eval()
            print(f"✓ BiLSTM-Softmax model loaded on {device}")
    except Exception as e:
        print(f"Error loading models: {e}")


def _predict_with_model(model, text: str):
    if not model or not mappings:
        return []
    words = text.strip().split()
    if not words:
        return []
    word_to_id = mappings["word_to_id"]
    char_to_id = mappings["char_to_id"]
    data = prepare_sentence(words, word_to_id, char_to_id,
                            lower=mappings["parameters"]["lower"])
    device = getattr(model, "device", torch.device("cpu"))
    sentence_in = torch.LongTensor(data["words"]).to(device)
    chars = data["chars"]
    if not chars or any(len(c) == 0 for c in chars):
        for i, c in enumerate(chars):
            if len(c) == 0:
                chars[i] = [char_to_id.get('<UNK>', 0)]
    chars_length = [max(len(c), 1) for c in chars]
    char_maxl = max(chars_length)
    chars_mask = np.zeros((len(chars_length), char_maxl), dtype="int")
    for i, c in enumerate(chars):
        chars_mask[i, :chars_length[i]] = c[:chars_length[i]]
    chars_mask = torch.LongTensor(chars_mask).to(device)
    caps = torch.LongTensor(data["caps"]).to(device)
    d = {}
    with torch.no_grad():
        score, tag_seq = model(sentence_in, chars_mask, caps, chars_length, d)
    return [{"word": w, "tag": id_to_tag.get(t, "O")} for w, t in zip(words, tag_seq)]


# ── API Routes ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = os.path.join(static_dir, "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/status")
async def status():
    return {
        "models_loaded": {
            "crf": model_crf is not None,
            "softmax": model_softmax is not None,
        },
        "mappings_loaded": mappings is not None,
        "model_dir": models_dir,
        "mapping_exists": os.path.exists(mapping_path),
    }


class PredictRequest(BaseModel):
    text: str


@app.post("/api/predict")
async def api_predict(req: PredictRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if not mappings:
        # Return mock data for demo mode
        words = req.text.strip().split()
        mock_tags_crf = _mock_predict(words)
        return {
            "crf": mock_tags_crf,
            "softmax": _introduce_boundary_errors(mock_tags_crf),
            "demo_mode": True,
            "message": "Models not loaded. Showing demo predictions."
        }

    res_crf = _predict_with_model(model_crf, req.text)
    res_softmax = _predict_with_model(model_softmax, req.text)
    return {"crf": res_crf, "softmax": res_softmax, "demo_mode": False}


@app.get("/api/f1-scores")
async def get_f1_scores():
    return F1_SCORES


@app.get("/api/boundary-errors")
async def get_boundary_errors():
    return BOUNDARY_ERRORS


@app.get("/api/transition-matrix")
async def get_transition_matrix():
    """Trả transition matrix thật từ CRF model nếu có."""
    if model_crf and hasattr(model_crf, 'transitions'):
        try:
            trans = model_crf.transitions.detach().cpu().numpy()
            tag_to_id = mappings["tag_to_id"]
            id_to_tag_local = {v: k for k, v in tag_to_id.items()}
            # Lọc bỏ START/STOP tags
            valid_ids = [i for i, t in id_to_tag_local.items()
                         if t not in ('<START>', '<STOP>')]
            valid_tags = [id_to_tag_local[i] for i in valid_ids]
            sub_matrix = trans[np.ix_(valid_ids, valid_ids)]
            return {
                "tags": valid_tags,
                "values": np.round(sub_matrix, 4).tolist(),
                "is_real": True,
            }
        except Exception as e:
            print(f"Error extracting transition matrix: {e}")

    # Fallback to sample
    return {
        "tags": ["O", "B-PER", "I-PER", "B-GEO", "I-GEO", "B-ORG", "I-ORG", "B-GPE", "B-TIM", "I-TIM"],
        "values": [
            [ 0.80, 0.05, -9.99, 0.05, -9.99, 0.04, -9.99, 0.03, 0.03, -9.99],
            [-9.99, -9.99, 0.90, -9.99, -9.99, -9.99, -9.99, -9.99, -9.99, -9.99],
            [-9.99, -9.99, 0.65, -9.99, -9.99, -9.99, -9.99, -9.99, -9.99, -9.99],
            [-9.99, -9.99, -9.99, -9.99, 0.88, -9.99, -9.99, -9.99, -9.99, -9.99],
            [-9.99, -9.99, -9.99, -9.99, 0.60, -9.99, -9.99, -9.99, -9.99, -9.99],
            [-9.99, -9.99, -9.99, -9.99, -9.99, -9.99, 0.82, -9.99, -9.99, -9.99],
            [-9.99, -9.99, -9.99, -9.99, -9.99, -9.99, 0.55, -9.99, -9.99, -9.99],
            [-9.99, -9.99, -9.99, -9.99, -9.99, -9.99, -9.99, -9.99, -9.99, -9.99],
            [-9.99, -9.99, -9.99, -9.99, -9.99, -9.99, -9.99, -9.99, -9.99, 0.85],
            [-9.99, -9.99, -9.99, -9.99, -9.99, -9.99, -9.99, -9.99, -9.99, 0.60],
        ],
        "is_real": False,
    }


@app.get("/api/demo-sentences")
async def get_demo_sentences():
    return DEMO_SENTENCES


@app.get("/api/training-history")
async def get_training_history():
    """Trả training history cho cả 2 model."""
    result = {}
    for model_name in ['bilstm_softmax', 'bilstm_crf']:
        hist_path = os.path.join(models_dir, f'training_history_{model_name}.json')
        if os.path.exists(hist_path):
            with open(hist_path, 'r', encoding='utf-8') as f:
                result[model_name] = json.load(f)
    if not result:
        # Return mock data
        result = {
            "bilstm_softmax": {
                "model_type": "BiLSTM+Softmax",
                "epochs": [
                    {"epoch": 1, "avg_loss": 2.1, "dev_f1": 15.0, "test_f1": 14.5},
                    {"epoch": 2, "avg_loss": 1.2, "dev_f1": 20.0, "test_f1": 19.5},
                    {"epoch": 3, "avg_loss": 0.8, "dev_f1": 22.5, "test_f1": 22.0},
                    {"epoch": 4, "avg_loss": 0.6, "dev_f1": 23.5, "test_f1": 23.2},
                    {"epoch": 5, "avg_loss": 0.5, "dev_f1": 24.0, "test_f1": 23.8},
                ],
                "demo_mode": True,
            },
            "bilstm_crf": {
                "model_type": "BiLSTM+CRF",
                "epochs": [
                    {"epoch": 1, "avg_loss": 4.5, "dev_f1": 55.0, "test_f1": 54.5},
                    {"epoch": 2, "avg_loss": 1.8, "dev_f1": 62.0, "test_f1": 61.5},
                    {"epoch": 3, "avg_loss": 1.0, "dev_f1": 64.5, "test_f1": 64.0},
                    {"epoch": 4, "avg_loss": 0.7, "dev_f1": 65.5, "test_f1": 65.0},
                    {"epoch": 5, "avg_loss": 0.5, "dev_f1": 66.0, "test_f1": 65.5},
                ],
                "demo_mode": True,
            }
        }
    return result


@app.get("/api/ablation-results")
async def get_ablation_results():
    """Trả kết quả ablation study."""
    ablation_path = os.path.join(models_dir, 'ablation_results.json')
    if os.path.exists(ablation_path):
        with open(ablation_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    # Mock data
    return {
        "variants": [
            {"label": "BiLSTM+Softmax (no char)", "best_dev_f1": 18.5, "final_test_f1": 17.8, "crf": False, "char_dim": 0},
            {"label": "BiLSTM+Softmax (char CNN)", "best_dev_f1": 24.0, "final_test_f1": 23.5, "crf": False, "char_dim": 25},
            {"label": "BiLSTM+CRF (no char)", "best_dev_f1": 58.0, "final_test_f1": 57.5, "crf": True, "char_dim": 0},
            {"label": "BiLSTM+CRF (char CNN)", "best_dev_f1": 65.0, "final_test_f1": 64.5, "crf": True, "char_dim": 25},
        ],
        "demo_mode": True,
    }


@app.get("/api/confusion-matrix")
async def get_confusion_matrix():
    """Trả confusion matrix từ evaluation results."""
    eval_path = os.path.join(models_dir, 'evaluation_results.json')
    if os.path.exists(eval_path):
        with open(eval_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        result = {}
        for model_name in ['bilstm_softmax', 'bilstm_crf']:
            if model_name in data and 'confusion_matrix' in data[model_name]:
                result[model_name] = data[model_name]['confusion_matrix']
        if result:
            return result

    # Mock
    tags = ["O", "B-geo", "I-geo", "B-per", "I-per", "B-org", "I-org", "B-gpe", "B-tim", "I-tim"]
    return {
        "bilstm_crf": {"tags": tags, "matrix": [[0]*len(tags)]*len(tags)},
        "demo_mode": True,
    }


# ── Mock predictions for demo when models are not loaded ──────────────────────

_ENTITY_PATTERNS = {
    "B-PER": ["obama", "trump", "biden", "jobs", "musk", "gates", "zuckerberg",
              "einstein", "napoleon", "lincoln", "shakespeare"],
    "B-ORG": ["apple", "google", "microsoft", "tesla", "amazon", "facebook",
              "netflix", "nasa", "un", "nato", "who", "ibm"],
    "B-GEO": ["hawaii", "california", "berlin", "paris", "london", "tokyo",
              "washington", "america", "europe", "asia", "africa"],
    "B-GPE": ["united", "states", "france", "germany", "china", "russia",
              "japan", "india", "brazil", "australia"],
    "B-TIM": ["monday", "tuesday", "wednesday", "thursday", "friday",
              "january", "february", "march", "2024", "2025", "next", "last", "today"],
}

_CONTINUE = {
    "B-PER": "I-PER", "B-ORG": "I-ORG",
    "B-GEO": "I-GEO", "B-GPE": "I-GPE", "B-TIM": "I-TIM"
}


def _mock_predict(words: list) -> list:
    result = []
    prev_tag = "O"
    for w in words:
        w_lower = w.lower().rstrip(".,;:!?")
        tag = "O"
        for b_tag, patterns in _ENTITY_PATTERNS.items():
            if w_lower in patterns:
                tag = b_tag
                break
        if tag == "O" and prev_tag != "O" and prev_tag.startswith("I-"):
            ent = prev_tag[2:]
            if w_lower not in {"the", "a", "an", "of", "and", "or", "in", "at", "on"}:
                if len(w) > 1 and w[0].isupper():
                    tag = f"I-{ent}"
        result.append({"word": w, "tag": tag})
        prev_tag = tag
    return result


def _introduce_boundary_errors(predictions: list) -> list:
    """Simulate softmax errors by occasionally converting B- to I- at boundaries."""
    import random
    random.seed(42)
    result = []
    for i, item in enumerate(predictions):
        tag = item["tag"]
        if tag.startswith("B-") and i > 0 and predictions[i-1]["tag"] == "O":
            if random.random() < 0.25:
                tag = "I-" + tag[2:]
        result.append({"word": item["word"], "tag": tag})
    return result


# ── Main Entrypoint ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print("\nStarting local server on http://127.0.0.1:8000...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
