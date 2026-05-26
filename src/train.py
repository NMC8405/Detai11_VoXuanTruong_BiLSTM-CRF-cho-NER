# coding=utf-8
"""
train.py (enhanced version — training history logging)
Huấn luyện BiLSTM-CRF / BiLSTM-Softmax cho NER.
Tự động lưu training history (loss, F1) vào JSON file.
"""
import os, sys, pickle, argparse, time, itertools, json
import numpy as np
import torch
from collections import OrderedDict
from torch.autograd import Variable

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

import loader
from utils import *
from loader import *
from model import BiLSTM_CRF

# ── Argument parser ──────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument('--train',      default='data/eng.train')
p.add_argument('--dev',        default='data/eng.testa')
p.add_argument('--test',       default='data/eng.testb')
p.add_argument('--test_train', default='data/eng.train50000')
p.add_argument('--tag_scheme', default='iob', choices=['iob','iobes'])
p.add_argument('--lower',      default=1, type=int)
p.add_argument('--zeros',      default=0, type=int)
p.add_argument('--char_dim',      default=25,  type=int)
p.add_argument('--char_lstm_dim', default=25,  type=int)
p.add_argument('--char_bidirect', default=1,   type=int)
p.add_argument('--word_dim',      default=100, type=int)
p.add_argument('--hidden_dim',    default=200, type=int)
p.add_argument('--word_lstm_dim', default=200, type=int)
p.add_argument('--pre_emb',       default='')
p.add_argument('--all_emb',       default=0, type=int)
p.add_argument('--cap_dim',       default=0, type=int)
p.add_argument('--crf',           default=1, type=int)
p.add_argument('--dropout',       default=0.5, type=float)
p.add_argument('--epoch',         default=1, type=int)
p.add_argument('--use_gpu',       default=0, type=int)
p.add_argument('--name',          default='model')
p.add_argument('--char_mode',     default='CNN', choices=['CNN','LSTM'])
p.add_argument('--models_path',   default='models')
p.add_argument('--reload',        default=0, type=int)
opts = p.parse_args()

# ── Tham số ──────────────────────────────────────────────────────
parameters = OrderedDict()
parameters['tag_scheme']  = opts.tag_scheme
parameters['lower']       = opts.lower == 1
parameters['zeros']       = opts.zeros == 1
parameters['char_dim']    = opts.char_dim
parameters['char_lstm_dim']= opts.char_lstm_dim
parameters['char_bidirect']= opts.char_bidirect == 1
parameters['word_dim']    = opts.word_dim
parameters['word_lstm_dim']= opts.hidden_dim or opts.word_lstm_dim
parameters['pre_emb']     = opts.pre_emb
parameters['all_emb']     = opts.all_emb == 1
parameters['cap_dim']     = opts.cap_dim
parameters['crf']         = opts.crf == 1
parameters['dropout']     = opts.dropout
parameters['char_mode']   = opts.char_mode
parameters['name']        = opts.name

use_gpu = opts.use_gpu == 1 and torch.cuda.is_available()
device  = torch.device('cuda' if use_gpu else 'cpu')

models_path  = opts.models_path
os.makedirs(models_path, exist_ok=True)
model_name   = os.path.join(models_path, opts.name)
mapping_file = os.path.join(models_path, 'mapping.pkl')

eval_path = './evaluation'
eval_temp = os.path.join(eval_path, 'temp')
eval_script = os.path.join(eval_path, 'conlleval')
os.makedirs(eval_temp, exist_ok=True)

# ── Training History ─────────────────────────────────────────────
history_file = os.path.join(models_path, f'training_history_{opts.name}.json')
training_history = {
    "model_name": opts.name,
    "model_type": "BiLSTM+CRF" if parameters['crf'] else "BiLSTM+Softmax",
    "parameters": {k: str(v) if not isinstance(v, (int, float, bool)) else v
                   for k, v in parameters.items()},
    "device": str(device),
    "epochs": []
}


# ── Evaluation (dùng seqeval, fallback conlleval) ────────────────
def evaluating(model, datas, best_F, return_report=False):
    predictions, save = [], False
    new_F = 0.0
    report_data = {}

    for data in datas:
        words   = data['str_words']
        chars   = data['chars']
        caps    = data['caps']
        gt_ids  = data['tags']

        d = {}
        chars_length = [max(len(c), 1) for c in chars]
        char_maxl    = max(chars_length)
        chars_mask   = np.zeros((len(chars_length), char_maxl), dtype='int')
        for i, c in enumerate(chars):
            chars_mask[i, :chars_length[i]] = c[:chars_length[i]]
        chars_mask = Variable(torch.LongTensor(chars_mask)).to(device)
        dwords = Variable(torch.LongTensor(data['words'])).to(device)
        dcaps  = Variable(torch.LongTensor(caps)).to(device)

        with torch.no_grad():
            _, out = model(dwords, chars_mask, dcaps, chars_length, d)

        for w, true_id, pred_id in zip(words, gt_ids, out):
            predictions.append(f'{w} {id_to_tag[true_id]} {id_to_tag[pred_id]}')
        predictions.append('')

    # Dùng seqeval
    try:
        from seqeval.metrics import f1_score as seq_f1, classification_report
        y_true, y_pred = [], []
        cur_true, cur_pred = [], []
        for line in predictions:
            if line.strip() == '':
                if cur_true:
                    y_true.append(cur_true); y_pred.append(cur_pred)
                    cur_true, cur_pred = [], []
            else:
                parts = line.split()
                if len(parts) >= 3:
                    cur_true.append(parts[1])
                    cur_pred.append(parts[2])
        if cur_true:
            y_true.append(cur_true); y_pred.append(cur_pred)
        new_F = seq_f1(y_true, y_pred) * 100

        if return_report:
            report_str = classification_report(y_true, y_pred, digits=4, output_dict=True)
            report_data = report_str
    except Exception as e:
        # Fallback: dùng conlleval nếu có
        predf  = os.path.join(eval_temp, f'pred.{opts.name}')
        scoref = os.path.join(eval_temp, f'score.{opts.name}')
        with open(predf, 'w') as f:
            f.write('\n'.join(predictions))
        if os.path.isfile(eval_script):
            os.system(f'{eval_script} < {predf} > {scoref}')
            lines = open(scoref, encoding='utf8').readlines()
            if len(lines) > 1:
                try:
                    new_F = float(lines[1].strip().split()[-1])
                except Exception:
                    pass

    if new_F > best_F:
        best_F, save = new_F, True
        print(f'   ✓ Best F={new_F:.2f}% → saving model')

    if return_report:
        return best_F, new_F, save, report_data
    return best_F, new_F, save


# ── Load dữ liệu ─────────────────────────────────────────────────
lower      = parameters['lower']
zeros      = parameters['zeros']
tag_scheme = parameters['tag_scheme']

print(f'\n[Data] Loading sentences...')
train_sents    = loader.load_sentences(opts.train,      lower, zeros)
dev_sents      = loader.load_sentences(opts.dev,        lower, zeros)
test_sents     = loader.load_sentences(opts.test,       lower, zeros)
testtrain_sents= loader.load_sentences(opts.test_train, lower, zeros)

for sents in [train_sents, dev_sents, test_sents, testtrain_sents]:
    update_tag_scheme(sents, tag_scheme)

dico_words_train = word_mapping(train_sents, lower)[0]

# Pretrained embeddings
if opts.pre_emb and os.path.isfile(opts.pre_emb):
    dico_words, word_to_id, id_to_word = augment_with_pretrained(
        dico_words_train.copy(), opts.pre_emb,
        list(itertools.chain.from_iterable(
            [[w[0] for w in s] for s in dev_sents + test_sents]
        )) if not parameters['all_emb'] else None
    )
else:
    dico_words  = dico_words_train
    word_to_id, id_to_word = word_mapping(train_sents, lower)[1:]

dico_chars, char_to_id, id_to_char = char_mapping(train_sents)
dico_tags,  tag_to_id,  id_to_tag  = tag_mapping(train_sents)

train_data    = prepare_dataset(train_sents,     word_to_id, char_to_id, tag_to_id, lower)
dev_data      = prepare_dataset(dev_sents,       word_to_id, char_to_id, tag_to_id, lower)
test_data     = prepare_dataset(test_sents,      word_to_id, char_to_id, tag_to_id, lower)
testtrain_data= prepare_dataset(testtrain_sents, word_to_id, char_to_id, tag_to_id, lower)

print(f'[Data] {len(train_data):,} train / {len(dev_data):,} dev / {len(test_data):,} test sentences')

# Word embeddings
word_embeds = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06),
                                (len(word_to_id), opts.word_dim))
if opts.pre_emb and os.path.isfile(opts.pre_emb):
    all_we = {}
    for line in open(opts.pre_emb, 'r', encoding='utf-8'):
        s = line.strip().split()
        if len(s) == opts.word_dim + 1:
            all_we[s[0]] = np.array([float(x) for x in s[1:]])
    for w in word_to_id:
        if w in all_we:      word_embeds[word_to_id[w]] = all_we[w]
        elif w.lower() in all_we: word_embeds[word_to_id[w]] = all_we[w.lower()]
    print(f'[Data] Loaded {len(all_we):,} pretrained embeddings')

# Save mappings
with open(mapping_file, 'wb') as f:
    pickle.dump({'word_to_id': word_to_id, 'tag_to_id': tag_to_id,
                 'char_to_id': char_to_id, 'parameters': parameters,
                 'word_embeds': word_embeds}, f)
print(f'[Data] Mappings saved → {mapping_file}')

# ── Build model ────────────────────────────────────────────────
model = BiLSTM_CRF(
    vocab_size=len(word_to_id),
    tag_to_ix=tag_to_id,
    embedding_dim=parameters['word_dim'],
    hidden_dim=parameters['word_lstm_dim'],
    char_lstm_dim=parameters['char_lstm_dim'],
    char_to_ix=char_to_id,
    pre_word_embeds=word_embeds,
    use_crf=parameters['crf'],
    char_mode=parameters['char_mode'],
    use_gpu=use_gpu,
)
model.to(device)

if opts.reload and os.path.isfile(model_name):
    model = torch.load(model_name, map_location=device, weights_only=False)
    model.to(device)
    print(f'[Model] Reloaded from {model_name}')

tag_type = 'BiLSTM+CRF' if parameters['crf'] else 'BiLSTM+Softmax'
print(f'\n[Train] Model: {tag_type}  |  Epochs: {opts.epoch}  |  Device: {device}')

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'[Model] Total params: {total_params:,}  |  Trainable: {trainable_params:,}')
training_history["total_params"] = total_params
training_history["trainable_params"] = trainable_params

# ── Training loop ─────────────────────────────────────────────
lr        = 0.015
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
best_dev_F= -1.0
count     = 0

model.train(True)
for epoch in range(1, opts.epoch + 1):
    epoch_loss = 0.0
    epoch_losses = []
    t0 = time.time()
    indices = np.random.permutation(len(train_data))

    for k, idx in enumerate(indices):
        data = train_data[idx]
        model.zero_grad()
        count += 1

        sentence_in = Variable(torch.LongTensor(data['words'])).to(device)
        targets     = torch.LongTensor(data['tags']).to(device)
        chars       = data['chars']
        d = {}
        chars_length = [max(len(c), 1) for c in chars]
        char_maxl    = max(chars_length)
        chars_mask   = np.zeros((len(chars_length), char_maxl), dtype='int')
        for i, c in enumerate(chars):
            chars_mask[i, :chars_length[i]] = c[:chars_length[i]]
        chars_mask = Variable(torch.LongTensor(chars_mask)).to(device)
        caps = Variable(torch.LongTensor(data['caps'])).to(device)

        loss = model.neg_log_likelihood(sentence_in, targets, chars_mask, caps, chars_length, d)
        step_loss = loss.item() / max(len(data['words']), 1)
        epoch_loss += step_loss
        epoch_losses.append(step_loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        if (k + 1) % 200 == 0:
            pct = 100 * (k+1) / len(train_data)
            avg = epoch_loss / (k+1)
            elapsed = time.time() - t0
            print(f'  Epoch {epoch}/{opts.epoch} | {pct:5.1f}% | loss={avg:.4f} | {elapsed:.0f}s')
            sys.stdout.flush()

    # Evaluate at end of epoch
    model.train(False)
    elapsed = time.time() - t0
    avg_loss = epoch_loss / len(train_data)
    print(f'\n  Epoch {epoch} done — avg_loss={avg_loss:.4f} — {elapsed:.1f}s')

    best_dev_F, dev_F, save, dev_report = evaluating(model, dev_data, best_dev_F, return_report=True)
    print(f'  Dev  F1: {dev_F:.2f}%')
    if save:
        torch.save(model, model_name)
        print(f'  Model saved → {model_name}')
    _, test_F, _, test_report = evaluating(model, test_data, -1, return_report=True)
    print(f'  Test F1: {test_F:.2f}%\n')
    sys.stdout.flush()

    # ── Save epoch history ────────────────────────────────────
    epoch_record = {
        "epoch": epoch,
        "avg_loss": round(avg_loss, 6),
        "dev_f1": round(dev_F, 2),
        "test_f1": round(test_F, 2),
        "best_dev_f1": round(best_dev_F, 2),
        "learning_rate": round(lr / (1 + 0.05 * count / len(train_data)), 6),
        "elapsed_seconds": round(elapsed, 1),
        "dev_report": dev_report if isinstance(dev_report, dict) else {},
        "test_report": test_report if isinstance(test_report, dict) else {},
    }
    training_history["epochs"].append(epoch_record)

    # Save history incrementally
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(training_history, f, indent=2, ensure_ascii=False, default=str)

    model.train(True)
    adjust_learning_rate(optimizer, lr=lr / (1 + 0.05 * count / len(train_data)))

# ── Final summary ────────────────────────────────────────────────
training_history["completed"] = True
training_history["best_dev_f1"] = round(best_dev_F, 2)
training_history["total_training_time"] = round(
    sum(e["elapsed_seconds"] for e in training_history["epochs"]), 1
)

with open(history_file, 'w', encoding='utf-8') as f:
    json.dump(training_history, f, indent=2, ensure_ascii=False, default=str)

print(f'\n[Done] Training hoàn tất!')
print(f'       Model: {model_name}')
print(f'       Best Dev F1: {best_dev_F:.2f}%')
print(f'       History: {history_file}')
