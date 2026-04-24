import os, json
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

from transformers import (
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset, DatasetDict

print(f'PyTorch : {torch.__version__}')
print(f'GPU     : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU     : {torch.cuda.get_device_name(0)}')
    print(f'VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

PROJECT_DIR = Path("/home/onyxia/work/Named_Entity_Extraction_Archelec_Corpus/projet_archelec")
BIO_DIR     = PROJECT_DIR / "data" / "bio_distantsup"
MODELS_DIR  = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "data" / "results" / "CamemBERT"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

with open(BIO_DIR / "label2id.json") as f:
    LABEL2ID = json.load(f)
ID2LABEL   = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)
print(f'Labels ({NUM_LABELS}) : {LABEL2ID}')

def charger_split(chemin):
    with open(chemin, encoding='utf-8') as f:
        data = json.load(f)
    return Dataset.from_list([
        {'input_ids': d['input_ids'], 'ner_tags': d['ner_tags'], 'id': str(d['id']), 'annee': str(d['annee'])}
        for d in data
    ])

dataset = DatasetDict({
    'train': charger_split(BIO_DIR / 'train.json'),
    'val':   charger_split(BIO_DIR / 'val.json'),
    'test':  charger_split(BIO_DIR / 'test.json'),
})
print(f'Train : {len(dataset["train"])} | Val : {len(dataset["val"])} | Test : {len(dataset["test"])}')

MAX_LENGTH   = 512
PAD_TOKEN_ID = 1

def preprocess_batch(examples):
    batch_input_ids, batch_attention_mask, batch_labels = [], [], []
    for input_ids, ner_tags in zip(examples['input_ids'], examples['ner_tags']):
        input_ids = input_ids[:MAX_LENGTH]
        ner_tags  = ner_tags[:MAX_LENGTH]
        pad_len        = MAX_LENGTH - len(input_ids)
        attention_mask = [1] * len(input_ids) + [0] * pad_len
        input_ids      = input_ids + [PAD_TOKEN_ID] * pad_len
        labels         = ner_tags  + [-100] * pad_len
        batch_input_ids.append(input_ids)
        batch_attention_mask.append(attention_mask)
        batch_labels.append(labels)
    return {'input_ids': batch_input_ids, 'attention_mask': batch_attention_mask, 'labels': batch_labels}

print('Preprocessing...')
dataset_proc = dataset.map(preprocess_batch, batched=True, remove_columns=['ner_tags', 'id', 'annee'])
dataset_proc.set_format('torch')
print('Dataset pret')

def extraire_spans(tags):
    spans = set()
    i = 0
    while i < len(tags):
        tag = tags[i] if isinstance(tags[i], str) else ID2LABEL[tags[i]]
        if tag.startswith('B-'):
            entite = tag[2:]
            debut  = i
            i += 1
            while i < len(tags):
                t = tags[i] if isinstance(tags[i], str) else ID2LABEL[tags[i]]
                if t == f'I-{entite}':
                    i += 1
                else:
                    break
            spans.add((entite, debut, i))
        else:
            i += 1
    return spans

def compute_metrics(pred):
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=2)
    stats        = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    stats_global = {'tp': 0, 'fp': 0, 'fn': 0}
    for pred_seq, label_seq in zip(predictions, labels):
        pred_clean  = [ID2LABEL[p] for p, l in zip(pred_seq, label_seq) if l != -100]
        label_clean = [ID2LABEL[l] for l in label_seq if l != -100]
        spans_pred  = extraire_spans(pred_clean)
        spans_gold  = extraire_spans(label_clean)
        for span in spans_pred:
            if span in spans_gold:
                stats[span[0]]['tp'] += 1; stats_global['tp'] += 1
            else:
                stats[span[0]]['fp'] += 1; stats_global['fp'] += 1
        for span in spans_gold:
            if span not in spans_pred:
                stats[span[0]]['fn'] += 1; stats_global['fn'] += 1
    def prf(tp, fp, fn):
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2*p*r / (p+r)  if (p  + r) > 0 else 0
        return p*100, r*100, f1*100
    res = {}
    for entite in ['PER', 'ORG', 'LOC', 'MISC']:
        s = stats[entite]
        p, r, f1 = prf(s['tp'], s['fp'], s['fn'])
        res[f'f1_{entite}'] = round(f1, 2); res[f'precision_{entite}'] = round(p, 2); res[f'recall_{entite}'] = round(r, 2)
    p_g, r_g, f1_g = prf(stats_global['tp'], stats_global['fp'], stats_global['fn'])
    res['f1_global'] = round(f1_g, 2); res['precision_global'] = round(p_g, 2); res['recall_global'] = round(r_g, 2)
    return res

MODEL_NAME = 'camembert-base'
USE_FP16   = torch.cuda.is_available()
print(f'Chargement {MODEL_NAME}... fp16={USE_FP16}')

model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME, num_labels=NUM_LABELS, id2label=ID2LABEL, label2id=LABEL2ID, ignore_mismatched_sizes=True
)

args = TrainingArguments(
    output_dir                  = str(MODELS_DIR / 'stage1_checkpoints'),
    num_train_epochs            = 5,
    per_device_train_batch_size = 16,
    per_device_eval_batch_size  = 32,
    learning_rate               = 2e-5,
    weight_decay                = 0.01,
    warmup_ratio                = 0.1,
    eval_strategy               = 'epoch',
    save_strategy               = 'epoch',
    load_best_model_at_end      = True,
    metric_for_best_model       = 'f1_global',
    greater_is_better           = True,
    logging_steps               = 200,
    fp16                        = USE_FP16,
    report_to                   = 'none',
    save_total_limit            = 1,
)

trainer = Trainer(
    model=model, args=args,
    train_dataset=dataset_proc['train'], eval_dataset=dataset_proc['val'],
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

print('Entrainement Stage 1...')
trainer.train()
print('Entrainement termine !')

print('\n=== EVALUATION TEST ===')
results = trainer.evaluate(dataset_proc['test'])
print(f'{"":<10} {"Precision":>12} {"Recall":>10} {"F1":>10}')
print('-' * 45)
for entite in ['PER', 'ORG', 'LOC', 'MISC']:
    p  = results.get(f'eval_precision_{entite}', 0)
    r  = results.get(f'eval_recall_{entite}', 0)
    f1 = results.get(f'eval_f1_{entite}', 0)
    print(f'{entite:<10} {p:>11.2f}% {r:>9.2f}% {f1:>9.2f}%')
print('-' * 45)
print(f'{"GLOBAL":<10} {results.get("eval_precision_global", 0):>11.2f}% {results.get("eval_recall_global", 0):>9.2f}% {results.get("eval_f1_global", 0):>9.2f}%')

with open(RESULTS_DIR / 'stage1_results.json', 'w') as f:
    json.dump(results, f, indent=2)

stage1_dir = MODELS_DIR / 'stage1_model'
trainer.save_model(str(stage1_dir))
print(f'\nModele sauvegarde dans {stage1_dir}')
print('ETAPE SUIVANTE : stage2_manual_sspcloud.ipynb')
