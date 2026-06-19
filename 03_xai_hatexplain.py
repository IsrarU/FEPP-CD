"""
FEPP-CD: Federated, Explainable, and Privacy-Preserving
Cyberbullying Detection Framework
=======================================================
Script 3 of 3: XAI and HateXplain Validation
Experiments:
  - Integrated Gradients on Kaggle IID model
  - HateXplain: FL IID classification
  - HateXplain: Rationale alignment (IG vs human)
  - HateXplain: Community sensitivity analysis

Author  : Muhammad Israr Ul Haq
          Institute of Computing, KUST Kohat
Note    : Run after Script 1 (needs iid_predictions.csv)
          OR run standalone (retrains IID model)
"""

import os, copy, json, random, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    roc_auc_score, average_precision_score,
    precision_score, recall_score)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════
KAGGLE_DATA  = "/kaggle/input/datasets/israrul/hatexplain-for-xai-validation/dataset.json"
HATEXPLAIN_JSON = '/kaggle/working/dataset.json'
OUT          = '/kaggle/working/xai_outputs'
os.makedirs(OUT, exist_ok=True)

SEED = 42
def set_seed(s=SEED):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(s)
set_seed()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device : {DEVICE}')

CFG = {
    'model_name'   : 'distilbert-base-uncased',
    'max_len'      : 128,
    'dropout'      : 0.3,
    'n_clients'    : 5,
    'n_rounds'     : 10,
    'local_epochs' : 3,
    'batch_size'   : 32,
    'lr'           : 2e-5,
    'clip_norm'    : 1.0,
    'test_size'    : 0.20,
    'ig_steps'     : 50,
    'seed'         : SEED,
}

# ═══════════════════════════════════════════════════════════════
# DOWNLOAD HATEXPLAIN IF NEEDED
# ═══════════════════════════════════════════════════════════════
if not os.path.exists(HATEXPLAIN_JSON):
    import subprocess
    print('[DOWNLOAD] HateXplain dataset.json...')
    subprocess.run([
        'wget', '-q',
        'https://raw.githubusercontent.com/hate-speech-and-offensive-language/'
        'HateXplain/master/Data/dataset.json',
        '-O', HATEXPLAIN_JSON])
    print(f'  Downloaded: {os.path.getsize(HATEXPLAIN_JSON)/1024:.0f} KB')

# ═══════════════════════════════════════════════════════════════
# MODEL CLASS
# ═══════════════════════════════════════════════════════════════
tokenizer = DistilBertTokenizerFast.from_pretrained(CFG['model_name'])

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.enc = tokenizer(list(texts), truncation=True,
                             padding='max_length',
                             max_length=CFG['max_len'],
                             return_tensors='pt')
        self.labels = torch.tensor(list(labels), dtype=torch.long)
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return {'input_ids'     : self.enc['input_ids'][i],
                'attention_mask': self.enc['attention_mask'][i],
                'labels'        : self.labels[i]}

class DistilBERTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.distilbert  = DistilBertModel.from_pretrained(CFG['model_name'])
        self.dropout     = nn.Dropout(CFG['dropout'])
        self.classifier  = nn.Linear(768, 2)
    def forward(self, input_ids, attention_mask):
        out = self.distilbert(input_ids=input_ids,
                              attention_mask=attention_mask)
        return self.classifier(self.dropout(
            out.last_hidden_state[:, 0, :]))

def evaluate(model, loader):
    model.eval(); yt, yp, ypr = [], [], []
    with torch.no_grad():
        for batch in loader:
            ids    = batch['input_ids'].to(DEVICE)
            mask   = batch['attention_mask'].to(DEVICE)
            logits = model(ids, mask)
            ypr.extend(F.softmax(logits,dim=1)[:,1].cpu().numpy())
            yp.extend(logits.argmax(dim=1).cpu().numpy())
            yt.extend(batch['labels'].numpy())
    return np.array(yt), np.array(yp), np.array(ypr)

def metrics(yt, yp, ypr, label=''):
    cm = confusion_matrix(yt, yp, labels=[0,1])
    tn,fp,fn,tp = cm.ravel()
    ok = abs((tn+tp)/len(yt) - accuracy_score(yt,yp)) < 1e-6
    return {
        'label'      : label,
        'accuracy'   : round(accuracy_score(yt,yp), 4),
        'macro_f1'   : round(f1_score(yt,yp,average='macro',zero_division=0), 4),
        'roc_auc'    : round(roc_auc_score(yt,ypr), 4),
        'auprc'      : round(average_precision_score(yt,ypr), 4),
        'prec_0'     : round(precision_score(yt,yp,pos_label=0,zero_division=0), 4),
        'rec_0'      : round(recall_score(yt,yp,pos_label=0,zero_division=0), 4),
        'prec_1'     : round(precision_score(yt,yp,pos_label=1,zero_division=0), 4),
        'rec_1'      : round(recall_score(yt,yp,pos_label=1,zero_division=0), 4),
        'TN':int(tn),'FP':int(fp),'FN':int(fn),'TP':int(tp),
        'TMR': round(fn/(fn+tp)*100, 2) if (fn+tp)>0 else 0,
        'FAR': round(fp/(fp+tn)*100, 2) if (fp+tn)>0 else 0,
        'consistent': '✓' if ok else '⚠',
    }

# FL helpers
def local_train(model, loader, opt, criterion, epochs):
    model.train()
    for _ in range(epochs):
        for batch in loader:
            ids  = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            lbls = batch['labels'].to(DEVICE)
            opt.zero_grad()
            loss = criterion(model(ids, mask), lbls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG['clip_norm'])
            opt.step()

def fedavg(gm, cms, sizes):
    total = sum(sizes)
    gs    = gm.state_dict()
    for key in gs:
        w = torch.zeros_like(gs[key], dtype=torch.float)
        for m, s in zip(cms, sizes):
            w += (s/total)*m.state_dict()[key].float()
        gs[key] = w.to(gs[key].dtype)
    gm.load_state_dict(gs); return gm

def iid_partition(df, n=5):
    set_seed(SEED)
    s = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    k = len(s)//n
    return [s.iloc[i*k:(i+1)*k if i<n-1
            else len(s)].reset_index(drop=True) for i in range(n)]

# ═══════════════════════════════════════════════════════════════
# PART A — INTEGRATED GRADIENTS ON KAGGLE IID MODEL
# ═══════════════════════════════════════════════════════════════
print('\n' + '='*56)
print('  PART A: INTEGRATED GRADIENTS — KAGGLE IID')
print('='*56)

# Load Kaggle data and train IID model
print('\n[DATA] Loading Kaggle...')
df = pd.read_csv(KAGGLE_DATA)
df.columns = ['text', 'label_orig']
df = df.dropna(subset=['text','label_orig'])
df['text'] = df['text'].astype(str).str.strip()
df = df[df['text'].apply(lambda x: len(x.split()) >= 3)]
df['binary_label'] = (df['label_orig'] != 'not_cyberbullying').astype(int)
df = df.drop_duplicates(subset=['text','binary_label'])
conflicts = df[df.duplicated(subset=['text'],keep=False)]['text'].unique()
df = df[~df['text'].isin(conflicts)].reset_index(drop=True)

idx = np.arange(len(df))
train_idx, test_idx = train_test_split(
    idx, test_size=CFG['test_size'],
    random_state=CFG['seed'],
    stratify=df['binary_label'].values)
train_df_k = df.iloc[train_idx].reset_index(drop=True)
test_df_k  = df.iloc[test_idx].reset_index(drop=True)

vc2 = train_df_k['binary_label'].value_counts()
pos_weight_k = torch.tensor([vc2[0]/vc2[1]],
                              dtype=torch.float).to(DEVICE)
criterion_k  = nn.CrossEntropyLoss(
    weight=torch.tensor([1.0, pos_weight_k.item()]).to(DEVICE))

test_ds_k     = TextDataset(test_df_k['text'], test_df_k['binary_label'])
test_loader_k = DataLoader(test_ds_k, batch_size=CFG['batch_size'],
                            shuffle=False)

parts_k = iid_partition(train_df_k)
sizes_k = [len(p) for p in parts_k]
client_ds_k = [TextDataset(p['text'], p['binary_label']) for p in parts_k]

print('[TRAIN] Running IID FL on Kaggle (for XAI)...')
set_seed(SEED)
iid_model = DistilBERTClassifier().to(DEVICE)

for rnd in range(1, CFG['n_rounds']+1):
    cms = []
    for i in range(CFG['n_clients']):
        local = DistilBERTClassifier().to(DEVICE)
        local.load_state_dict(copy.deepcopy(iid_model.state_dict()))
        loader = DataLoader(client_ds_k[i], batch_size=CFG['batch_size'],
                            shuffle=True)
        opt = torch.optim.AdamW(local.parameters(), lr=CFG['lr'])
        local_train(local, loader, opt, criterion_k, CFG['local_epochs'])
        cms.append(local)
    iid_model = fedavg(iid_model, cms, sizes_k)
    yt, yp, _ = evaluate(iid_model, test_loader_k)
    acc = accuracy_score(yt, yp)
    mf1 = f1_score(yt, yp, average='macro', zero_division=0)
    print(f'  Round {rnd:2d}/{CFG["n_rounds"]} | Acc={acc:.4f} | MacroF1={mf1:.4f}')

# IG COMPUTATION
SPECIAL = {'[CLS]', '[SEP]', '[PAD]'}

def compute_ig(model, text, pred_class, m=50):
    model.eval()
    enc  = tokenizer(text, truncation=True, padding='max_length',
                     max_length=CFG['max_len'], return_tensors='pt')
    ids  = enc['input_ids'].to(DEVICE)
    mask = enc['attention_mask'].to(DEVICE)
    toks = tokenizer.convert_ids_to_tokens(ids[0])

    embed = model.distilbert.embeddings.word_embeddings
    inp_e = embed(ids).detach()
    base  = torch.zeros_like(inp_e)
    ig    = torch.zeros_like(inp_e)
    attn  = (1.0 - mask.unsqueeze(1).unsqueeze(2).float()) * (-10000.0)

    for k in range(m+1):
        alpha  = k/m
        interp = (base + alpha*(inp_e-base)).requires_grad_(True)
        pos_ids= torch.arange(ids.shape[1], device=DEVICE).unsqueeze(0)
        pos_e  = model.distilbert.embeddings.position_embeddings(pos_ids)
        comb   = model.distilbert.embeddings.LayerNorm(interp+pos_e)
        comb   = model.distilbert.embeddings.dropout(comb)
        t_out  = model.distilbert.transformer(comb, attn)[-1]
        logit  = model.classifier(model.dropout(t_out[:,0,:]))[0, pred_class]
        model.zero_grad(); logit.backward()
        ig    += interp.grad.detach()

    attr   = ((inp_e-base)*ig/(m+1)).squeeze(0)
    scores = attr.norm(dim=-1).cpu().numpy()
    active = mask[0].cpu().numpy()

    word_toks, word_scores = [], []
    for i, tok in enumerate(toks):
        if tok in SPECIAL or active[i] == 0: continue
        if tok.startswith('##'):
            if word_toks:
                word_scores[-1] = max(word_scores[-1], float(scores[i]))
        else:
            word_toks.append(tok)
            word_scores.append(float(scores[i]))

    if not word_scores: return [], []
    mx = max(word_scores)
    if mx > 0: word_scores = [s/mx for s in word_scores]
    return word_toks, word_scores

# Sample IG analysis — 6 representative cases
print('\n[XAI] Computing IG attributions...')
test_sample = test_df_k.copy()
yt_all, yp_all, ypr_all = evaluate(iid_model, test_loader_k)
test_sample['y_true'] = yt_all
test_sample['y_pred'] = yp_all
test_sample['y_prob'] = ypr_all

# Select 3 correct harmful, 3 correct non-harmful, 2 FP, 2 FN
correct_harmful  = test_sample[(test_sample.y_true==1) &
                               (test_sample.y_pred==1)].head(3)
correct_safe     = test_sample[(test_sample.y_true==0) &
                               (test_sample.y_pred==0)].head(3)
false_positives  = test_sample[(test_sample.y_true==0) &
                               (test_sample.y_pred==1)].head(2)
false_negatives  = test_sample[(test_sample.y_true==1) &
                               (test_sample.y_pred==0)].head(2)
samples = pd.concat([correct_harmful, correct_safe,
                     false_positives, false_negatives])

ig_results = []
for _, row in samples.iterrows():
    pred_class = int(row['y_pred'])
    toks, scores = compute_ig(iid_model, row['text'], pred_class)
    ig_results.append({
        'text'       : row['text'][:100],
        'y_true'     : int(row['y_true']),
        'y_pred'     : int(row['y_pred']),
        'y_prob'     : round(float(row['y_prob']), 4),
        'case_type'  : ('TP' if row['y_true']==1 and row['y_pred']==1
                        else 'TN' if row['y_true']==0 and row['y_pred']==0
                        else 'FP' if row['y_true']==0 and row['y_pred']==1
                        else 'FN'),
        'top_tokens' : str(sorted(zip(toks,scores),
                               key=lambda x:-x[1])[:5]) if toks else '[]',
    })
pd.DataFrame(ig_results).to_csv(f'{OUT}/ig_sample_analysis.csv', index=False)
print('  [SAVED] ig_sample_analysis.csv')

# Global token importance
print('[XAI] Computing global token importance...')
harmful_subset = test_sample[test_sample.y_true==1].head(200)
token_scores   = {}
for _, row in harmful_subset.iterrows():
    toks, scores = compute_ig(iid_model, row['text'], 1)
    for tok, score in zip(toks, scores):
        if tok not in token_scores:
            token_scores[tok] = []
        token_scores[tok].append(score)

stopwords = {'the','a','an','is','it','in','of','to','and','or',
             'for','on','at','by','with','this','that','i','you',
             'he','she','we','they','was','are','be','has','have',
             '<user>','<number>','<percent>','.', ',', '!', '?'}
global_imp = {t: np.mean(v) for t,v in token_scores.items()
              if t not in stopwords and len(v) >= 3}
top20 = sorted(global_imp.items(), key=lambda x:-x[1])[:20]

pd.DataFrame(top20, columns=['token','mean_attr'])\
  .to_csv(f'{OUT}/global_token_importance.csv', index=False)

# Plot global importance
fig, ax = plt.subplots(figsize=(8, 6))
tokens_ = [t for t,_ in top20[:15]]
scores_ = [s for _,s in top20[:15]]
ax.barh(tokens_[::-1], scores_[::-1], color='#C62828')
ax.set(xlabel='Mean Normalised IG Score',
       title='Top-15 Globally Important Tokens (Harmful Class)')
plt.tight_layout()
plt.savefig(f'{OUT}/global_token_importance.png', dpi=150)
plt.close()
print('  [SAVED] global_token_importance.png')

# ═══════════════════════════════════════════════════════════════
# PART B — HATEXPLAIN: FL IID + RATIONALE ALIGNMENT + COMMUNITY
# ═══════════════════════════════════════════════════════════════
print('\n' + '='*56)
print('  PART B: HATEXPLAIN VALIDATION')
print('='*56)

print('\n[DATA] Loading HateXplain...')
with open(HATEXPLAIN_JSON, 'r', encoding='utf-8') as f:
    hx_data = json.load(f)

records = []
for post_id, post in hx_data.items():
    tokens   = post['post_tokens']
    text     = ' '.join(tokens)
    labels   = [ann['label'] for ann in post['annotators']]
    majority = Counter(labels).most_common(1)[0][0]
    binary   = 0 if majority == 'normal' else 1
    all_targets = []
    for ann in post['annotators']:
        for t in ann.get('target', []):
            if t != 'None': all_targets.append(t)
    targets = list(set(all_targets))
    rationales = post.get('rationales', [])
    rationale_union = set()
    if rationales:
        for rat in rationales:
            for i, val in enumerate(rat):
                if val == 1: rationale_union.add(i)
    records.append({
        'post_id'       : post_id,
        'text'          : text,
        'tokens'        : tokens,
        'label_orig'    : majority,
        'binary_label'  : binary,
        'targets'       : targets,
        'rationale_pos' : list(rationale_union),
        'has_rationale' : len(rationale_union) > 0,
    })

hx_df = pd.DataFrame(records)
hx_df = hx_df[hx_df['text'].apply(lambda x: len(x.split())>=3)]
hx_df = hx_df.reset_index(drop=True)
vc_hx = hx_df['binary_label'].value_counts().sort_index()
print(f'  HateXplain: {len(hx_df):,} posts | '
      f'Normal:{vc_hx[0]:,}  Harmful:{vc_hx[1]:,}')

idx_hx = np.arange(len(hx_df))
tr_idx, te_idx = train_test_split(
    idx_hx, test_size=CFG['test_size'],
    random_state=CFG['seed'],
    stratify=hx_df['binary_label'].values)
hx_train = hx_df.iloc[tr_idx].reset_index(drop=True)
hx_test  = hx_df.iloc[te_idx].reset_index(drop=True)

vc_hx2 = hx_train['binary_label'].value_counts()
pos_weight_hx = torch.tensor([vc_hx2[0]/vc_hx2[1]],
                               dtype=torch.float).to(DEVICE)
criterion_hx  = nn.CrossEntropyLoss(
    weight=torch.tensor([1.0, pos_weight_hx.item()]).to(DEVICE))

hx_test_ds  = TextDataset(hx_test['text'], hx_test['binary_label'])
hx_test_ldr = DataLoader(hx_test_ds, batch_size=CFG['batch_size'],
                          shuffle=False)

# IID FL on HateXplain
print('\n[TRAIN] HateXplain IID FL (10 rounds)...')
parts_hx  = iid_partition(hx_train)
sizes_hx  = [len(p) for p in parts_hx]
client_hx = [TextDataset(p['text'], p['binary_label']) for p in parts_hx]

set_seed(SEED)
hx_model = DistilBERTClassifier().to(DEVICE)

for rnd in range(1, CFG['n_rounds']+1):
    cms = []
    for i in range(CFG['n_clients']):
        local = DistilBERTClassifier().to(DEVICE)
        local.load_state_dict(copy.deepcopy(hx_model.state_dict()))
        loader = DataLoader(client_hx[i], batch_size=CFG['batch_size'],
                            shuffle=True)
        opt = torch.optim.AdamW(local.parameters(), lr=CFG['lr'])
        local_train(local, loader, opt, criterion_hx, CFG['local_epochs'])
        cms.append(local)
    hx_model = fedavg(hx_model, cms, sizes_hx)
    yt, yp, _ = evaluate(hx_model, hx_test_ldr)
    acc = accuracy_score(yt, yp)
    mf1 = f1_score(yt, yp, average='macro', zero_division=0)
    print(f'  Round {rnd:2d}/{CFG["n_rounds"]} | Acc={acc:.4f} | MacroF1={mf1:.4f}')

yt_hx, yp_hx, ypr_hx = evaluate(hx_model, hx_test_ldr)
d_hx = metrics(yt_hx, yp_hx, ypr_hx, 'HateXplain IID')
print(f'\n  HateXplain Results:')
print(f'  Accuracy: {d_hx["accuracy"]} | MacroF1: {d_hx["macro_f1"]}')
print(f'  AUC: {d_hx["roc_auc"]} | AUPRC: {d_hx["auprc"]}')
print(f'  TMR: {d_hx["TMR"]}% | FAR: {d_hx["FAR"]}%')
print(f'  {d_hx["consistent"]} CONSISTENT')

# Save predictions with metadata
hx_pred_df = hx_test.copy()
hx_pred_df['y_true'] = yt_hx
hx_pred_df['y_pred'] = yp_hx
hx_pred_df['y_prob'] = ypr_hx
hx_pred_df.to_csv(f'{OUT}/hatexplain_predictions.csv', index=False)

# EXPERIMENT B — RATIONALE ALIGNMENT
print('\n[XAI] Rationale alignment (IG vs human)...')
has_rat = hx_pred_df[
    (hx_pred_df['binary_label']==1) &
    (hx_pred_df['rationale_pos'].apply(
        lambda x: len(x)>0 if isinstance(x,list) else False))
].head(100)
print(f'  Evaluating on {len(has_rat)} samples...')

rat_f1, rat_pre, rat_rec = [], [], []
for _, row in has_rat.iterrows():
    rat_pos    = set(row['rationale_pos']) if isinstance(
        row['rationale_pos'], list) else set()
    if not rat_pos: continue
    toks, scores = compute_ig(hx_model, row['text'], int(row['y_pred']))
    if not scores: continue
    k       = max(1, len(rat_pos))
    top_idx = set(np.argsort(scores)[::-1][:k])
    human   = {p for p in rat_pos if p < len(toks)}
    if not human: continue
    tp_r = len(top_idx & human)
    fp_r = len(top_idx - human)
    fn_r = len(human - top_idx)
    pre  = tp_r/(tp_r+fp_r) if (tp_r+fp_r)>0 else 0
    rec  = tp_r/(tp_r+fn_r) if (tp_r+fn_r)>0 else 0
    f1_r = 2*pre*rec/(pre+rec) if (pre+rec)>0 else 0
    rat_f1.append(f1_r); rat_pre.append(pre); rat_rec.append(rec)

rat_results = {
    'n_samples'          : len(rat_f1),
    'rationale_f1'       : round(np.mean(rat_f1), 4),
    'rationale_precision': round(np.mean(rat_pre), 4),
    'rationale_recall'   : round(np.mean(rat_rec), 4),
}
print(f'\n  RATIONALE ALIGNMENT — PERMANENT RECORD')
print(f'  Samples   : {rat_results["n_samples"]}')
print(f'  F1        : {rat_results["rationale_f1"]}')
print(f'  Precision : {rat_results["rationale_precision"]}')
print(f'  Recall    : {rat_results["rationale_recall"]}')

pd.DataFrame([rat_results])\
  .to_csv(f'{OUT}/rationale_alignment_summary.csv', index=False)
pd.DataFrame({'f1':rat_f1,'precision':rat_pre,'recall':rat_rec})\
  .to_csv(f'{OUT}/rationale_alignment_percase.csv', index=False)

# EXPERIMENT C — COMMUNITY SENSITIVITY
print('\n[FAIRNESS] Community sensitivity analysis...')
rows_exp = []
for _, row in hx_pred_df.iterrows():
    targets = row['targets'] if isinstance(row['targets'],list) else []
    if not targets: targets = ['Unspecified']
    for t in targets:
        r = row.to_dict(); r['community'] = t
        rows_exp.append(r)
exploded = pd.DataFrame(rows_exp)

comm_counts = exploded['community'].value_counts()
valid_comms = comm_counts[comm_counts >= 20].index.tolist()

comm_results = []
print(f'\n  {"Community":<22} | {"N":>5} | {"F1":>6} | '
      f'{"Miss%":>7} | {"FA%":>7}')
print(f'  ' + '─'*56)

for comm in valid_comms[:12]:
    sub  = exploded[exploded['community']==comm]
    yt_c = sub['y_true'].values
    yp_c = sub['y_pred'].values
    if len(np.unique(yt_c)) < 2: continue
    cm_c = confusion_matrix(yt_c, yp_c, labels=[0,1])
    tn_c,fp_c,fn_c,tp_c = cm_c.ravel()
    miss_c = fn_c/(fn_c+tp_c)*100 if (fn_c+tp_c)>0 else 0
    fa_c   = fp_c/(fp_c+tn_c)*100 if (fp_c+tn_c)>0 else 0
    f1_c   = f1_score(yt_c, yp_c, average='macro', zero_division=0)
    print(f'  {comm:<22} | {len(sub):>5} | {f1_c:.4f} | '
          f'{miss_c:>6.2f}% | {fa_c:>6.2f}%')
    comm_results.append({
        'community':comm, 'n':len(sub),
        'macro_f1':round(f1_c,4),
        'miss_rate':round(miss_c,2),
        'false_alarm':round(fa_c,2),
    })

comm_df = pd.DataFrame(comm_results)
comm_df.to_csv(f'{OUT}/community_sensitivity.csv', index=False)

# CFI
f1_vals = comm_df['macro_f1'].values
cfi = 1 - (np.std(f1_vals)/np.mean(f1_vals))
print(f'\n  F1 range: {f1_vals.min():.4f} – {f1_vals.max():.4f}')
print(f'  CFI     : {cfi:.4f}')

# Community plot
fig, ax = plt.subplots(figsize=(8, 6))
comm_sorted = comm_df.sort_values('macro_f1', ascending=True)
ax.barh(comm_sorted['community'], comm_sorted['macro_f1'],
        color='#1565C0', alpha=0.8)
ax.axvline(x=f1_vals.mean(), color='red', linestyle='--',
           label=f'Mean = {f1_vals.mean():.3f}')
ax.set(xlabel='Macro F1', title='Community-Level Macro F1 (HateXplain)')
ax.legend(); ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT}/community_f1.png', dpi=150)
plt.close()

# ═══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════
print('\n' + '='*56)
print('  COMPLETE XAI SUMMARY — PERMANENT RECORD')
print('='*56)
print(f'  [A] Kaggle IID IG:')
print(f'      Top tokens computed on {len(ig_results)} samples')
print(f'      Global importance: {len(top20)} tokens ranked')
print(f'  [B] HateXplain FL:')
print(f'      Accuracy  : {d_hx["accuracy"]}')
print(f'      Macro F1  : {d_hx["macro_f1"]}')
print(f'      AUC       : {d_hx["roc_auc"]}')
print(f'      TMR       : {d_hx["TMR"]}% | FAR: {d_hx["FAR"]}%')
print(f'  [C] Rationale F1: {rat_results["rationale_f1"]}')
print(f'      Precision   : {rat_results["rationale_precision"]}')
print(f'      Recall      : {rat_results["rationale_recall"]}')
print(f'  [D] Communities : {len(comm_results)} evaluated')
print(f'      F1 range    : {f1_vals.min():.4f} – {f1_vals.max():.4f}')
print(f'      CFI         : {cfi:.4f}')

shutil.make_archive('/kaggle/working/XAI_HATEXPLAIN', 'zip', OUT)
print(f'\n✓ Download XAI_HATEXPLAIN.zip')
print('\n[FILES]')
for f in sorted(os.listdir(OUT)):
    print(f'  {f:<50} {os.path.getsize(f"{OUT}/{f}")/1024:>7.1f} KB')