# Datasets

The raw datasets are public and are **not** redistributed in this repository.
Download them from the original providers and place them as follows (paths can
be changed in `configs/*.yaml` via `data_path`).

| Dataset | Source | Expected file | Binary mapping (1 = harmful) |
|---|---|---|---|
| Kaggle Cyberbullying Classification (47,692 tweets, 6 classes) | Kaggle: `andrewmvd/cyberbullying-classification` | `data/raw/kaggle/cyberbullying_tweets.csv` | `religion`, `age`, `gender`, `ethnicity`, `other_cyberbullying` → 1; `not_cyberbullying` → 0 |
| Jigsaw Toxic Comment Classification | Kaggle competition `jigsaw-toxic-comment-classification-challenge` (`train.csv`) | `data/raw/jigsaw/train.csv` | any of `toxic, severe_toxic, obscene, threat, insult, identity_hate` → 1 |
| HateXplain (Twitter + Gab) | github.com/hate-alert/HateXplain (`Data/dataset.json`) | `data/raw/hatexplain/dataset.json` | majority label `hatespeech` or `offensive` → 1; `normal` → 0 |

```bash
# with the Kaggle CLI configured
kaggle datasets download -d andrewmvd/cyberbullying-classification -p data/raw/kaggle --unzip
kaggle competitions download -c jigsaw-toxic-comment-classification-challenge -p data/raw/jigsaw
unzip -o data/raw/jigsaw/jigsaw-toxic-comment-classification-challenge.zip -d data/raw/jigsaw
unzip -o data/raw/jigsaw/train.csv.zip -d data/raw/jigsaw
curl -L -o data/raw/hatexplain/dataset.json \
  https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/dataset.json
```

## Frozen splits used in the paper

| Dataset | Cleaned corpus | Train (federated) | Frozen test | Test positives |
|---|---:|---:|---:|---:|
| Kaggle CB | 43,932 (6,264 non-CB / 37,668 CB) | 35,145 | 8,787 | 7,534 (85.7 %) |
| Jigsaw | — | 127,524 (non-IID partition total) | 31,882 | 3,242 (10.2 %) |
| HateXplain | — | — | 4,029 | 2,399 (59.5 %) |

The test sets are stratified 80/20 splits with seed 42 (`fepp_cd.data.frozen_split`).
Counts come from the original runs (`results/original_plots/kaggle/class_distribution.png`,
`results/logs/*client_distribution.csv`, `results/predictions/*`). `scripts/train.py`
prints the counts it obtains; small differences indicate a different cleaning step
(the cleaning in `fepp_cd.data.clean_text` lower-cases, removes URLs, mentions,
`#` and non-alphanumeric characters, drops empty texts and exact duplicates).

## HateXplain fields in `results/predictions/hatexplain/predictions.csv`

`post_id, n_tokens, label_orig, binary_label, targets, rationale_pos, has_rationale,
n_annotators, label_agree, y_true, y_pred, y_prob`. The post text was removed;
recover it from `dataset.json` with `post_id`. `targets` lists the annotated target
communities (an empty list is reported as `unspecified` in the community analysis);
`rationale_pos` lists word positions marked by the human rationale annotators.
