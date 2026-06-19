# FEPP-CD

**Federated, Explainable, and Privacy-Preserving 
Cyberbullying Detection under Non-IID Social Media Data**

Muhammad Israr Ul Haq ,DR.Muhammd Muneer Umar ,  Dr. Amjad Mehmood
Institute of Computing, KUST Kohat, Pakistan

---

## Framework Components

- DistilBERT local classifier
- FedAvg with IID and Non-IID Dirichlet partitioning (α = 0.5)
- Secure Aggregation (SecAgg)
- Integrated Gradients token attribution
- Label-flipping adversarial analysis
- HateXplain rationale alignment and community fairness

---

## Key Results

| Condition | Accuracy | Macro F1 | TMR |
|---|---|---|---|
| FL IID + SecAgg (Kaggle) | 0.9286 | 0.8500 | 3.62% |
| FL Non-IID (Kaggle) | 0.9245 | 0.8233 | 1.94% |
| FL IID (Jigsaw) | 0.9625 | 0.9010 | 14.25% |
| FL Non-IID (Jigsaw) | 0.9606 | 0.8732 | 35.87% |
| HateXplain | 0.7607 | 0.7497 | 18.47% |
| Rationale F1 | — | 0.5683 | — |

---

## Scripts

| File | Description |
|---|---|
| `01_kaggle_all_experiments.py` | Centralised baseline, FL IID, Non-IID, SecAgg, Label Flip |
| `02_jigsaw_iid_noniid.py` | Jigsaw IID and Non-IID FL |
| `03_xai_hatexplain.py` | IG attribution, HateXplain FL, rationale alignment, community fairness |

---

## Datasets

- [Kaggle Cyberbullying Classification](https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification)
- [Jigsaw Toxic Comment Classification](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
- [HateXplain](https://github.com/hate-speech-and-offensive-language/HateXplain)
All experiments use seed = 42.
---

## Requirements
