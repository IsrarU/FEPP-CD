# Method summary (code map)

| Paper element | Code |
|---|---|
| Class-weighted BCE, ω⁺ = n_neg/n_pos | `fepp_cd.data.positive_class_weight`, `fepp_cd.federated.local_train` |
| DistilBERT classifier p = σ(W·Dropout(h_[CLS]) + b) | `fepp_cd.model.DistilBertClassifier` |
| Dirichlet non-IID partition, α = 0.5 | `fepp_cd.partition.dirichlet_partition` |
| FedAvg w^{t+1} = Σ (n_k/n) w_k^{t+1} | `fepp_cd.federated.fedavg_aggregate` |
| SecAgg masking ũ_i = u_i + m_i (Proposition 1) | `fepp_cd.secagg`, `fepp_cd.federated.secagg_aggregate` |
| Communication accounting (Eq. comm) | `fepp_cd.comm.comm_cost` |
| Label flipping N_flip = ⌊γ·|D₀⁺|⌋ | `fepp_cd.attacks.flip_labels` |
| Integrated Gradients (S = 50), L2 token scores, WordPiece merge | `fepp_cd.explain.IGExplainer` |
| Rationale P/R/F1 with R_m = top-k words, k = |R_h| (*) | `fepp_cd.explain.rationale_alignment`, `fepp_cd.metrics.rationale_prf` |
| TMR, FAR, SUS | `fepp_cd.metrics.tmr_far`, `safety_utility_score` |
| Calibrated operating point τ*(t) | `fepp_cd.metrics.calibrated_operating_point` |
| ECE (10 bins) | `fepp_cd.metrics.expected_calibration_error` |
| Bootstrap CIs, McNemar | `fepp_cd.metrics.bootstrap_ci`, `mcnemar` |
| Community sensitivity | `fepp_cd.metrics.community_sensitivity` |

(*) The paper does not state how the model rationale R_m is selected. This
repository uses the k = |R_h| highest-scoring words, which makes token-set precision
and recall equal up to WordPiece/word-merging effects; this is consistent with the
near-identical precision (0.5681) and recall (0.5686) in the released
`results/xai/rationale_alignment.csv`, but it is an implementation choice of this
repository rather than a documented setting of the original runs. The released file
also does not record which 100 posts were evaluated; `scripts/explain.py` samples
100 rationale-bearing test posts with seed 42 and stores their `post_id`.
