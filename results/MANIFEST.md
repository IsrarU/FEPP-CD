# Results manifest

Provenance and SHA-256 checksums of every released artifact. The `Original file` column refers to the folder `Three Data Set results/` produced by the original Kaggle notebook runs (`israrul/mscs-experment`).

| File | Rows | SHA-256 | Original file |
|---|---|---|---|
| `results/predictions/kaggle/iid.csv` | 8,787 | `140bf95cdf1a35043e066472db536d43d9052e6e0835de3b717b0b60627fac98` | Kaggle Cyberbullying results/IID/predictions_iid.csv (identical to flip attck/security_flip000.csv) |
| `results/predictions/kaggle/noniid.csv` | 8,787 | `eb934d04488219b81e9f43d080a3c6fe1dc70ebb205ae1e6c18247406537f736` | Kaggle Cyberbullying results/non-IID/predictions_noniid.csv |
| `results/predictions/kaggle/flip_010.csv` | 8,787 | `99b4e482fc2e3ef6cf4deb1a9c8dca28a18d08c29f4d33baeb40f903d95214c6` | Kaggle Cyberbullying results/flip attck/security_flip010.csv |
| `results/predictions/kaggle/flip_020.csv` | 8,787 | `20c60307846e33ecff48823b3cbdde38d7ef35505cbe153b6722e886877a3861` | Kaggle Cyberbullying results/flip attck/security_flip020.csv |
| `results/predictions/kaggle/flip_030.csv` | 8,787 | `14a8c5e739bd2b5c03afe8c4e3e1bfa9974d996a36805f7d868f11b6b83e51a7` | Kaggle Cyberbullying results/flip attck/security_flip030.csv |
| `results/predictions/kaggle/centralized_supplementary.csv` | 8,787 | `cf12fa4518f1769676a3a666ff15b48c89a081a0e29c23b854b17297d317f8b8` | Kaggle Cyberbullying results/centralised_predictions.csv |
| `results/predictions/jigsaw/iid.csv` | 31,882 | `d1eafccbcef32226baf95c5279df9c4d76e9efe32e88c99acfbc8b0cd59f04c4` | jigswa Results/IID/jigsaw_iid_predictions.csv |
| `results/predictions/jigsaw/noniid.csv` | 31,882 | `09304b5dca4ebbe60127df769e456e863d7b1caf0a53b8e4daadc28b12337fe3` | jigswa Results/NON-IID/jigsaw_noniid_predictions.csv |
| `results/predictions/hatexplain/predictions.csv` | 4,029 | `1f85844afb65d2293760ba90484f339527aa6c1bd8b3e8326f8f231679019434` | hatexplain results/hatexplain_predictions.csv (columns text and tokens removed) |
| `results/logs/jigsaw_noniid_round_log.csv` | 5 | `506905022590a675ee231171b62a88908364e4f502ac7ee2b40c82a952cb0792` | jigswa Results/NON-IID/jigsaw_noniid_round_log.csv |
| `results/logs/jigsaw_noniid_client_distribution.csv` | 5 | `b32ab35ccaf0fcf48f73b61f4b8bf6eb130e4c59946f9f3c84287572b14e951e` | jigswa Results/NON-IID/jigsaw_noniid_client_distribution.csv |
| `results/logs/jigsaw_noniid_summary.csv` | 1 | `fc075832ac7b034705d882ca2cebe9eacea6865527b6cfa103956ce5bc4e0b87` | jigswa Results/NON-IID/jigsaw_noniid_summary.csv |
| `results/logs/kaggle_noniid_client_distribution.csv` | 5 | `9c2e16e9c667dac09131dc445a254a9d480121b1954a0730d18176066e2cba17` | Transcribed from the notebook partition printout (paper Table 3); consistent with original_plots/kaggle/noniid_client_distribution.png and class_distribution.png (5,011 + 1,253 = 6,264 non-CB; 30,134 + 7,534 = 37,668 CB) |
| `results/xai/rationale_alignment.csv` | 100 | `e04a79c19985ba9b119038b36465ab67ca0b86c9f1c53d794caf0af18db63ffb` | hatexplain results/rationale_alignment.csv |
| `results/xai/global_token_importance.csv` | 5 | `f5b29fa1a52f98b6aac68bcdf280f6c9c2a81cec923453d9a361bf6b3b8e7530` | xai_global_importance.csv (paper source bundle) |
| `results/reports/community_sensitivity_original.csv` | 12 | `0bc79c4d7fcf96e5742d9172efcb1423d2c42d45193da01b98f5b882f4167ef4` | hatexplain results/community_sensitivity.csv |
| `results/reports/kaggle_iid_classification_report.csv` | 5 | `94b508395bf66a77c32517289d6e603099265101101080d5f852c4b4d4e7611b` | Kaggle Cyberbullying results/IID/predictions_iid_classification_report.csv |
| `results/reports/kaggle_noniid_classification_report.csv` | 5 | `b4049f668e055175ac38de74d4b92380dc33a3ab2a7e0842830b9d755fabc875` | Kaggle Cyberbullying results/non-IID/predictions_noniid_classification_report.csv |
| `results/reports/master_metrics_original.csv` | 5 | `87ebb148ae87e77aa266274b4f64ccdadf4f3ad497dcc14aeb810168730dcc95` | master_metrics.csv (paper source bundle) |

Model checkpoints (`predictions_iid_model.pt`, ~265 MB) exceed GitHub's 100 MB file limit and are not included; attach them to a GitHub Release or a Zenodo record if they are to be shared.
