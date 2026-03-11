# CKDu analysis suite v4 - usage notes

This updated script adds:
- actual feature names instead of F1..F30
- PCA-guided fusion modeling
- SHAP and LIME explainability
- class-imbalance handling in the fusion workflow
- `all` mode to run the full pipeline in one command
- auto-generated GenAI discussion notes for the report

## Feature mapping
- Age, S.cr, Na, Mg, K, Ca, Li, Be, Al, V, Cr, Mn, Fe, Co, Ni, Cu,
  Zn, Ga, As, Se, Rb, Sr, Ag, Cd, In, Cs, Ba, Hg, Tl, Pb, Bi, U

Legacy aliases are still accepted:
- F1=Na, F2=Mg, F3=K, ..., F30=U

## Main commands
Run the full pipeline:
```bash
python ckdu_analysis_suite_v4_fusion.py --csv CKDu_processed.csv --analysis all