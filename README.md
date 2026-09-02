# Multi-Disease Prediction System

Streamlit app estimating liver disease, chronic kidney disease, and
Parkinson's likelihood from three independently trained models.

## Structure

```
app.py                       # everything: preprocessing, model loading, styling, all 4 pages
requirements.txt
.streamlit/config.toml        # base theme
models/
  liver_model.joblib          # {model, scaler, feature_columns, skewed_cols}
  kidney_model.joblib          # {model, scaler, feature_columns, skewed_cols, cont_cols,
                               #  binary_cols, binary_value_maps, impute_medians,
                               #  impute_modes, sod_pot_outlier_bounds}
  parkinsons_model.joblib      # {model, scaler, feature_columns}
```

## Why preprocessing is inside app.py, not a Pipeline

Each `.joblib` is a dict, not a self-contained `sklearn.Pipeline` — the
notebooks apply feature engineering (gender encoding, the A/G-ratio formula,
categorical mapping, outlier nulling, log1p, median/mode imputation) as
manual pandas cells. The `preprocess_liver` / `preprocess_kidney` /
`preprocess_parkinsons` functions near the top of `app.py` replicate those
exact steps, driven by the metadata already stored in each artifact
(`feature_columns`, `skewed_cols`, `impute_medians`, etc.) rather than
hardcoded column lists.

If you retrain a model with a different manual preprocessing sequence,
update the matching `preprocess_*` function to match — the rest of the app
doesn't need to change.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

Push this folder's contents to a GitHub repo root (so `app.py` sits at the
top level), then point a new Streamlit Community Cloud app at the repo and
`app.py`. Models are small enough to commit directly — no Git LFS needed.

## Notes

- Not a diagnostic tool — the app states this under every result.
- Kidney sodium/potassium inputs are bounded in the form to the same range
  the model was trained to treat as valid (`sod >= 50`, `pot <= 15`);
  outside that range those values were nulled and median-imputed during
  training.
