# Multi-Step ECG Time-Series Forecasting

**Does fixing 3 bad data points beat adding a whole extra LSTM layer?**

This project forecasts 17-step ECG-derived clinical sequences with LSTM and GRU networks, then asks a narrower question than "which architecture wins": how much of the accuracy gap between models is actually just outlier noise in the training data. Imputing three flagged outlier values in one target column turns out to move the error more than doubling model depth does.

---

## Key Results

| Scenario | Model | MSE | RMSE | MAE |
|---|---|---|---|---|
| Before imputation | LSTM (1 layer) | 16.27 | 3.68 | 1.96 |
| Before imputation | LSTM (2 layers) | 9.04 | 2.96 | 2.03 |
| Before imputation | GRU (1 layer) | 16.28 | 3.66 | 1.92 |
| Before imputation | GRU (2 layers) | 8.88 | 2.91 | 1.86 |
| After imputation | LSTM (1 layer) | 10.42 | 3.22 | 2.37 |
| After imputation | LSTM (2 layers) | 9.06 | 3.00 | 2.22 |
| After imputation | GRU (1 layer) | 9.06 | 3.00 | 2.19 |
| **After imputation** | **GRU (2 layers)** | **7.11** | **2.65** | **1.85** |

Imputing 3 outlier values in `Output13` (raw value 124 against a normal range of roughly 10–23, replaced with the column mean of ≈14.79) cuts MSE by **19.9%** on the best model and by **36.0%** on the single-layer LSTM — a bigger swing than going from 1 to 2 recurrent layers produces on its own.

---

## Dataset

- 65 patients, one row each in `data/ecg_raw.csv`
- 8 input features (`O1`–`O8`) recorded at each of 17 sequential time-steps
- 17 target outputs (`Output1`–`Output17`), one per time-step
- `Output13` carries 3 extreme outliers (value 124) that dominate its variance until treated

`src/data.py` handles the full pipeline: CSV ingestion, coercion of the `NT` sentinel to `NaN`, IQR-based outlier detection (`detect_outliers_iqr`, k=3.0 to isolate genuine extremes rather than normal spread), mean imputation (`impute_outliers`), and MinMax scaling before the arrays are handed to a PyTorch `Dataset`.

---

## Models

`src/models.py` defines `LSTMForecaster` and `GRUForecaster`, both configurable via a shared `model_factory`:

| Architecture | Layers | Hidden dim | Dropout |
|---|---|---|---|
| LSTM / GRU, 1 layer | 1 | 64 | 0.0 |
| LSTM / GRU, 2 layers | 2 | 64 | 0.2 (inter-layer only) |

Each model maps an `(batch, 17, 8)` input sequence to a `(batch, 17, 1)` per-step prediction through a stacked recurrent encoder and a final dense layer.

**Training** (`src/training.py`): Adam (`lr=1e-3`), early stopping (`patience=10`, restores best weights), 5-fold cross-validation via `cross_validate`. **Evaluation** (`src/metrics.py`): MSE/RMSE/MAE per fold plus paired t-tests (`scipy.stats.ttest_rel`) to check whether differences between model variants are statistically meaningful, not just noise.

---

## Project structure

```
ecg-forecasting/
├── data/
│   ├── ecg_raw.csv              # 65 patients × 17 steps × 8 features
│   └── training_results.pkl     # Cached cross-validation results
├── notebooks/
│   ├── 01_EDA_and_Preprocessing.ipynb
│   ├── 02_Model_Training.ipynb
│   ├── 03_Results_and_Analysis.ipynb
│   └── 04_Outlier_Impact_Study.ipynb
├── src/
│   ├── data.py       # Loading, IQR outlier detection, imputation, scaling, Dataset
│   ├── models.py     # LSTMForecaster / GRUForecaster / model_factory
│   ├── training.py   # train_model, EarlyStopper, cross_validate, evaluate_model
│   └── metrics.py    # MSE/RMSE/MAE, paired_ttest
├── figures/          # Generated plots (correlation heatmap, loss curves, etc.)
└── requirements.txt
```

---

## Running it

```bash
git clone https://github.com/Prabhas-1505/ecg-forecasting.git
cd ecg-forecasting
pip install -r requirements.txt
jupyter lab notebooks/
```

Run the notebooks in order — each one depends on artifacts from the previous:

1. **`01_EDA_and_Preprocessing`** — explore the raw data, visualize the Output13 outliers
2. **`02_Model_Training`** — train all four model variants with 5-fold CV
3. **`03_Results_and_Analysis`** — compare metrics across variants, run the paired t-tests
4. **`04_Outlier_Impact_Study`** — retrain before/after imputation and compare directly

---

## Tech stack

PyTorch · pandas / NumPy / scikit-learn · matplotlib / seaborn · SciPy (`ttest_rel`)

---

## Takeaways

1. **GRU beats LSTM here**, and does it with ~20-25% fewer parameters at the same depth.
2. **Depth helps, but less than expected**: going 1→2 layers narrows the gap between LSTM and GRU, but doesn't close it.
3. **Outlier treatment is the biggest lever tested**: imputing 3 values in one column outperforms adding an entire extra recurrent layer, and the effect is larger on the weaker single-layer models.
4. **Best configuration overall**: 2-layer GRU on the imputed data (MSE 7.11) — the combination of the better architecture and the cleaner target.

---

*Educational / research project.*
