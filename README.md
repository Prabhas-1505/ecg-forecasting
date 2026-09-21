# 🫀 Multi-Step ECG Time-Series Forecasting

**Impact of Outlier Imputation on Stacked LSTM Performance**

A deep learning project exploring multi-step forecasting of ECG-derived clinical data using recurrent neural networks (LSTM, GRU). The study demonstrates that **data quality** (outlier treatment) can produce larger accuracy gains than increasing model complexity.

---

## 📊 Key Results

| Scenario | Model | MSE | RMSE | MAE |
|---|---|---|---|---|
| Before Imputation | LSTM-1 layer | 16.27 | 3.68 | 1.96 |
| Before Imputation | LSTM-2 layers | 9.04 | 2.96 | 2.03 |
| Before Imputation | GRU-1 layer | 16.28 | 3.66 | 1.92 |
| Before Imputation | GRU-2 layers | 8.88 | 2.91 | 1.86 |
| **After Imputation** | LSTM-1 layer | 10.42 | 3.22 | 2.37 |
| **After Imputation** | LSTM-2 layers | 9.06 | 3.00 | 2.22 |
| **After Imputation** | GRU-1 layer | 9.06 | 3.00 | 2.19 |
| **After Imputation** | **GRU-2 layers** | **7.11** | **2.65** | **1.85** |

> **19.9% MSE reduction** on the best model (GRU-2 layers) by imputing 3 outlier values in Output13 (124 → mean ≈ 14.79)

---

## 🏗️ Architecture

```
Input: (batch, 17 time-steps, 8 features)
                    │
            ┌───────┴───────┐
            │   LSTM Layer 1  │  (hidden_dim=64)
            │   (or GRU)      │
            └───────┬───────┘
                    │
            ┌───────┴───────┐
            │   LSTM Layer 2  │  (hidden_dim=64, dropout=0.2)
            │   (optional)    │
            └───────┬───────┘
                    │
            ┌───────┴───────┐
            │   Dense (1)     │  (per time-step)
            └───────┬───────┘
                    │
Output: (batch, 17 time-steps, 1)
```

---

## 📁 Project Structure

```
ecg-forecasting/
├── README.md                           # This file
├── requirements.txt                    # Dependencies
├── data/
│   └── ecg_raw.csv                     # Cleaned dataset (65 patients)
├── notebooks/
│   ├── 01_EDA_and_Preprocessing.ipynb  # Exploratory data analysis
│   ├── 02_Model_Training.ipynb         # Model training & cross-validation
│   ├── 03_Results_and_Analysis.ipynb   # Results deep-dive & statistical tests
│   └── 04_Outlier_Impact_Study.ipynb   # Before vs after imputation study
├── src/
│   ├── __init__.py
│   ├── data.py                         # Dataset class & preprocessing
│   ├── models.py                       # LSTM & GRU model definitions
│   ├── training.py                     # Training loop & cross-validation
│   └── metrics.py                      # Evaluation metrics & statistics
└── figures/                            # Generated publication-quality plots
```

---

## 🔬 Dataset

- **65 patients** with ECG-derived clinical measurements
- **8 input features** (O1–O8) per time-step
- **17 sequential time-steps** per patient
- **17 target outputs** (Output1–Output17)
- **Key anomaly**: Output13 contains 3 instances of value **124** (vs normal range 10–23)

---

## 🧠 Models Compared

| Architecture | Layers | Hidden Dim | Parameters | Dropout |
|---|---|---|---|---|
| LSTM-1 layer | 1 | 64 | ~17K | 0.0 |
| LSTM-2 layers | 2 | 64 | ~52K | 0.2 |
| GRU-1 layer | 1 | 64 | ~13K | 0.0 |
| GRU-2 layers | 2 | 64 | ~40K | 0.2 |

**Training**: 5-fold cross-validation, Adam optimiser (lr=0.001), early stopping (patience=10)

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/Prabhas-1505/ecg-forecasting.git
cd ecg-forecasting

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the notebooks
jupyter lab notebooks/
```

### Running Order
1. `01_EDA_and_Preprocessing.ipynb` — Explore data, detect outliers
2. `02_Model_Training.ipynb` — Train all models with cross-validation
3. `03_Results_and_Analysis.ipynb` — Analyse results (requires step 2)
4. `04_Outlier_Impact_Study.ipynb` — Before/after imputation comparison

---

## 🛠️ Tech Stack

- **Framework**: PyTorch 2.x
- **Data**: pandas, NumPy, scikit-learn
- **Visualisation**: matplotlib, seaborn
- **Validation**: 5-fold cross-validation with paired t-tests

---

## 📈 Key Findings

1. **Stacked GRU-2 layers** outperforms all other variants across all metrics, achieving the best overall performance (MSE = **7.11**).
2. **Mean imputation** of the 3 outlier values reduces Output13 variance by **97%** (583.55 → 17.02).
3. **Data quality > model complexity**: A simple preprocessing step (imputing 3 values) produced a **19.9% MSE reduction** for the stacked GRU-2 model and **36.0%** for the single-layer LSTM-1 model.
4. **GRU** models outperform their LSTM counterparts while utilizing approximately 20-25% fewer parameters.

---

## 📄 License

This project is for educational and research purposes.

---

*Built with PyTorch • Prabhas Avvaru*
