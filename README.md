# Liver Cirrhosis Prediction using Machine Learning

**Repository:** Machine-learning project for predicting liver cirrhosis from clinical / tabular records — includes EDA, preprocessing, model training, evaluation, and a short demo.
**Original inspiration / reference notebook:** *Liver Cirrhosis Prediction — EDA + Models* by vizeno on Kaggle. ([Kaggle][1])

---

## Table of contents

* [Project overview](#project-overview)
* [Dataset](#dataset)
* [Repository structure](#repository-structure)
* [Requirements](#requirements)
* [Quick start](#quick-start)
* [How it works (high level)](#how-it-works-high-level)
* [Modeling & evaluation](#modeling--evaluation)
* [Results](#results)
* [Contributing](#contributing)
* [License & acknowledgements](#license--acknowledgements)
* [Contact](#contact)

---

# Project overview

This project demonstrates exploratory data analysis (EDA), preprocessing, feature engineering, and supervised classification for predicting liver cirrhosis (or general liver disease outcome) using clinical tabular data. It provides reusable code and clear steps to reproduce model training and evaluation. The repository is intended as a learning / prototyping pipeline for clinicians or data scientists interested in non-image liver disease prediction.

---

# Dataset

This work uses public liver disease / cirrhosis datasets commonly used in the community (examples: Indian Liver Patient Dataset and Cirrhosis Prediction datasets). The Indian Liver Patient dataset contains **583 records** (416 patients with liver disease, 167 without) and a small set of clinical features (age, gender, bilirubin, enzymes, proteins, etc.). ([UCI Machine Learning Repository][2])

> **Note:** check `data/` or the notebook in this repo to see the exact dataset file used. If you use the original Kaggle notebook as a starting point, the input and preprocessing steps are shown there. ([Kaggle][1])

---

# Repository structure (suggested)

```
.
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                    # raw dataset files (not committed if large)
│   └── processed/              # cleaned / processed data used for training
├── notebooks/
│   └── eda_and_models.ipynb    # interactive EDA & modelling notebook (from Kaggle)
├── src/
│   ├── data_preprocessing.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── models/                     # saved model artifacts (.pkl / joblib)
├── results/
│   └── metrics_report.json
└── LICENSE
```

---

# Requirements

Minimum recommended environment:

* Python 3.8+
* pandas
* numpy
* scikit-learn
* matplotlib / seaborn (for plots)
* xgboost (optional)
* joblib or pickle (for saving models)
* jupyterlab / notebook (for exploring notebooks)

Example `requirements.txt` (starter):

```
pandas>=1.3
numpy>=1.21
scikit-learn>=1.0
matplotlib>=3.4
seaborn>=0.11
xgboost>=1.5
joblib
jupyterlab
```

---

# Quick start

1. **Clone the repo**

```bash
git clone <this-repo-url>
cd liver-cirrhosis-prediction
```

2. **Create venv & install**

```bash
python -m venv venv
source venv/bin/activate     # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

3. **Place dataset**

* Put your dataset CSV into `data/raw/`. If you replicate the Kaggle notebook, make sure the same file names are used (or update paths in code). See the referenced Kaggle notebook for example input and EDA. ([Kaggle][1])

4. **Run preprocessing**

```bash
python src/data_preprocessing.py --input data/raw/liver_data.csv --output data/processed/clean.csv
```

5. **Train a model**

```bash
python src/train.py --config configs/train_config.yaml
```

6. **Evaluate / predict**

```bash
python src/evaluate.py --model models/best_model.joblib --test data/processed/test.csv
python src/predict.py --model models/best_model.joblib --input-sample samples/sample1.json
```

(Adjust flags to your script implementation — example CLI options are shown above.)

---

# How it works (high level)

1. **EDA** — investigate distributions, missingness, and relationships between features and target. The example Kaggle notebook provides a clear walkthrough with visualizations and basic feature importance. ([Kaggle][1])
2. **Preprocessing** — handle missing values, encode categorical variables (e.g., gender), scale numeric features if needed, and optionally perform feature selection or dimensionality reduction.
3. **Modeling** — train several classifiers (typical choices: Logistic Regression, Random Forest, XGBoost, SVM) with cross-validation and hyperparameter tuning.
4. **Evaluation** — report metrics relevant to medical classification: accuracy, precision, recall, F1-score, ROC-AUC, and a confusion matrix. For imbalanced classes, consider precision-recall curves and class-weighted models.
5. **Deployment / Inference** — simple script to load a model and predict on new patient records.

---

# Modeling & evaluation

* Typical models used in similar notebooks/projects: **Logistic Regression**, **Random Forest**, **XGBoost**, and sometimes **SVM**. Cross-validation and stratified splits are recommended. ([Kaggle][1])
* Important metrics: **ROC-AUC**, **precision**, **recall (sensitivity)**, **specificity**, **F1-score**. In clinical settings, sensitivity (recall) is often prioritized to reduce false negatives.

---

# Results

Place your final metrics and a short summary here (after running experiments). Example placeholder:

```
Model: Random Forest
Accuracy: 0.87
Precision: 0.85
Recall: 0.89
F1-score: 0.87
ROC-AUC: 0.92
```

> For reference, many public experiments on liver disease datasets report high-performing models after careful feature selection — check the referenced Kaggle notebooks and datasets for comparable results and implementation details. ([Kaggle][1])

---

# Notes, limitations & ethics

* **Medical disclaimer:** This project is for educational and research purposes only. Predictions from these models are **not** clinical diagnoses and should not be used for medical decision-making. Consult healthcare professionals for any clinical interpretation.
* **Data quality & bias:** Public clinical datasets may be limited (small sample size, demographic skew). Validate models on representative, high-quality clinical data before any real-world use.
* **Privacy:** Do not commit any sensitive patient data to public repos.

---

# Contributing

Contributions are welcome — PRs for bug fixes, improved preprocessing, new models, or better evaluation scripts are appreciated. When contributing, please:

* Create an issue describing the change or improvement
* Open a PR with tests / reproducible steps
* Keep changes focused and documented

---

# License & acknowledgements

* Add your chosen license file (e.g., MIT / Apache-2.0).
* Acknowledgements:

  * The Kaggle notebook `vizeno / Liver Cirrhosis Prediction — EDA + Models` was used as inspiration and reference. ([Kaggle][1])
  * Public datasets such as the Indian Liver Patient Dataset (ILPD) and other cirrhosis datasets used in community notebooks. ([UCI Machine Learning Repository][2])

---

# Contact

If you have questions or suggestions, open an issue or contact me

[1]: https://www.kaggle.com/code/vizeno/liver-cirrhosis-prediction-eda-models "Liver Cirrhosis Prediction 💊 | EDA + Models  | Kaggle"
[2]: https://archive.ics.uci.edu/ml/datasets/ILPD%2B%28Indian%2BLiver%2BPatient%2BDataset%29?utm_source=chatgpt.com "ILPD (Indian Liver Patient Dataset)"
