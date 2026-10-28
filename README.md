# E-Waste Technological Hubs: Exploratory Data Analysis & Machine Learning Classification Framework

An end-to-end Data Science and Exploratory Data Analysis (EDA) framework designed to characterize electronic waste (e-waste) generation dynamics, formal-informal recycling splits, and environmental management risk tiers across 12 prominent technological hubs in developing economies.

---

## Abstract & Research Overview

Rapid consumer electronics refresh cycles, enterprise server decommissioning, and hardware miniaturization have accelerated municipal e-waste generation globally. Developing technological hubs (such as Bengaluru, Hyderabad, Shenzhen, Lagos, São Paulo, and Hanoi) face unique operational challenges characterized by surging per-capita generation, extensive reliance on informal scrap recovery networks, and limited formal recycling capacity.

This repository provides an empirical analytics framework utilizing multi-year observations across 12 tech hubs (2018–2025). The workflow integrates:
1. **Multi-Source Data Ingestion & Quality Auditing:** Schema standardization, missing value audits, and distribution profiling.
2. **Natural Language Processing (NLP):** 15-dimensional TF-IDF vectorization and domain keyword frequency extraction from qualitative municipal stream descriptions.
3. **Domain Feature Engineering:** Formulation of the *Obsolescence Velocity Index*, *Formal-to-Informal Processing Ratio*, *Informal Burden Percentage*, *Toxic Burden Load*, and *Precious Metal Recoverable Value*.
4. **Statistical Hypothesis Testing:** Parametric ANOVA and non-parametric correlation analysis (Pearson & Spearman).
5. **Comparative Machine Learning Classification:** Benchmarking six predictive algorithms (*Logistic Regression, XGBoost, Support Vector Machines, Random Forest, Decision Trees, and Gaussian Naive Bayes*) to categorize environmental management vulnerability tiers (T1–T4).

---

## Repository Structure

```text
├── data/
│   ├── raw/
│   │   └── ewaste_technological_hubs_dataset.csv     # Primary dataset (1,200 multi-year records)
│   └── processed/
│       ├── cleaned_ewaste_dataset.csv               # Cleaned & validated empirical dataset
│       └── engineered_ewaste_features.csv           # Transformed feature matrix (45 features)
├── notebooks/
│   └── ewaste_eda_trend_characterization.ipynb      # Interactive Jupyter notebook with visual outputs
├── outputs/
│   ├── figures/                                     # Publication-quality charts (300 DPI)
│   ├── models/                                      # Serialized scikit-learn & XGBoost model weights (.joblib)
│   └── tables/                                      # Descriptive statistics, metrics & confusion matrices
├── scripts/
│   ├── run_pipeline.py                              # End-to-end master data science pipeline
│   └── generate_notebook.py                         # Programmatic notebook generator script
├── app.py                                           # Interactive Streamlit web application & ML dashboard
├── .gitignore                                       # Git ignore specifications
├── requirements.txt                                 # Core Python package dependencies
└── README.md                                        # Project documentation
```

---

## Empirical Benchmark Performance

The classification framework categorizes environmental management vulnerability into four tiers (Low, Moderate, High, Severe). Evaluation on an 80:20 stratified holdout split yields the following performance benchmarks:

| Model Architecture | Accuracy | Macro Precision | Macro Recall | Macro F1-score |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **94.2%** | **94.2%** | **94.2%** | **94.2%** |
| **XGBoost Classifier** | **81.7%** | **82.4%** | **81.7%** | **81.9%** |
| **Support Vector Machine (SVM)** | **79.6%** | **82.6%** | **79.6%** | **80.3%** |
| **Random Forest Classifier** | **79.2%** | **80.9%** | **79.2%** | **79.7%** |
| **Decision Tree Classifier** | **74.2%** | **76.1%** | **74.2%** | **74.6%** |
| **Gaussian Naive Bayes** | **64.2%** | **68.8%** | **64.2%** | **65.0%** |

---

## Setup & Execution

### Prerequisites
- Python 3.10 or higher
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/nagul-a/E-Waste-Technological-Hubs-EDA-Framework.git
cd E-Waste-Technological-Hubs-EDA-Framework
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the End-to-End Pipeline
To run data cleaning, NLP vectorization, domain feature computation, model training, and figure export:
```bash
python scripts/run_pipeline.py
```
Outputs (figures, models, and summary metrics) are automatically saved to `outputs/`.

### 4. Interactive Jupyter Notebook
Open the notebook in VS Code, JupyterLab, or Jupyter Notebook:
```bash
jupyter notebook notebooks/ewaste_eda_trend_characterization.ipynb
```

### 5. Launch the Interactive Web Dashboard
Run the Streamlit application to explore global hubs, test live scenario predictions, and inspect model benchmarks in your browser:
```bash
streamlit run app.py
```
App will automatically open at `http://localhost:8501`.

---

## License & Attribution
Distributed under the MIT License. Developed for research and educational purposes in environmental data science and circular economy characterization.
