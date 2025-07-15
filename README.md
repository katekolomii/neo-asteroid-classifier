## Hazardous Near-Earth Object Classification

This machine learning project classifies potentially hazardous asteroids using NASA’s Near-Earth Object (NEO) dataset. The goal is to predict hazard status based on features such as absolute magnitude, miss distance, and relative velocity.

📍 **Live Demo**:  
👉 [Try it on Streamlit →](https://asteroid-classifier.streamlit.app/)

This project was developed as part of the **Introduction to Machine Learning** course offered by the **Stanford Pre-Collegiate Summer Institutes**.

---

## Problem Statement

In planetary defense, **recall is the critical metric**—failing to identify a hazardous object can have serious consequences, while false positives are more tolerable.

**Objective:** Maximize recall for the hazardous class, even at the cost of lower precision.

---

## Dataset

- **Source**:
  - [NASA Open API](https://api.nasa.gov/)
  - [NEO Dataset on Kaggle](https://www.kaggle.com/datasets/sameepvani/nasa-nearest-earth-objects?resource=download)
  - [NASA CNEOS Close Approaches](https://cneos.jpl.nasa.gov/ca/)
- **Features:** Diameter, velocity, miss distance, magnitude, and others.
- **Target:** Binary label indicating whether a NEO is hazardous.

---

## Project Workflow

### 1. Data Cleaning & Preprocessing
- Dropped redundant columns (`id`, `name`, `orbiting_body`, `sentry_object`)
- Handled class imbalance using **SMOTE**
- Normalized numerical features using `MinMaxScaler`

### 2. Exploratory Data Analysis (EDA)
- Univariate and bivariate visualizations
- Correlation matrix, scatter plots, and boxplots

### 3. Model Training & Evaluation
Trained and tuned the following supervised classifiers using `GridSearchCV` and `StratifiedKFold`:
- `Logistic Regression`
- `Random Forest`
- `K-Nearest Neighbors (KNN)`
- `Support Vector Machine (SVM)`

### 4. Metrics & Model Comparison
- Evaluated using confusion matrices and classification reports
- Compared models based on **precision**, **recall**, and **F1-score**

---

## Results Summary

| Model                  | Precision | Recall   | F1-Score |
|------------------------|-----------|----------|----------|
| K-Nearest Neighbors    | 0.35      | 0.85     | **0.50** |
| Random Forest          | 0.32      | **0.97** | 0.48     |
| Support Vector Machine | 0.30      | 0.95     | 0.46     |
| Logistic Regression    | 0.30      | 0.96     | 0.46     |


## Key Takeaways

- The dataset was highly imbalanced. Applying **SMOTE** significantly improved model recall.
- **KNN** yielded the highest F1-score (0.50), offering a good balance between recall and precision.
- **Random Forest** achieved the best recall (0.97), essential for minimizing false negatives in planetary defense.
- All models showed moderate performance, with F1-scores ranging from 0.46–0.50.

---

## Impact & Interpretation

The best model identifies ~96% of truly hazardous asteroids (recall ≈ 0.96), meaning it catches almost all real threats. The trade-off is about 30% precision, indicating some non-hazardous objects are falsely flagged – an acceptable compromise in a safety-critical context (better to have false alarms than miss a dangerous asteroid).

---

## Future Improvements

- Explore advanced models: `XGBoost`, ensemble techniques, or `LightGBM`
- Tune classification threshold using **precision-recall tradeoff**
- Incorporate additional features (e.g., orbital trajectory, approach time)

---

## Skills & Tools Demonstrated

- **Languages**: Python  
- **Libraries**: `scikit-learn`, `imblearn`, `pandas`, `numpy`, `matplotlib`, `seaborn`  
- **Techniques**:  
  - Data cleaning & feature engineering  
  - Feature scaling with `MinMaxScaler`  
  - Handling class imbalance with `SMOTE`  
  - Model tuning using `GridSearchCV` & `StratifiedKFold`  
  - Evaluation with confusion matrix, precision, recall, F1  
  - Visualizations: scatter plots, histograms, bar plots
