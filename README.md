# 🚢 Titanic Survival Prediction — Machine Learning Baseline

A clean, leak-free, and reproducible Machine Learning project predicting passenger survival on the RMS Titanic using a **Logistic Regression** baseline. Built with `scikit-learn`'s `Pipeline` and `ColumnTransformer` to enforce strict methodological standards and prevent data leakage.

---

## 📌 Table of Contents
- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Dataset Summary](#dataset-summary)
- [Data Preprocessing & Leakage Prevention](#data-preprocessing--leakage-prevention)
- [Feature Engineering & Selection](#feature-engineering--selection)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Baseline Model Pipeline](#baseline-model-pipeline)
- [Model Evaluation & Actual Metrics](#model-evaluation--actual-metrics)
- [Interpretation of Results](#interpretation-of-results)
- [Limitations & Future Work](#limitations--future-work)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Author](#author)

---

## 📖 Project Overview

The sinking of the RMS Titanic is one of the most well-known maritime tragedies in modern history. Of the 2,224 passengers and crew aboard, more than 1,500 died. This project investigates the demographic and socioeconomic factors that influenced survival and establishes an interpretable, leakage-free classification baseline.

### Key Highlights
* **Zero Data Leakage:** Imputation, scaling, and categorical encoding are encapsulated inside `Pipeline` and `ColumnTransformer`, fitted strictly on training data.
* **Stratified 80/20 Train/Test Split:** Preserves class balance across partitions.
* **Thorough EDA:** Includes visualizations of survival rates by gender, class, and age bracket with domain explanations.
* **Interpretable Baseline:** Logistic Regression with log-odds coefficient analysis.

---

## 🎯 Problem Statement

Develop a binary classification model:
$$\hat{y} \in \{0, 1\}$$
where:
* **0 = Perished** (Did not survive)
* **1 = Survived**

The model predicts whether a given passenger survived the disaster based on demographic, ticket, and familial features.

---

## 📊 Dataset Summary

* **Source:** Kaggle Titanic Competition (`titanic.csv`)
* **Total Records:** 891 passengers
* **Total Columns:** 12
* **Target Distribution:**
  * Perished (`0`): 549 passengers (**61.62%**)
  * Survived (`1`): 342 passengers (**38.38%**)
* **Missing Value Audit:**
  * `Cabin`: 687 missing (**77.10%**) — excluded due to excessive missingness.
  * `Age`: 177 missing (**19.87%**) — imputed via median inside the training pipeline.
  * `Embarked`: 2 missing (**0.22%**) — imputed via mode inside the training pipeline.

---

## ⚙️ Data Preprocessing & Leakage Prevention

A critical flaw in naive ML workflows is computing imputation statistics or scaling parameters across the full dataset prior to splitting. In this project:
1. The raw dataset is split into `X_train`, `X_test`, `y_train`, and `y_test` using:
   ```python
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.20, random_state=42, stratify=y
   )
   ```
2. Imputation statistics (median for numerical, mode for categorical) and standardization parameters ($\mu, \sigma$) are computed **only** on `X_train` during `pipeline.fit()`.
3. The test set (`X_test`) is held out and transformed strictly using the training parameters during `pipeline.predict()`.

---

## 🛠️ Feature Engineering & Selection

### 1. Engineered Feature: `FamilySize`
Combining individual family columns into a unified measure:
$$\text{FamilySize} = \text{SibSp} + \text{Parch} + 1$$
*(where $+1$ accounts for the passenger).*

### 2. Feature Selection
* **Numerical Features (5):** `Age`, `Fare`, `SibSp`, `Parch`, `FamilySize`
* **Categorical Features (3):** `Sex`, `Embarked`, `Pclass` (encoded categorically to avoid imposing arbitrary linear distance between ticket tiers)

### 3. Excluded Features & Justification
* `PassengerId`: Arbitrary database primary key; zero generalizable signal.
* `Name`: High-cardinality unique text string.
* `Ticket`: Irregular, semi-structured alphanumeric codes with high cardinality.
* `Cabin`: Missing in **77.10%** of rows. Imputing over three-quarters of missing values would introduce substantial artificial noise into a linear baseline.

---

## 🔍 Exploratory Data Analysis (EDA)

### Key Insights:
1. **Survival by Gender:**
   * Female survival rate: **74.2%**
   * Male survival rate: **18.9%**
   * *Takeaway:* Gender was the single most dominant factor determining survival, driven by the *"women and children first"* protocol.
2. **Survival by Passenger Class:**
   * 1st Class: **63.0%**
   * 2nd Class: **47.3%**
   * 3rd Class: **24.2%**
   * *Takeaway:* Strong socioeconomic gradient; first-class passengers enjoyed proximity to the boat deck and priority evacuation.
3. **Survival by Age Bracket:**
   * Children (0–12 years): **58.0%** survival rate.
   * Elderly (61+ years): **22.7%** survival rate.
   * *Takeaway:* Young children were prioritized during evacuation; older passengers faced severe physical barriers.
4. **Correlation Analysis:**
   * Negative correlation between `Pclass` and survival ($r = -0.34$).
   * Positive correlation between `Fare` and survival ($r = +0.26$).

---

## 🤖 Baseline Model Pipeline

A scikit-learn `Pipeline` combining `ColumnTransformer` with `LogisticRegression`:

```python
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

baseline_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])
```

### Why Logistic Regression?
* Transparent and highly interpretable via log-odds coefficients.
* Fast, stable, and convex optimization with minimal risk of arbitrary overfitting.
* Yields well-calibrated probabilities, serving as the benchmark for any future complex architectures.

---

## 📈 Model Evaluation & Actual Metrics

The pipeline was fitted **only** on the 712 training samples and evaluated on the **179 untouched held-out test samples**.

### Held-Out Test Set Performance (Actual Results)

| Metric | Score | Percentage |
| :--- | :---: | :---: |
| **Accuracy** | **0.8045** | **80.45%** |
| **Precision** | **0.7931** | **79.31%** |
| **Recall** | **0.6667** | **66.67%** |
| **F1 Score** | **0.7244** | **72.44%** |

### Confusion Matrix Breakdown (N = 179)

```
                 Predicted Perished (0)    Predicted Survived (1)
Actual Perished (0)        98 (TN)                   12 (FP)
Actual Survived (1)        23 (FN)                   46 (TP)
```

* **True Negatives (TN = 98):** Correctly predicted passenger perished.
* **True Positives (TP = 46):** Correctly predicted passenger survived.
* **False Positives (FP = 12):** Predicted to survive, but actually perished.
* **False Negatives (FN = 23):** Predicted to perish, but actually survived.

### Classification Report

```
              precision    recall  f1-score   support

Perished (0)     0.8099    0.8909    0.8485       110
Survived (1)     0.7931    0.6667    0.7244        69

    accuracy                         0.8045       179
   macro avg     0.8015    0.7788    0.7864       179
weighted avg     0.8034    0.8045    0.8007       179
```

---

## 💡 Interpretation of Results

### Feature Coefficients (Log-Odds Impact):
* **Strongest Positive Drivers:**
  * `Sex_female` ($+1.3627$): Being female drastically increased log-odds of survival.
  * `Pclass_1` ($+1.0361$): 1st class ticket holder status provided substantial survival advantage.
* **Strongest Negative Drivers:**
  * `Sex_male` ($-1.2431$): Being male heavily decreased log-odds of survival.
  * `Pclass_3` ($-1.0558$): 3rd class steerage ticket holder status significantly reduced survival odds.
* **Continuous Features:**
  * `Age` ($-0.4892$): Each standard deviation increase in age reduced survival probability.
  * `Fare` ($+0.1339$): Higher fare provided a slight positive survival probability boost.

---

## 🚀 Limitations & Future Work

### Current Limitations:
1. **Linear Decision Boundary:** Cannot model complex feature interactions (e.g., third-class women vs. first-class women).
2. **Missing Spatial Deck Information:** Excluding `Cabin` omitted deck proximity signals.
3. **Unused Honorifics:** Names contain titles (*Master*, *Miss*, *Mrs*, *Mr*) that provide richer social and age status.

### Planned Improvements:
1. **Feature Engineering:**
   * Extract social titles from `Name` (`Mr`, `Mrs`, `Miss`, `Master`, `Noble`).
   * Extract deck levels (`A` through `G`) from non-null `Cabin` records with an `Unknown` indicator.
2. **Ensemble Modeling:**
   * Benchmark against Random Forest, LightGBM, and XGBoost.
3. **Cross-Validation & Tuning:**
   * Perform 5-fold Stratified Cross-Validation with `GridSearchCV` to optimize regularization parameters.

---

## 📂 Project Structure

```
Titanic-Survival-Prediction/
├── titanic.csv               # Kaggle Titanic training dataset
├── Titanic_Project.ipynb     # Complete, executed end-to-end Jupyter Notebook
└── README.md                 # Project documentation and reproduction guide
```

---

## 💻 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/pavankarthikeyaatchyuta-lab/Titanic-Survival-Prediction.git
cd Titanic-Survival-Prediction
```

### 2. Set up Python environment & dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 3. Run the notebook
Launch Jupyter Notebook or Jupyter Lab:
```bash
jupyter notebook Titanic_Project.ipynb
```
Or execute headlessly from the command line:
```bash
jupyter nbconvert --to notebook --execute Titanic_Project.ipynb --output Titanic_Project_executed.ipynb
```

---

## 👨‍💻 Author

**Pavan Karthikeya Atchyuta**  
GitHub: [pavankarthikeyaatchyuta-lab](https://github.com/pavankarthikeyaatchyuta-lab)\n