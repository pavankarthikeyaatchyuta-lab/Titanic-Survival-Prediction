# 🚢 Titanic Survival Prediction: Machine Learning Baseline

A clean, leak-free, and reproducible Machine Learning project predicting passenger survival on the RMS Titanic using a **Logistic Regression** baseline. Built with `scikit-learn`'s `Pipeline` and `ColumnTransformer` to enforce strict methodological standards and prevent data leakage.

---

## 📌 Table of Contents
- [Project Overview](#project-overview)
- [Dataset Attribution & Description](#dataset-attribution--description)
- [Data Quality & Initial Inspection](#data-quality--initial-inspection)
- [Data Preprocessing & Leakage Prevention](#data-preprocessing--leakage-prevention)
- [Feature Engineering & Selection](#feature-engineering--selection)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Baseline Model Pipeline](#baseline-model-pipeline)
- [Model Evaluation & Actual Metrics](#model-evaluation--actual-metrics)
- [Metric Reasoning & Error Analysis](#metric-reasoning--error-analysis)
- [Model Interpretability (Coefficient Analysis)](#model-interpretability-coefficient-analysis)
- [Limitations & Future Improvements](#limitations--future-improvements)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Author](#author)

---

## 📖 Project Overview

This project builds an interpretable Machine Learning classification baseline to predict passenger survival outcomes from the 1912 RMS Titanic disaster. The project emphasizes core data science methodology:
* **Zero Data Leakage:** Preprocessing operations (imputation, scaling, encoding) learn statistics strictly from training data.
* **Stratified Splitting:** Preserves empirical class proportions across training and test partitions.
* **Question-Driven EDA:** Evaluates survival associations using statistically careful observational language.
* **Interpretable Reference Baseline:** Leverages Logistic Regression with log-odds coefficient analysis.

---

## 📊 Dataset Attribution & Description

* **Dataset:** **Kaggle Titanic `train.csv` dataset** (`titanic.csv`).
* **Source:** Open benchmark dataset provided by Kaggle for the Titanic: Machine Learning from Disaster competition. *(The dataset was not created or collected by the author; it is the official benchmark dataset used for this assignment).*
* **Scope:** Contains demographic, accommodation, ticket, and survival records for 891 passengers.

---

## 🔍 Data Quality & Initial Inspection

| Data-Quality Check | Observed Result |
| :--- | :--- |
| **Total Rows (Passengers)** | 891 |
| **Total Columns (Features)** | 12 |
| **Missing Age Values** | 177 (19.87%) |
| **Missing Cabin Values** | 687 (77.10%) |
| **Missing Embarked Values** | 2 (0.22%) |
| **Target Column** | `Survived` (Binary: 0 = Perished, 1 = Survived) |
| **Target Distribution** | Perished (0): 549 (61.62%), Survived (1): 342 (38.38%) |

---

## ⚙️ Data Preprocessing & Leakage Prevention

To prevent subtle data leakage, preprocessing statistics are never computed on the full dataset prior to partitioning:
1. The dataset is split into `X_train`, `X_test`, `y_train`, and `y_test` using:
   ```python
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.20, random_state=42, stratify=y
   )
   ```
   * **80% training / 20% held-out test data:** 712 training samples and 179 test samples.
   * **`random_state=42`:** Guarantees exact reproducibility across runs.
   * **`stratify=y`:** Preserves the 61.62% / 38.38% class distribution in both splits.
   * **Untouched Test Set:** The test set is held out completely until final model evaluation.
2. Imputation statistics (median for numerical features, mode for categorical features), standardization parameters ($\mu, \sigma$), and one-hot encoding vocabularies are learned **strictly on `X_train`** during `pipeline.fit()`.

---

## 🛠️ Feature Engineering & Selection

### 1. Engineered Feature: `FamilySize`
$$\text{FamilySize} = \text{SibSp} + \text{Parch} + 1$$
Combines sibling/spouse count and parent/child count with the passenger themselves ($+1$) to represent traveling party size.

### 2. Feature Selection Setup
* **Numerical Features (5):** `Age`, `Fare`, `SibSp`, `Parch`, `FamilySize`
* **Categorical Features (3):** `Sex`, `Embarked`, `Pclass`  
  *(Note: `Pclass` is explicitly treated as a categorical variable to avoid assuming equal linear distance between ticket classes).*

### 3. Methodological Exclusion Rationales
* **`PassengerId`:** Identifier with no intended predictive meaning; excluded from the baseline to prevent arbitrary memorization.
* **`Name`:** Contains potentially useful information (such as honorific titles) but requires additional feature extraction; excluded from this simple baseline.
* **`Ticket`:** High-cardinality identifier/group information with irregular alphanumeric formatting; excluded from the baseline.
* **`Cabin`:** Very high missingness in the original dataset (**687 out of 891 records, 77.10%**); excluded from this baseline rather than introducing a more complex missingness/deck imputation strategy.

---

## 📈 Exploratory Data Analysis (EDA)

All visualizations are accompanied by statistically careful interpretations without making unsupported causal claims from observational data:

1. **Survival by Gender:**
   * Female observed survival rate: **74.2%**
   * Male observed survival rate: **18.9%**
   * *Interpretation:* Survival rates differed substantially by gender in the dataset. While historical accounts of the evacuation give context regarding lifeboat access, the observational data demonstrates a very strong empirical association between gender and survival outcomes.
2. **Survival by Passenger Class:**
   * 1st Class observed survival rate: **63.0%**
   * 2nd Class observed survival rate: **47.3%**
   * 3rd Class observed survival rate: **24.2%**
   * *Interpretation:* Observed survival rates differed across ticket classes. While historical records describe differences in cabin proximity to the boat decks, within this observational dataset, passenger class provides a strong predictive indicator of survival probability.
3. **Survival by Age Bracket:**
   * Children (0–12 years): **58.0%** observed survival rate.
   * Elderly (61+ years): **22.7%** observed survival rate.
   * Adults (19–60 years): Observed rates between **38% and 40%**.
   * *Interpretation:* Children had a higher observed survival rate than older passengers in this dataset, indicating that age contains useful predictive information.
4. **Correlation Analysis:**
   * Inverse correlation between `Pclass` and survival ($r = -0.34$).
   * Positive correlation between `Fare` and survival ($r = +0.26$).

---

## 🤖 Baseline Model Pipeline

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

### Baseline Model Choice
Logistic Regression was selected as the baseline because Titanic survival is a binary classification problem. It provides a simple and interpretable reference model, while its coefficients allow us to examine the direction and relative contribution of encoded features.

---

## 📊 Model Evaluation & Actual Metrics

The pipeline was fitted **only** on the 712 training samples and evaluated on the **179 untouched held-out test samples**.

### Held-Out Test Set Performance (Actual Un-Fabricated Results)

| Metric | Score | Percentage |
| :--- | :---: | :---: |
| **Accuracy** | **0.8045** | **80.45%** |
| **Precision** | **0.7931** | **79.31%** |
| **Recall** | **0.6667** | **66.67%** |
| **F1 Score** | **0.7244** | **72.44%** |

### Confusion Matrix Breakdown ($N = 179$)

```
                     Predicted Perished (0)    Predicted Survived (1)
Actual Perished (0)            98 (TN)                   12 (FP)
Actual Survived (1)            23 (FN)                   46 (TP)
```

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

## 💡 Metric Reasoning & Error Analysis

### Why Report Multiple Metrics?
Accuracy measures the overall proportion of correct predictions. Precision measures how often passengers predicted to survive actually survived, while recall measures how many actual survivors were correctly identified. F1 combines precision and recall into a single balanced measure. Reporting all four provides a more complete evaluation than accuracy alone.

In the context of Titanic survival prediction:
* **False Positive (FP = 12):** The model predicted a passenger survived, but they actually did not survive.
* **False Negative (FN = 23):** The model predicted a passenger did not survive, but they actually survived.

Neither type of error is inherently more important than the other in this retrospective historical baseline; reporting both precision and recall provides full transparency into model error trade-offs.

---

## 🔍 Model Interpretability (Coefficient Analysis)

The coefficients represent statistical associations learned by this linear model on the training data and do not establish causation:
* The Logistic Regression coefficients indicate a **strong positive association** between the female indicator (`Sex_female`, $+1.3627$) and predicted survival probability.
* First-class status (`Pclass_1`, $+1.0361$) had a **strong positive association** with predicted survival.
* Third-class status (`Pclass_3`, $-1.0558$), male gender (`Sex_male`, $-1.2431$), and increasing age (`Age`, $-0.4892$) had **negative associations** in the fitted model.
* Continuous fare (`Fare`, $+0.1339$) exhibited a modest positive association with predicted survival probability.

---

## 🚀 Limitations & Future Improvements

### Current Limitations:
1. **Linearity Assumption:** Logistic Regression models linear log-odds decision boundaries, missing complex non-linear feature interactions without manual interaction terms.
2. **Cabin Omission:** Omitting `Cabin` due to 77.10% missingness discarded potential spatial deck proximity signals.
3. **Unused Text Data:** Passenger names contain social honorifics (*Master*, *Miss*, *Mrs*, *Mr*) that could offer additional demographic nuance.

### Planned Next Steps:
1. **Title Extraction:** Extract honorific titles from `Name` to refine age imputation and capture marital/social status.
2. **Cabin Deck Extraction:** Extract deck letters from non-null `Cabin` records while handling missing values with an explicit `Missing` category.
3. **Cross-Validation & Hyperparameter Tuning:** Cross-validation and hyperparameter tuning should be performed using the training data, with the final test set remaining untouched until final evaluation.
4. **Non-Linear Ensembles:** Benchmark the Logistic Regression baseline against tree-based ensembles (Random Forest, Gradient Boosting, XGBoost).

---

## 📂 Project Structure

```
Titanic-Survival-Prediction/
├── .gitignore                # Git exclusion rules for Python & Jupyter artifacts
├── titanic.csv               # Kaggle Titanic train.csv dataset
├── Titanic_Project.ipynb     # Fully executed Jupyter Notebook with all outputs
└── README.md                 # Complete project documentation and reproduction guide
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
Launch Jupyter Notebook:
```bash
jupyter notebook Titanic_Project.ipynb
```
Or execute headlessly from the command line:
```bash
jupyter nbconvert --to notebook --execute Titanic_Project.ipynb --output Titanic_Project.ipynb
```

---

## 👨‍💻 Author

**Pavan Karthikeya Atchyuta**  
GitHub: [pavankarthikeyaatchyuta-lab](https://github.com/pavankarthikeyaatchyuta-lab)\n