# Titanic-Survival-Prediction
# 🚢 Titanic Survival Prediction

A Machine Learning project that predicts whether a passenger survived the Titanic disaster using Logistic Regression. This project demonstrates the complete data science workflow including data preprocessing, visualization, feature engineering, and model evaluation.

---

## Project Overview

The Titanic dataset is one of the most popular beginner datasets in data science.
In this project, we analyze passenger data and build a predictive model to classify survival outcomes.

---

## Objective

To apply data science techniques to:

* Clean and preprocess real-world data
* Visualize patterns and relationships
* Build a classification model
* Evaluate model performance

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---

## Dataset

The dataset contains passenger details such as:

* Age
* Sex
* Passenger Class (Pclass)
* Fare
* Family members (SibSp, Parch)
* Embarked location

---

## Workflow

1. **Data Collection**

   * Loaded dataset from CSV file

2. **Data Cleaning**

   * Handled missing values
   * Removed unnecessary columns

3. **Feature Engineering**

   * Created `FamilySize` feature
   * Converted categorical data to numerical

4. **Data Visualization**

   * Survival count plot
   * Age distribution
   * Correlation heatmap

5. **Model Building**

   * Logistic Regression model

6. **Evaluation**

   * Accuracy score
   * Confusion matrix

---

## Results

* **Accuracy:** ~75%
* The model performs well for a basic classification task

### Confusion Matrix

```
[[82 24]
 [19 54]]
```

---

## Sample Outputs

* Survival Count Plot
<img width="718" height="565" alt="image" src="https://github.com/user-attachments/assets/b9b451fb-192c-4cc2-b095-ab5a815c6fc4" />

* Age Distribution Plot
  <img width="714" height="565" alt="image" src="https://github.com/user-attachments/assets/5127837b-dd7c-4fa8-98c4-e33ac27b2693" />

* Correlation Heatmap
  <img width="945" height="685" alt="image" src="https://github.com/user-attachments/assets/017f6460-0025-4ee4-be73-f7c227664688" />

* Model Evaluation Output
<img width="378" height="99" alt="image" src="https://github.com/user-attachments/assets/62119fbf-5fb2-4299-94d1-bdd64fdb886d" />

---

## How to Run

1. Clone the repository:

```
git clone https://github.com/pavankarthikeyaatchyuta-lab/Titanic-Survival-Prediction.git
```

2. Navigate to the project folder:

```
cd Titanic-Survival-Prediction
```

3. Install dependencies:

```
pip install pandas numpy matplotlib seaborn scikit-learn
```

4. Run the notebook in Jupyter or Google Colab

---

## Project Structure

```
Titanic-Survival-Prediction/
│
├── Titanic_Project.ipynb
├── titanic.csv
├── Report.pdf
└── README.md
```

---

## Key Learnings

* Importance of data cleaning
* Role of visualization in understanding data
* Feature engineering improves model performance
* Logistic Regression is effective for binary classification

---

## Conclusion

This project demonstrates how machine learning can be used to extract insights and make predictions from real-world data. It provides a strong foundation in data science concepts and workflow.

---

## 👨‍💻 Author

**Pavan Karthikeya Atchyuta**
GitHub: https://github.com/pavankarthikeyaatchyuta-lab

---
