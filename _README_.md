# E-Commerce Customer Behavior & Spending Prediction

**Course:** 23CSE301 – Machine Learning  
**Assessment:** Review 1  
**Project Type:** Machine Learning Case Study

This project explores two e-commerce datasets using classification and regression. The classification task predicts whether an online shopping session results in a purchase, while the regression task estimates customer spending.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Datasets and Tasks](#datasets-and-tasks)
- [Model Comparison and Results](#model-comparison-and-results)
- [Technology Stack](#technology-stack)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Workflow](#workflow)
- [Team Members](#team-members)
- [Review 1 Scope](#review-1-scope)
- [Reproducibility and Data Notes](#reproducibility-and-data-notes)

---

## Project Overview

The repository contains two Jupyter notebooks:

| Notebook | Machine Learning Task | Target Variable | Objective |
|---|---|---|---|
| `23CSE301_Classification(2).ipynb` | Binary classification | `Revenue` | Predict whether an online shopping session results in a purchase |
| `23CSE301_Regression(2).ipynb` | Regression | `Monetary` | Estimate customer spending |

## Datasets and Tasks

### 1. Online Shoppers Purchasing Intention

**Expected CSV filename:** `online_shoppers_intention.csv`

This dataset describes online shopping sessions using features such as browsing activity, session duration, bounce and exit rates, page values, and visitor type.

| Item | Description |
|---|---|
| Task | Binary classification |
| Target | `Revenue` |
| Prediction | Whether a shopping session results in a purchase |
| Evaluation metrics | Accuracy, weighted F1-score, purchase-class F1-score, classification report, and confusion matrix |

The classification notebook audits the data, explores numerical and categorical features, checks missing and duplicate values, prepares the features, encodes categorical columns, scales data where needed, and compares several classification algorithms.

**Models evaluated:**
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes
- Decision Tree
- Support Vector Machine (SVM)

### 2. E-Commerce Customer Spending

**Expected CSV filename:** `ecommerce_user_segmentation.csv`

This dataset contains customer and shopping-behavior information used to estimate the `Monetary` target.

| Item | Description |
|---|---|
| Task | Regression |
| Target | `Monetary` |
| Prediction | Customer spending |
| Evaluation metrics | R² score, Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE) |

The regression notebook audits the dataset, performs exploratory data analysis, examines feature-target relationships, prepares features, compares regression algorithms, tunes selected models, and performs cross-validation.

**Models evaluated:**
- Linear Regression
- Ridge Regression
- Lasso Regression
- ElasticNet Regression
- Polynomial Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- Support Vector Regressor (SVR)
- K-Nearest Neighbors Regressor

## Model Comparison and Results

The following values are the results recorded in the notebooks. Re-run the notebooks if the data, preprocessing, or model settings change.

### Classification results

| Model | Accuracy | Weighted F1-score | Purchase-class F1-score |
|---|---:|---:|---:|
| K-Nearest Neighbors | 0.8222 | 0.7821 | 0.1423 |
| Gaussian Naive Bayes | 0.6796 | 0.7205 | 0.3744 |
| Support Vector Machine | 0.6731 | 0.7168 | **0.4045** |
| Logistic Regression | 0.6448 | 0.6933 | 0.4033 |
| Decision Tree | 0.6022 | 0.6561 | 0.3748 |

**Interpretation:** KNN records the highest accuracy in this comparison. However, its purchase-class F1-score is low, indicating weak performance in identifying purchase sessions. SVM records the highest purchase-class F1-score among the listed models, making it the strongest choice when identifying purchases is the main priority. Its score is close to Logistic Regression, so this difference should not be overstated.

### Regression results

The initial model comparison recorded the following results:

| Model | R² score | RMSE | MAE |
|---|---:|---:|---:|
| Random Forest Regressor | **0.8469** | **507.84** | **351.94** |
| Gradient Boosting Regressor | 0.8441 | 512.45 | 356.60 |
| KNN Regressor | 0.8352 | 527.00 | 360.31 |
| Polynomial Regression | 0.8179 | 553.89 | 380.43 |
| ElasticNet Regression | 0.7898 | 595.15 | 438.98 |
| Ridge Regression | 0.7893 | 595.75 | 439.52 |
| Linear Regression | 0.7893 | 595.75 | 439.53 |
| Lasso Regression | 0.7893 | 595.76 | 439.54 |
| Decision Tree Regressor | 0.7050 | 704.94 | 464.97 |
| Support Vector Regressor | 0.4174 | 990.74 | 630.68 |

The notebook then tunes the top models. The recorded tuned results are:

| Tuned Model | Best recorded settings | R² score | RMSE | MAE |
|---|---|---:|---:|---:|
| Random Forest Regressor | `max_depth=10`, `n_estimators=100` | **0.8484** | **505.45** | **349.81** |
| Gradient Boosting Regressor | `learning_rate=0.05`, `max_depth=5`, `n_estimators=100` | 0.8473 | 507.21 | 353.25 |

**Interpretation:** The tuned Random Forest Regressor achieves the highest recorded test-set R² score among these tuned models and has the lowest RMSE and MAE of the two. Its five-fold cross-validation mean R² score is approximately **0.8518**. These results support selecting tuned Random Forest as the leading regression model in this experiment.

> **Metric note:** Higher R² and F1 scores are generally better. Lower RMSE and MAE indicate smaller prediction errors. Classification accuracy should not be considered alone when the target classes are imbalanced.

## Technology Stack

- **Language:** Python
- **Environment:** Jupyter Notebook
- **Data handling:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine learning:** Scikit-learn

## Repository Structure

```text
.
├── README.md
├── 23CSE301_Classification(2).ipynb
├── 23CSE301_Regression(2).ipynb
└── datasets/
    ├── online_shoppers_intention.csv
    └── ecommerce_user_segmentation.csv
```

The `datasets/` directory shows a suggested layout. Add the CSV files only if their licenses and terms permit redistribution. If your files are stored in another location, update the relevant `pd.read_csv(...)` paths in the notebooks.

## Getting Started

### 1. Clone the repository

Replace the placeholders with your actual GitHub repository URL and folder name.

```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

Create a `requirements.txt` file with the following packages:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

Install them:

```bash
pip install -r requirements.txt
```

### 4. Add the datasets

Place each CSV file in the location expected by its notebook. The filenames referenced in the notebooks are:

```python
pd.read_csv("online_shoppers_intention.csv")
pd.read_csv("ecommerce_user_segmentation.csv")
```

If the files are inside a `datasets/` folder, update the notebook paths accordingly, for example:

```python
pd.read_csv("datasets/online_shoppers_intention.csv")
```

### 5. Run the notebooks

```bash
jupyter notebook
```

Open each notebook and run its cells from top to bottom. Check the generated metrics and plots to confirm that the outputs match the current dataset and code.

## Workflow

1. Load and audit each dataset.
2. Perform exploratory data analysis.
3. Inspect missing values, duplicates, distributions, class balance, and feature relationships.
4. Prepare features and target variables.
5. Split data into training and test sets.
6. Train and compare suitable machine learning models.
7. Evaluate models using task-specific metrics.
8. Tune selected models and perform cross-validation where included.
9. Interpret the results and document limitations.

## Team Members

| No. | Team Member | Section | Registration Number |
|---:|---|---|---|
| 1 | Bikkam Nikhi Reddy | CSE-D | `CB.SC.U4CSE24309` |
| 2 | Kattumuri Santosh | CSE-D | `CB.SC.U4CSE24326` |
| 3 | Shaik Moyeen | CSE-D | `CB.SC.U4CSE24347` |

## Review 1 Scope

The notebooks cover dataset auditing, exploratory data analysis, preprocessing and feature preparation, initial model comparison, and regression-model tuning and cross-validation. Keep this section aligned with the work actually completed and demonstrated during the review.

## Reproducibility and Data Notes

- Run notebook cells in order so that variables and transformations are defined before use.
- Keep dataset paths consistent with the repository layout.
- Model scores may vary if the data, random seeds, preprocessing, or model settings change.
- Verify dataset sources, terms of use, and licenses before sharing the CSV files publicly.
- Do not commit passwords, API keys, personal data, virtual environments, or unrelated generated files.
- Add a `LICENSE` file only after deciding on a suitable project license and confirming that dataset terms allow the intended use.
