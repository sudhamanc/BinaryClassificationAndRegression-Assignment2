# DSCI 631 - Applied Machine Learning: Assignment 2

## Overview

This assignment explores core machine learning workflows through two real-world prediction tasks:

1. **Binary Classification** — Predicting whether an individual's income exceeds $50K/year using the [Adult Income Dataset](https://www.kaggle.com/datasets/isathyam31/adult-income-prediction-classification)
2. **Regression** — Predicting used vehicle prices from the [Craigslist Cars/Trucks Dataset](https://www.kaggle.com/austinreese/craigslist-carstrucks-data)

---

## Dataset Setup

> **Note:** The dataset files are not included in this repository due to their large size. You must download them manually and place them in the `data/` folder before running the notebook.

### Required Files

| File | Destination | Size |
|------|-------------|------|
| `data.csv` | `data/data.csv` | ~4 MB |
| `vehicles.csv` | `data/vehicles.csv` | ~1.4 GB |

### Option 1 — Kaggle API (Recommended)

Make sure you have the [Kaggle API](https://github.com/Kaggle/kaggle-api) installed and your `~/.kaggle/kaggle.json` credentials set up.

```bash
# Adult Income dataset → saves as data/data.csv
kaggle datasets download -d isathyam31/adult-income-prediction-classification --unzip -p ./data

# Craigslist Cars/Trucks dataset → saves as data/vehicles.csv
kaggle datasets download -d austinreese/craigslist-carstrucks-data --unzip -p ./data
```

### Option 2 — Manual Download

1. Go to the dataset pages on Kaggle:
   - [Adult Income Dataset](https://www.kaggle.com/datasets/isathyam31/adult-income-prediction-classification)
   - [Craigslist Cars/Trucks Dataset](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data)
2. Download and unzip each dataset.
3. Place the files in the `data/` folder at the project root:
   ```
   Assignment2AppMLDS/
   └── data/
       ├── data.csv        ← Adult Income dataset
       └── vehicles.csv    ← Craigslist Cars/Trucks dataset
   ```

---

## Machine Learning Concepts & Definitions

### Feature Engineering & Analysis

| Concept | Definition |
|---------|-----------|
| **Continuous Features** | Numeric variables that can take any value within a range (e.g., age, hours-per-week, odometer). |
| **Categorical Features** | Variables with a finite set of discrete categories (e.g., workclass, fuel type, manufacturer). |
| **Feature Derivation** | Creating new features from existing ones to capture more meaningful signals (e.g., `capital-net = capital-gain - capital-loss`). |
| **Feature Selection** | Choosing which features (columns) to keep and which to drop before training. Removes noise and irrelevant signals so the model focuses on what actually matters. Can be done manually (dropping known-irrelevant columns), via correlation analysis, or automatically by models like Lasso that drive weak feature coefficients to exactly zero. |
| **Correlation Analysis** | Measuring the linear relationship between features and the target variable to inform feature selection. |

### Data Preprocessing

| Concept | Definition |
|---------|-----------|
| **StandardScaler** | Transforms numeric features to have zero mean and unit variance by subtracting the mean and dividing by the standard deviation. Essential for Logistic Regression (gradient-based optimisation) and KNN (distance-based classification) — without it, features on larger numeric scales dominate and distort the model. Tree-based models are scale-invariant and do not require it. |
| **OneHotEncoding** | Converts each categorical string column into a set of binary (0/1) columns — one per unique category value (e.g. `sex` → `sex_Male`, `sex_Female`). Required because ML models need numeric input. Using ordinal integers instead (Male=0, Female=1) would imply a false numeric ordering between categories. `handle_unknown='ignore'` ensures unseen test categories produce all-zero columns rather than an error. |
| **ColumnTransformer** | Applies different preprocessing steps to different column subsets simultaneously — StandardScaler to numeric columns and OneHotEncoder to categorical columns in one step. Used as the first stage of each Pipeline so that transformations are fitted only on training data and applied to test data, preventing data leakage during cross-validation and evaluation. |
| **Train-Test Split** | Partitioning data into separate training and test sets to evaluate generalization performance on unseen data. |
| **Stratified Split** | Ensures the class distribution in train/test sets mirrors the original dataset — critical for imbalanced data. |

### Class Imbalance

| Concept | Definition |
|---------|-----------|
| **Class Imbalance** | When one class significantly outnumbers another (here: 75% ≤50K vs 25% >50K). Models tend to be biased toward the majority class. |
| **SMOTE (Synthetic Minority Oversampling Technique)** | Generates synthetic samples for the minority class by interpolating between existing minority samples. Applied only to training data to avoid data leakage. |
| **F1 Score** | Harmonic mean of precision and recall. More informative than accuracy for imbalanced datasets since it penalizes both false positives and false negatives. |

### Evaluation Tools

| Concept | Definition |
|---------|-----------|
| **Confusion Matrix** | A grid that shows the counts of correct and incorrect predictions broken down by class. For binary classification it has four cells: **True Negatives** (predicted <=50K, actually <=50K), **False Positives** (predicted >50K, actually <=50K), **False Negatives** (predicted <=50K, actually >50K), and **True Positives** (predicted >50K, actually >50K). Precision, Recall, and F1 are all derived from these four counts. Visualised as a heatmap so misclassification patterns are easy to spot at a glance. |
| **Classification Report** | A text table produced by scikit-learn (`classification_report`) that computes Precision, Recall, F1, and Support for each class from the confusion matrix counts, plus macro and weighted averages across all classes. |

### Model Training & Evaluation

| Concept | Definition |
|---------|-----------|
| **Pipeline** | Chains preprocessing and model steps into a single object, ensuring consistent transformations and preventing data leakage during cross-validation. |
| **Cross-Validation (CV)** | Evaluates model performance by splitting training data into k equal folds, training on k−1 folds and validating on the remaining one, rotating until every fold has been the validation set once. The scores are averaged for a more reliable estimate than a single split. `cv=5` means 5 folds — each fold is 20% of the training data. |
| **GridSearchCV** | Exhaustive search over specified hyperparameter combinations using cross-validation to find the optimal configuration. Tries every combination, scores each with CV, and exposes the winner via `best_params_` and `best_score_`. |
| **Hyperparameter Tuning** | Optimizing model configuration parameters that are not learned from data — such as regularization strength (alpha), tree depth, or number of neighbors. These must be set before training and searched via GridSearchCV. |
| **Overfitting** | When a model learns noise in the training data and fails to generalize to unseen data. Detected when training performance greatly exceeds test performance. Regularization, cross-validation, and pruning (for trees) help prevent it. |
| **Regularization** | Adding a penalty term to the model's cost function to constrain coefficient size, preventing overfitting. The model minimises: `Total cost = Prediction error + λ × penalty`. A higher λ (alpha) = stronger penalty = simpler model. Two types: **L1** (Lasso) penalises the sum of absolute coefficient values and can drive weak ones to exactly zero. **L2** (Ridge) penalises the sum of squared coefficient values and shrinks all coefficients proportionally but never to zero. |
| **Alpha (λ)** | The regularization strength hyperparameter in Ridge and Lasso. Controls how hard the penalty squeezes the coefficients. `alpha=0` means no penalty (plain linear regression). Higher alpha = more shrinkage = simpler model but potentially underfitting. Tuned via GridSearchCV. |
| **Coefficients** | The values a linear model learns for each feature, representing the direction and magnitude of that feature's impact on the prediction. A positive coefficient means the feature increases the predicted value (e.g. `manufacturer_ferrari` raises price). A negative coefficient means it decreases the predicted value (e.g. `vehicle_age` lowers price). Features with a coefficient of zero (Lasso) have been excluded from the model. |
| **ElasticNet** | A regularization method that combines both L1 and L2 penalties. Useful when you want Lasso's feature selection behaviour but with more stable coefficient estimates when features are correlated. Not used in this assignment but a common next step. |

### Classification Models Used

| Model | How It Works | Strengths |
|-------|-------------|-----------|
| **Logistic Regression** | Models the probability of class membership using a logistic (sigmoid) function with a linear decision boundary. | Interpretable, fast, works well with regularization. |
| **K-Nearest Neighbors (KNN)** | Classifies a sample based on the majority class of its K closest neighbors in feature space. | Non-parametric, captures non-linear boundaries. |
| **Decision Tree** | Recursively splits data on feature thresholds that maximize information gain/purity. | Handles interactions, interpretable, no scaling needed. |

### Regression Models Used

| Model | How It Works | Strengths |
|-------|-------------|-----------|
| **Ridge Regression** | Linear regression with L2 penalty (sum of squared coefficients). Shrinks all coefficients proportionally toward zero but never to exactly zero — every feature stays in the model. Squaring the coefficients means large ones are penalised much more than small ones, so Ridge aggressively controls big coefficients while barely affecting small ones. Best when most features are genuinely useful. | Handles multicollinearity, stable estimates. |
| **Lasso Regression** | Linear regression with L1 penalty (sum of absolute coefficient values). The absolute value penalty creates sharp corners in the cost function where the mathematical optimum lands exactly at zero for weak features — effectively removing them. Performs automatic feature selection. Best when you suspect many features are irrelevant. | Built-in feature selection, sparse solutions. |
| **Gradient Boosting** | Builds an ensemble of trees sequentially, where each tree corrects the residual errors of the previous ones. | High predictive power, handles non-linearity. |
| **Random Forest** | Builds many independent decision trees on bootstrapped samples and averages predictions. | Robust to overfitting, captures interactions. |

### Evaluation Metrics

| Metric | Used For | Definition |
|--------|----------|-----------|
| **Accuracy** | Classification | Fraction of correct predictions. Can be misleading for imbalanced data. |
| **Precision** | Classification | Of all positive predictions, how many are actually positive. |
| **Recall** | Classification | Of all actual positives, how many were correctly identified. |
| **F1 Score** | Classification | Harmonic mean of precision and recall: $F1 = 2 \cdot \frac{precision \cdot recall}{precision + recall}$ |
| **R² (Coefficient of Determination)** | Regression | Proportion of the variance in the target that the model explains. R²=1.0 means perfect predictions. R²=0.0 means the model does no better than predicting the mean every time. R²=0.83 (Gradient Boosting here) means 83% of the variation in vehicle prices is explained by the model. Negative R² means the model is worse than the mean baseline. |
| **RMSE (Root Mean Squared Error)** | Regression | Square root of the average squared prediction errors. Expressed in the same units as the target (dollars here), so RMSE=$6,018 means predictions are off by ~$6,018 on average. Squaring before averaging means large errors are penalised more heavily than small ones — one $30,000 miss hurts more than thirty $1,000 misses. |
| **MAE (Mean Absolute Error)** | Regression | Average of the absolute differences between predictions and actuals. Also in the same units as the target. MAE=$3,460 means the typical prediction is off by $3,460. Unlike RMSE, every error is weighted equally — a $30,000 miss is just 30× worse than a $1,000 miss, not 900×. More interpretable and more robust to outliers than RMSE. |

---

## Choices Made & Rationale

### Part 1: Income Classification

| Decision | Choice | Why |
|----------|--------|-----|
| Features dropped | `fnlwgt`, `education`, `country` | Census weight (not predictive), redundant with `education-num`, too many categories respectively. |
| Derived feature | `capital-net` | Combines two sparse features (capital-gain, capital-loss) into one meaningful signal. |
| Imbalance handling | SMOTE | Moderate imbalance (3.15:1). SMOTE creates synthetic minority samples without losing majority class information. |
| Models chosen | Logistic Regression, KNN, Decision Tree | Covers linear, instance-based, and tree-based approaches for comprehensive comparison. |
| Scoring metric | F1 Score | More appropriate than accuracy for imbalanced classification. |
| Best model | Decision Tree | Highest F1 (0.68) and accuracy (82%) on test set. Captures non-linear feature interactions. |

### Part 2: Price Prediction

| Decision | Choice | Why |
|----------|--------|-----|
| Subsampling | 50,000 rows | Full dataset (426K) is too large for iterative development; subsampling recommended by assignment. |
| Price filtering | $500–$100,000 | Removes scam listings ($0, $1) and unrealistic outliers. |
| Year filtering | 1990–2021 | Focuses on modern vehicles with consistent market behavior. |
| Feature excluded | `model`, `VIN`, `description`, `region` | High cardinality, identifier, free text, and geographic noise respectively. |
| Linear models | Ridge + Lasso | Ridge for stable predictions; Lasso for automatic feature selection via L1 sparsity. |
| Non-linear models | Gradient Boosting + Random Forest | Tree ensembles capture complex non-linear pricing relationships. |
| Best model | Gradient Boosting (R²=0.83) | 32% improvement over linear models. Iterative error correction excels at this task. |

---

## Results Summary

### Classification (Adult Income)

| Model | CV F1 | Test Accuracy | Test F1 (>50K) |
|-------|-------|---------------|----------------|
| Logistic Regression | 0.671 | 79.9% | 0.671 |
| K-Nearest Neighbors | 0.657 | 81.0% | 0.669 |
| **Decision Tree** | **0.673** | **81.7%** | **0.679** |

### Regression (Vehicle Prices)

| Model | CV R² | Test R² | Test RMSE | Test MAE |
|-------|-------|---------|-----------|----------|
| Ridge (Linear) | 0.631 | 0.626 | $8,893 | $6,134 |
| Lasso (Linear) | 0.631 | 0.626 | $8,891 | $6,133 |
| Random Forest | 0.806 | 0.805 | $6,413 | $3,772 |
| **Gradient Boosting** | **0.828** | **0.829** | **$6,018** | **$3,460** |

---

## Project Structure

```
Assignment2AppMLDS/
├── DSCI 631-assignment2.ipynb   # Main notebook with all code and analysis
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── data/
│   ├── data.csv                  # Adult Income dataset
│   └── vehicles.csv              # Craigslist Cars/Trucks dataset
└── .venv/                        # Python virtual environment
```

## Setup & Reproduction

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download datasets — see "Dataset Setup" section above
# Place data/data.csv and data/vehicles.csv in the data/ folder

# Open the notebook
jupyter notebook "DSCI 631-assignment2.ipynb"
```

## Dependencies

- pandas, numpy — Data manipulation
- scikit-learn — ML models, preprocessing, evaluation
- imbalanced-learn — SMOTE for class imbalance
- matplotlib, seaborn — Visualization
- kaggle — Dataset download

---

**Author:** Sudhaman Chandrasekaran  
**Course:** DSCI 631 — Applied Machine Learning, Drexel University
