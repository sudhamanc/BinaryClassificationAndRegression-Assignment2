# DSCI 631 — Assignment 2: Notebook Questions & Answers

---

## Part 1: Binary Classification — Adult Income Dataset

---

### Question 1-1
*Examine the features of the data. Identify which feature is continuous and which feature is categorical. Make some analyses and statistics, then use the results to discuss your selected predictors. You may also derive new features.*

**Dataset:** 32,561 rows × 15 columns

**Continuous features (6):** `age`, `fnlwgt`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`

**Categorical features (8):** `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `country`

**Continuous feature statistics:**

| Feature | Mean | Std | Min | Max |
|---|---|---|---|---|
| age | 38.58 | 13.64 | 17 | 90 |
| education-num | 10.08 | 2.57 | 1 | 16 |
| capital-gain | 1077.65 | 7385.29 | 0 | 99,999 |
| capital-loss | 87.30 | 402.96 | 0 | 4,356 |
| hours-per-week | 40.44 | 12.35 | 1 | 99 |

**Selected Predictors:**
- `age` — older individuals tend to earn more (moderate correlation with salary)
- `education-num` — higher education strongly correlates with higher income
- `hours-per-week` — more working hours associated with higher salary
- `capital-gain` / `capital-loss` — strong indicators of wealth/income level
- `marital-status` — married individuals (especially husbands) tend to earn more
- `occupation` — certain occupations pay significantly more
- `workclass` — type of employment matters

**Excluded:**
- `fnlwgt` — census sampling weight, not a meaningful predictor
- `education` — redundant with `education-num` (numeric encoding)
- `country` — too many categories with most being United-States

**Derived Feature:**
- `capital-net` = `capital-gain` − `capital-loss` — combines two sparse features into one meaningful signal
  - Mean: 990.35, Std: 7408.99, Min: −4356, Max: 99,999

---

### Question 1-2
*Check if the target label is balanced or not, what is your strategy if imbalanced? Apply your strategy.*

**Target Distribution:**

| Class | Count | Percentage |
|---|---|---|
| 0 (<=50K) | 24,720 | 75.9% |
| 1 (>50K) | 7,841 | 24.1% |

**Imbalance ratio: 3.15:1**

**Strategy: SMOTE (Synthetic Minority Oversampling Technique)**

The dataset is moderately imbalanced with ~75% of samples in class 0 (<=50K) and ~25% in class 1 (>50K).

- SMOTE generates synthetic samples for the minority class by interpolating between existing minority samples
- Applied **only to training data** (not test data) to avoid data leakage
- Integrated into the pipeline using `imblearn`'s `Pipeline`
- Stratified train/test split used to maintain class proportions in both sets

---

### Question 1-3
*Split the data into train and test data, then build Pipeline, apply transformer as needed, train 3 algorithms with hyperparameter tuning, compare performance on test set. Discuss results.*

**Train/Test Split (stratified):**
- Training set: 26,048 rows (80%)
- Test set: 6,513 rows (20%)
- Training class distribution: 19,775 (<=50K) / 6,273 (>50K)
- Test class distribution: 4,945 (<=50K) / 1,568 (>50K)

**Preprocessing (inside Pipeline):**
- `StandardScaler` on numeric features: `age`, `education-num`, `capital-net`, `hours-per-week`
- `OneHotEncoder` on categorical features: `workclass`, `marital-status`, `occupation`, `relationship`, `race`, `sex`

---

**Model 1: Logistic Regression**

Best parameters: `C=0.01`, `penalty=l2`, `solver=liblinear`
Best CV F1: 0.6711

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| <=50K | 0.94 | 0.78 | 0.86 | 4,945 |
| >50K | 0.55 | 0.85 | 0.67 | 1,568 |
| accuracy | | | 0.80 | 6,513 |

---

**Model 2: K-Nearest Neighbors**

Best parameters: `metric=manhattan`, `n_neighbors=11`, `weights=uniform`
Best CV F1: 0.6572

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| <=50K | 0.93 | 0.82 | 0.87 | 4,945 |
| >50K | 0.58 | 0.79 | 0.67 | 1,568 |
| accuracy | | | 0.81 | 6,513 |

---

**Model 3: Decision Tree**

Best parameters: `max_depth=15`, `min_samples_leaf=5`, `min_samples_split=2`
Best CV F1: 0.6733

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| <=50K | 0.93 | 0.82 | 0.87 | 4,945 |
| >50K | 0.59 | 0.80 | 0.68 | 1,568 |
| accuracy | | | 0.82 | 6,513 |

---

**Model Comparison:**

| Model | Best CV F1 | Test Accuracy | Test F1 (>50K) |
|---|---|---|---|
| Logistic Regression | 0.6711 | 79.9% | 0.6712 |
| K-Nearest Neighbors | 0.6572 | 81.0% | 0.6686 |
| **Decision Tree** | **0.6733** | **81.7%** | **0.6794** |

**Discussion:**
- All three models were trained with SMOTE to handle class imbalance
- Logistic Regression provides a good baseline with interpretable results
- KNN captures non-linear decision boundaries but is sensitive to feature scaling
- Decision Tree can capture complex interactions and achieved the best F1 score
- The best model is selected based on F1 score for the minority class (>50K), as this metric balances precision and recall for the imbalanced target

---

### Question 1-4
*Identify features that are important from your best model. Which features are most influential? Which features could be removed without decrease in performance? Does removing irrelevant features make the model better?*

**Top 15 Most Important Features (Decision Tree):**

| Feature | Importance |
|---|---|
| marital-status_Married-civ-spouse | 0.4172 |
| education-num | 0.2162 |
| capital-net | 0.1213 |
| age | 0.0968 |
| hours-per-week | 0.0687 |
| occupation_Exec-managerial | 0.0115 |
| workclass_Private | 0.0057 |
| workclass_Self-emp-not-inc | 0.0047 |
| occupation_Prof-specialty | 0.0046 |
| sex_Male | 0.0045 |
| occupation_Other-service | 0.0039 |
| occupation_Craft-repair | 0.0034 |
| occupation_Transport-moving | 0.0033 |
| relationship_Wife | 0.0033 |
| occupation_Sales | 0.0026 |

Features with importance < 0.01 (candidates for removal): **42 features**

**Reduced Model (removing `race` and `sex`):**

| | Original Model | Reduced Model |
|---|---|---|
| F1 (>50K) | 0.6794 | 0.6847 |
| Accuracy | 0.8174 | 0.8225 |

**Conclusion:**
- The most influential features are `capital-net`, `education-num`, `age`, `marital-status (Married-civ-spouse)`, `hours-per-week`, and `relationship (Husband)`
- Features like `race` and `sex` have relatively low importance
- Removing low-importance features (`race`, `sex`) results in **slightly better** performance, confirming they are not essential for prediction
- The derived `capital-net` feature is the third most important, validating the feature engineering decision

---

## Part 2: Regression — Craigslist Vehicle Price Prediction

---

### Question 2-1
*Assemble a dataset from subsampled data. What features are relevant? Are there features that should be excluded due to target leakage? Show visualizations or statistics to support your selection.*

**Full dataset:** 426,880 rows × 26 columns
**Subsample used:** 50,000 rows (random sample, seed=42)

**Missing Values (% in subsample):**

| Feature | Missing % |
|---|---|
| county | 100.0% |
| size | 71.9% |
| cylinders | 41.3% |
| condition | 40.5% |
| VIN | 37.9% |
| paint_color | 30.4% |
| drive | 30.3% |
| type | 21.5% |
| manufacturer | 4.2% |

**Relevant Features Selected:**
- `year` — vehicle age is a primary determinant of price
- `manufacturer` — brand significantly affects price
- `odometer` — mileage is a key indicator of wear/value
- `condition` — vehicle condition directly impacts price
- `cylinders` — engine size correlates with vehicle class/price
- `fuel` — fuel type affects desirability and price
- `title_status` — clean title vs salvage hugely impacts price
- `transmission` — manual/automatic affects price
- `drive` — 4wd/fwd/rwd affects price
- `type` — vehicle type (sedan, truck, SUV) affects price
- `paint_color` — minor effect on price

**Excluded Features:**
- `id`, `url`, `region_url`, `image_url` — identifiers/metadata, not predictive
- `VIN` — unique identifier, not generalizable
- `description` — free text, would require NLP (out of scope)
- `county` — mostly null
- `lat`/`long` — geographic coordinates less useful
- `posting_date` — when listed, not a vehicle attribute
- `model` — too many unique values (high cardinality)
- `region`/`state` — too many categories, minor effect

**Target Leakage:** None of the selected features directly leak price information. VIN could be used to look up price (excluded).

**Data Cleaning Applied:**
- Price filtered to $500–$100,000 (removes scam listings and unrealistic outliers)
- Year filtered to 1990–2021
- Rows with missing odometer dropped
- Derived feature: `vehicle_age` = 2021 − year

**Final cleaned dataset: 43,297 rows × 12 columns**

**Price statistics after cleaning:**
- Mean: $19,288 | Std: $14,354 | Min: $500 | Max: $100,000
- 25th percentile: $7,900 | Median: $15,990 | 75th percentile: $27,990

---

### Question 2-2
*Perform feature selection with a linear model, with appropriate preprocessing and cross-validation. Evaluate the generalization performance.*

**Train/Test Split:** 34,637 train / 8,660 test (80/20)

**Preprocessing:**
- `StandardScaler` on: `odometer`, `vehicle_age`
- `OneHotEncoder` on: `manufacturer`, `condition`, `cylinders`, `fuel`, `title_status`, `transmission`, `drive`, `type`, `paint_color`

---

**Ridge Regression (L2 regularisation):**

Best alpha: 0.01 | Best CV R²: 0.6311

| Metric | Value |
|---|---|
| Test R² | 0.6256 |
| Test RMSE | $8,892.51 |
| Test MAE | $6,133.62 |

Ridge 5-Fold CV R² scores: [0.6144, 0.6400, 0.6326, 0.6411, 0.6272]
Mean CV R²: 0.6311 ± 0.0097

---

**Lasso Regression (L1 regularisation — feature selection):**

Best alpha: 0.1 | Best CV R²: 0.6311

| Metric | Value |
|---|---|
| Test R² | 0.6257 |
| Test RMSE | $8,891.22 |
| Test MAE | $6,132.89 |

Features selected by Lasso (non-zero coefficients): **99 out of 105**

Top 15 features by absolute coefficient:

| Feature | Coefficient |
|---|---|
| manufacturer_ferrari | +59,693 |
| cylinders_12 cylinders | +27,662 |
| manufacturer_tesla | +15,563 |
| fuel_diesel | +13,237 |
| manufacturer_porsche | +12,402 |
| vehicle_age | −7,935 |
| manufacturer_fiat | −7,598 |
| cylinders_10 cylinders | +7,081 |
| condition_new | +6,688 |
| manufacturer_mitsubishi | −6,605 |
| type_convertible | +6,373 |
| title_status_lien | +5,915 |
| manufacturer_rover | +5,606 |
| type_bus | −5,465 |
| manufacturer_kia | −5,127 |

**Interpretation:** Luxury brands (Ferrari, Tesla, Porsche) and diesel/12-cylinder vehicles have strong positive coefficients. Older vehicles (`vehicle_age`) and brands like Fiat and Kia have negative coefficients. Lasso kept 99/105 features — minimal elimination, suggesting most categories carry some signal.

---

### Question 2-3
*Use non-linear regression models to improve results. Tune hyperparameters. What is the best prediction you can get?*

**Preprocessing for tree-based models:**
- Numeric features: `passthrough` (no scaling needed — trees split on thresholds, not distances)
- Categorical features: `OneHotEncoder` (still required — trees need numeric input)

---

**Gradient Boosting Regressor:**

Hyperparameters searched: `n_estimators` [100, 200], `max_depth` [5, 7, 10], `learning_rate` [0.05, 0.1], `subsample` [0.8, 1.0]

| Metric | Value |
|---|---|
| Best CV R² | 0.828 |
| Test R² | 0.829 |
| Test RMSE | $6,018 |
| Test MAE | $3,460 |

---

**Random Forest Regressor:**

Hyperparameters searched: `n_estimators` [100, 200], `max_depth` [10, 15, 20], `min_samples_split` [5, 10]

| Metric | Value |
|---|---|
| Best CV R² | 0.806 |
| Test R² | 0.805 |
| Test RMSE | $6,413 |
| Test MAE | $3,772 |

---

**Full Model Comparison:**

| Model | CV R² | Test R² | Test RMSE | Test MAE |
|---|---|---|---|---|
| Ridge (Linear) | 0.631 | 0.626 | $8,893 | $6,134 |
| Lasso (Linear) | 0.631 | 0.626 | $8,891 | $6,133 |
| Random Forest | 0.806 | 0.805 | $6,413 | $3,772 |
| **Gradient Boosting** | **0.828** | **0.829** | **$6,018** | **$3,460** |

**Discussion:**
- Non-linear models (Gradient Boosting, Random Forest) significantly outperform linear models (Ridge, Lasso) for vehicle price prediction
- The improvement from ~0.63 R² (linear) to ~0.83 R² (Gradient Boosting) shows that price has important non-linear relationships with features — combinations of manufacturer, age, and mileage interact in ways a linear model cannot capture
- Gradient Boosting achieves the best performance by iteratively correcting prediction errors from previous trees
- Key factors driving price: vehicle age, odometer, manufacturer (luxury brands), cylinders, fuel type, and condition
- Further improvements could come from: log-transforming the target, adding interaction features, or using more data

**Best model: Gradient Boosting (Test R² = 0.829, RMSE = $6,018)**

---
