# IronKaggle Predictive Sales Modeling

## Aim
The aim of this challenge is to predict daily sales for various stores using machine learning.

---

## Dataset
The dataset is available at:  
https://raw.githubusercontent.com/data-bootcamp-v4/data/main/sales.csv

### Metadata

| Column | Description |
|------|------------|
| shop_ID | Unique identifier for each shop |
| day_of_the_week | 1–7, representing the day of the week |
| date | Date of the observation |
| number_of_customers | Number of customers visiting the shop |
| open | 0 = closed, 1 = open |
| promotion | 0 = no promotion, 1 = promotion |
| state_holiday | 0, ‘a’, ‘b’, ‘c’ for state holidays (0 = none) |
| school_holiday | 0 = no, 1 = yes |

---

## Data Exploration
Before creating features, the dataset was inspected to ensure quality:

- Check for missing values (NaN) in all columns.
- Unique value counts for categorical variables (e.g., `store_id`, `state_holiday`).
- Distribution of numeric variables (`number_of_customers`) using box plots to detect outliers.

This step ensures the dataset is clean and ready for feature engineering and modeling.

---

## Feature Engineering
New features were derived to improve model performance:

- State holiday encoded as dummy variables.
- Date-based features extracted from the `date` column.

Additional temporal information was extracted from the date column to provide the model with more context about sales patterns over time. A binary feature, `is_weekend`, was also created to indicate whether a given day falls on a weekend, capturing potential changes in customer behavior.

### Engineered Features

| Feature | Description / Purpose |
|-------|------------------------|
| year | Captures yearly trends |
| month | Captures seasonal patterns |
| week_of_year | Detects weekly and seasonal cycles |
| weekday | Models weekday-specific effects |
| is_weekend | Indicates weekend-related sales behavior |

These transformations allow the model to capture temporal patterns and categorical effects in sales.

---

## Model Selection – LightGBM Regressor
LightGBM Regressor is a fast gradient boosting, tree-based model that builds trees sequentially, with each tree correcting the errors of the previous ones. It efficiently handles large datasets and high-cardinality categorical features internally, avoiding costly one-hot encoding, reducing memory usage, and enabling more efficient splits than scikit-learn trees—making it well suited for store sales prediction.

---

## Hyperparameter Choice

| Hyperparameter | Values Tested | Description |
|---------------|--------------|-------------|
| n_estimators | 300, 500 | Number of boosting trees built sequentially |
| learning_rate | 0.05, 0.1 | Contribution of each tree to the final model |
| num_leaves | 31, 50 | Controls tree complexity |
| max_depth | -1, 6 | Maximum tree depth (`-1` = unlimited) |
| min_child_samples | 20, 30 | Minimum samples per leaf to reduce overfitting |

- `RandomizedSearchCV` with 10 iterations was used to efficiently explore the hyperparameter space.

---

## Validation Strategy
- `TimeSeriesSplit` was used to respect the temporal nature of the data.
- Rationale:
  - Many features are derived from the date column.
  - Maintaining chronological order avoids data leakage.
  - Ensures realistic evaluation by testing on future data unseen during training.
- Metrics used: **CV Best R²**, **Test R²**, **RMSE**, **MAE**.
- Visualization: Predicted vs. True sales scatter plots to assess model fit.

---

## Prediction
- The final model was used to generate predictions on the unseen dataset:  
  https://raw.githubusercontent.com/data-bootcamp-v4/data/main/ironkaggle_notarget.csv

- Predictions were compared against actual sales values from the official solutions file:  
  https://raw.githubusercontent.com/data-bootcamp-v4/data/main/ironkaggle_solutions.csv

- Model performance was assessed using **R²**, **RMSE**, and **MAE**.

---

## Deliverables
- Trained LightGBM model with tuned hyperparameters.
- Predictions on the test dataset ready for evaluation.
- Plots showing predicted vs. true sales for model diagnostics.
- Evaluation metrics: **R²**, **RMSE**, **MAE**.
