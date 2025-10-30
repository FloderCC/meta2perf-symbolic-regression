"""
Script to evaluate base models model for MCC inference
"""

import logging
import os
import random
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, make_scorer, matthews_corrcoef
from sklearn.model_selection import train_test_split

def train_test_split_regression(X, y, test_size=0.2, b='auto', random_state=42):
    # print(f'y = {y}')
    if isinstance(b, str):
        bins = np.histogram_bin_edges(y, bins=b)
        # remove the last index (end point)
        bins = bins[:-1]
    elif isinstance(b, int):
        bins = np.linspace(min(y), max(y), num=b, endpoint=False)
    else:
        raise Exception(f'Undefined bins {b}')

    # print(f'Bins: {bins}')
    groups = np.digitize(y, bins)
    # print(f'Group: {groups}')
    return train_test_split(X, y, test_size=test_size, stratify=groups, random_state=random_state)

# Configure logging
class GreenStreamHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            message = self.format(record)
            green_message = f"\033[92m{message}\033[0m"  # ANSI escape code for green
            self.stream.write(green_message + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(processName)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler("./logs/explain_best_bm.txt", mode='w'),
        GreenStreamHandler()
    ]
)

random_seed = 42

# Set seeds
np.random.seed(random_seed)
random.seed(random_seed)

are_transformations_enabled = True

warnings.filterwarnings("ignore")

df_meta = pd.read_csv('./results/meta_dataset.csv')
df_meta = df_meta.drop(columns=["Seed", "Dataset", "Sample Size", "Model"])
# Defining the target column
target_column = 'MCC'

# now for the filtered version
df_meta = df_meta[df_meta['MCC'] > 0]


# Defining the regression score
def smape_score(true, pred):
    return np.mean(np.abs(pred - true) / ((np.abs(true) + np.abs(pred)) / 2))

# split into train and test sets
X = df_meta.iloc[:, :-1]  # Features
y = df_meta.iloc[:, -1]  # Target variable
# Apply the function
X_train, X_test, y_train, y_test = train_test_split_regression(X, y, test_size=0.2, b='auto')
# Recombine into train/test DataFrames
df_train = pd.DataFrame(X_train, columns=X.columns)
df_train[target_column] = y_train
df_test = pd.DataFrame(X_test, columns=X.columns)
df_test[target_column] = y_test


k = df_train.shape[1] - 1  # Number of predictors

# Extract features and target variable
X_train = df_train.iloc[:, :-1].values  # Features
y_train = df_train.iloc[:, -1].values  # Target variable

def r2(y_true, y_pred, sample_weight):
    try:
        return r2_score(y_true, y_pred, sample_weight=sample_weight) # Minimize the negative to maximize R^2
    except Exception as e:
        print(f"Error in r2_score: {e}")
        return -1

def eval_model(model):
    y_train = df_train.iloc[:, -1].values  # Target variable

    # Predict training set
    y_pred_train = model.predict(X_train)
    train_r2 = r2_score(y_train, y_pred_train)
    train_mape = smape_score(y_train, y_pred_train)
    train_mae = mean_absolute_error(y_train, y_pred_train)

    n = len(y_train)  # Total samples
    train_adj_r2 = 1 - (1 - train_r2) * ((n - 1) / (n - k - 1))

    # Predict test set
    X_test = df_test.iloc[:, :-1].values  # Features
    y_test = df_test.iloc[:, -1].values  # Target variable
    y_pred_test = model.predict(X_test)
    test_r2 = r2_score(y_test, y_pred_test)
    test_mape = smape_score(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)

    n = len(y_test)
    test_adj_r2 = 1 - (1 - test_r2) * ((n - 1) / (n - k - 1))

    # Logging results
    logging.info(f"Results for the model {model.__class__.__name__}:")
    logging.info(f"Train dataset ({len(y_train)} rows): R^2: {round(train_r2, 3)}, Adjusted R^2: {round(train_adj_r2, 3)}, sMAPE: {round(train_mape, 3)}, MAE: {round(train_mae, 3)}")
    logging.info(f"Test dataset ({len(y_test)} rows): R^2: {round(test_r2, 3)}, Adjusted R^2: {round(test_adj_r2, 3)}, sMAPE: {round(test_mape, 3)}, MAE: {round(test_mae, 3)}")
    logging.info(f"Selected hyperparameters: {model.hyperparameters}")


# defining hyperparameter grids for each model
from sklearn.model_selection import GridSearchCV
param_grids = {
    'RandomForestRegressor': {
        'n_estimators': [50, 100],
        'max_depth': [8, 16, 64, 128],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2', 0.5],
        'bootstrap': [True, False],
    },
}

from sklearn.ensemble import  RandomForestRegressor
model = RandomForestRegressor(random_state=random_seed, n_jobs=-1)

# Hyperparameter tuning and evaluation
model_name = model.__class__.__name__
print(f"Tuning model: {model_name}")

param_grid = param_grids[model_name]
grid_search = GridSearchCV(model, param_grid, scoring="neg_mean_squared_error", cv=4, n_jobs=-1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
best_model.hyperparameters = grid_search.best_params_

eval_model(best_model)

logging.info("Hyperparameter tuning and evaluation completed.")

logging.info("Explaining the final model with best hyperparameters.")

from shap import KernelExplainer, kmeans
logging.getLogger("shap").setLevel(logging.ERROR)
logging.getLogger("shap").propagate = False

def shap_explanation(X, model):
    background_data = kmeans(X, 10)
    explainer = KernelExplainer(model.predict, background_data)
    shap_values = explainer.shap_values(X, nsamples=100)
    return np.mean(np.abs(shap_values), axis=0)

# creating the new model with best hyperparameters
final_model = RandomForestRegressor(
    random_state=random_seed,
    n_jobs=-1,
    **best_model.hyperparameters
)
final_model.fit(X_train, y_train)

te_f_relevance = shap_explanation(X_test, final_model)

logging.info("SHAP explanation completed. Feature relevance scores:")
logging.info(str(te_f_relevance.tolist()))

logging.info("Input feature names:")
logging.info(str(df_train.columns[:-1].tolist()))