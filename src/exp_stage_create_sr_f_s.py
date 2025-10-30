"""
Script to find a symbolic regression model for MCC inference
"""

import logging
import os
import random
import warnings
import numpy as np
import pandas as pd
from HROCH import SymbolicRegressor
from sklearn.metrics import r2_score, mean_absolute_error
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
        logging.FileHandler("./logs/exp_stage_create_sr_f_s.txt", mode='w'),
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

# applying max min scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_meta.iloc[:, :-1] = scaler.fit_transform(df_meta.iloc[:, :-1])





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

# Define the Symbolic Regressor model
model = SymbolicRegressor(
    num_threads=os.cpu_count(),
    random_state=42,
    verbose=1,
    metric='MSE',
    time_limit=259200,
    # iter_limit=10000,
    iter_limit=0,
    problem='math',
    precision='f64',
    population_settings = {'size': 1000, 'tournament':500},
    init_const_settings = {'const_min':-1.0, 'const_max':5.0},
    algo_settings={
        'neighbours_count': 30,
        # Increased from default (15) to improve local search by evaluating more neighbors per iteration,
        # which enhances the algorithm's ability to explore complex symbolic structures.

        'alpha': 0.01,
        # A stricter acceptance threshold for worse solutions. By setting alpha low, we discourage accepting
        # degraded candidates, which promotes convergence toward higher-quality symbolic expressions.

        'beta': 0.9,  # A high breadth-expansion factor that encourages wider tree exploration.
        # This helps generate more diverse candidate expressions and is especially useful in discovering
        # complex relationships in high-dimensional or nonlinear data.

        'pretest_size': 2,
        # Each unit corresponds to a 64-row minibatch, so this uses 128 rows for fast pre-evaluation.
        # This pre-filtering stage reduces computation by discarding weak candidates before full evaluation.

        'sample_size': 200  # Uses sample_size = 25837 // 64 = 403 samples (about 99.8% of your dataset).
        # This is a robust sample size that offers a good trade-off between computational efficiency
        # and statistical stability in score estimation during training.
    }
)

# Train the model
model.fit(X_train, y_train)

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
logging.info(f"Final results:")
logging.info(f"Train dataset ({len(y_train)} rows): R^2: {round(train_r2, 3)}, Adjusted R^2: {round(train_adj_r2, 3)}, sMAPE: {round(train_mape, 3)}, MAE: {round(train_mae, 3)}")
logging.info(f"Test dataset ({len(y_test)} rows): R^2: {round(test_r2, 3)}, Adjusted R^2: {round(test_adj_r2, 3)}, sMAPE: {round(test_mape, 3)}, MAE: {round(test_mae, 3)}")

logging.info("Equation:")
logging.info(str(model.equation))  # Displays the final symbolic expression