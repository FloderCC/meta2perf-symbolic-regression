"""
This script evaluates a symbolic regression model on a meta-dataset for regression tasks when MCC > 0.
"""

import random
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error

def plot_regression_results(y_true, y_pred, title, target='', set_name='train'):
    """
    Plots true vs. predicted values for regression results, with R² and MAE metrics.

    Parameters:
    - y_true: array-like, true target values
    - y_pred: array-like, predicted target values
    - title: str, plot title

    Saves the figure as 'regression_plot.pdf' in high-resolution vector format.
    """

    # Create plot
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, alpha=0.5, edgecolors='#182e3d', label='Predictions')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)],
        linestyle='--',
        color='#c42525',
        label='Ideal Fit (y = x)')

    # limit the x and y axis to the range of the data
    x_margin = 0.03 * (max(y_true) - min(y_true))
    y_margin = 0.03 * (max(y_pred) - min(y_pred))  # Similarly for y_pred if needed

    plt.xlim(min(y_true) - x_margin, max(y_true) + x_margin)
    plt.ylim(min(y_pred)- y_margin, max(y_pred) + y_margin)
    # plt.ylim(0, 1)

    # Labels and title
    plt.xlabel(f'True {target} Values', fontsize=12)
    plt.ylabel(f'Predicted {target} Values', fontsize=12)

    plt.legend()
    plt.tight_layout()

    # Save and show
    plt.savefig(f'plots/pdf/regression_f_{set_name}.pdf', format='pdf', bbox_inches='tight')
    plt.savefig(f'plots/png/regression_f_{set_name}.png', format='png', bbox_inches='tight', dpi=300)

    plt.show()

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

random_seed = 42

# Set seeds
np.random.seed(random_seed)
random.seed(random_seed)

warnings.filterwarnings("ignore")

df_meta = pd.read_csv('./results/meta_dataset.csv')
df_meta = df_meta.drop(columns=["Seed", "Dataset", "Sample Size", "Model"])
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

# /// Model test begin \\\ #

# --- Symbolic regression expression ---


def predict(df): # claude
    # parsing the features names to x0, x1, x2, ...
    for i, col in enumerate(df.columns):
        df = df.rename(columns={col: f'x{i}'})
        print(col, '->', f'x{i}')

    a = 3.3237551265443805
    b = 1.4435745948155903
    c = 3.8006530371420686
    d = 12.379278274862575
    e = 2.1575551710877265
    f = -0.011386770840701929

    t1 = (df['x13'] - df['x15']) - a
    t2 = df['x11'] + b
    t3 = df['x11'] + (df['x9'] + df['x1'])
    t4 = (df['x14'] + t1) / t1 * df['x15'] * t2
    t5 = (df['x10'] * t4) / t3
    t6 = ((c + ((df['x4'] - (d / df['x6'])) * (df['x13'] - df['x15']))) * (df['x0'] / df['x2'])) ** 2 + (df['x4'] + t2)

    numerator = (df['x17'] + 1) * t5 - t6 - ((df['x5'] + df['x6']) / e) + df['x1']
    denominator = t3 * f * (t4 + (df['x9'] + df['x1'])) - t6

    # Final result
    result = numerator / denominator

    return result


# --- Rename columns to match expression variable names ---
# Converts 'nr_attr' -> 'nrattr', etc.
df_train.columns = [col.lower().replace('_', '') for col in df_train.columns]
df_test.columns = [col.lower().replace('_', '') for col in df_test.columns]

# inferencing the train dataset
y_pred_train = predict(df_train)

# \\\ Model test end/// #

y_train = df_train.iloc[:, -1].values  # Target variable

# Evaluating training set
train_r2 = r2_score(y_train, y_pred_train)
train_mape = smape_score(y_train, y_pred_train)
train_mae = mean_absolute_error(y_train, y_pred_train)

n = len(y_train)  # Total samples
k = df_train.shape[1] - 1  # Number of predictors
train_adj_r2 = 1 - (1 - train_r2) * ((n - 1) / (n - k - 1))

# inferencing the test dataset
y_pred_test = predict(df_test)

y_test = df_test.iloc[:, -1].values  # Target variable
test_r2 = r2_score(y_test, y_pred_test)
test_mape = smape_score(y_test, y_pred_test)
test_mae = mean_absolute_error(y_test, y_pred_test)

n = len(y_test)
test_adj_r2 = 1 - (1 - test_r2) * ((n - 1) / (n - k - 1))

# Logging results
print(f"Final results:")
print(f"Train dataset ({len(y_train)} rows): R^2: {round(train_r2, 3)}, Adjusted R^2: {round(train_adj_r2, 3)}, sMAPE: {round(train_mape, 3)}, MAE: {round(train_mae, 3)}")
print(f"Test dataset ({len(y_test)} rows): R^2: {round(test_r2, 3)}, Adjusted R^2: {round(test_adj_r2, 3)}, sMAPE: {round(test_mape, 3)}, MAE: {round(test_mae, 3)}")

plot_regression_results(y_test, y_pred_test, 'Regression Predictions vs. True Values (Test Set)', 'MCC', 'test')

# # exporting a csv with the real and predicted values
# df_test['y_pred'] = y_pred_test
# df_test['y_true'] = y_test
#
# #remove the columns that are not y_pred or y_true
# df_test = df_test[['y_true', 'y_pred']]
#
# df_test.to_csv('plots/regression_results_test.csv', index=False)
#
# # doing the same with train
# df_train['y_pred'] = y_pred_train
# df_train['y_true'] = y_train
# #remove the columns that are not y_pred or y_true
# df_train = df_train[['y_true', 'y_pred']]
# df_train.to_csv('plots/regression_results_train.csv', index=False)