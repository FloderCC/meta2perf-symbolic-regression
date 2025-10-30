"""
Script for training and evaluating tuned ML models
"""
import gc
import os
import warnings

import pandas as pd
import tensorflow as tf
from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split

from src.utils.dataset_utils import get_dataset_sample, preprocess_dataset, limit_dataset_size
from src.exp_setup import dataset_list, seed_list, dataset_percentage_list, model_list, test_size
from src.utils.model_utils import reset_seed, train_best_model

# Ignore warnings
os.environ['PYTHONWARNINGS'] = 'ignore'
# Ignore warnings of type UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
# Ignore warnings of type ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
# Ignore warnings of type FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)
# Ignore warnings of type UserWarning
warnings.filterwarnings("ignore", category=UserWarning)


# loading the results dataset to fill the model performance
df_name = f"results/results_stage_ml_tune.csv"
results_df = pd.read_csv(df_name)
results_df_to_iterate = results_df.copy()

# counter
training_number = 1
number_of_tunings = len(results_df)

print(f"Number of tunings to be done: {number_of_tunings}")

# iterating over the results_df_to_iterate to read from each row the Seed,Dataset,Sample Size,Model, and Model Parameters
for index, row in results_df_to_iterate.iterrows():
    seed = row['Seed']
    dataset_name = row['Dataset']
    dataset_percentage = row['Sample Size']
    model_name = row['Model']
    model_parameters = row['Model Parameters']

    # searching for the useful_columns in the dataset_list
    useful_columns = None
    class_name = None
    for dataset_setup in dataset_list:
        if dataset_setup[0] == dataset_name:
            useful_columns = dataset_setup[1]
            class_name = dataset_setup[2]
            break

    # loading the dataset
    dataset_folder = f"./datasets/{dataset_name}"
    full_df = pd.read_csv(
        f"{dataset_folder}/{[file for file in os.listdir(dataset_folder) if file.endswith('.csv')][0]}")

    if len(useful_columns) > 0:
        print(f"Removing columns {useful_columns}")
        full_df.drop(columns=useful_columns, inplace=True)

    # limiting dataset size to 1m rows
    full_df = limit_dataset_size(full_df, class_name)

    print(f"\033[92mStarted execution with dataset {dataset_name} {full_df.shape}\033[0m")

    # splitting the dataset
    df = get_dataset_sample(full_df, seed, dataset_percentage, class_name, test_size)

    print(f"\033[92m\nStarted execution with dataset sample size {dataset_percentage} ({df.shape[0]} rows)\033[0m")

    # codify & prepare the dataset
    print("Codifying & preparing dataset ...")
    df = preprocess_dataset(df)
    # df[class_name] = LabelEncoder().fit_transform(df[class_name])

    X = df.drop(class_name, axis=1)
    y = df[class_name]

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    reset_seed(seed)

    # training the model
    print(f"\033[92mStage 2: Training the model {training_number}/{number_of_tunings}\033[0m")
    model = train_best_model(model_name, seed, x_train, y_train, model_parameters)

    # evaluating the model
    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    mcc = matthews_corrcoef(y_test, y_pred)

    # saving the performance metrics
    results_df.loc[(results_df['Seed'] == seed) &
                   (results_df['Dataset'] == dataset_name) &
                   (results_df['Sample Size'] == dataset_percentage) &
                   (results_df['Model'] == model_name), 'Accuracy'] = accuracy
    results_df.loc[(results_df['Seed'] == seed) &
                   (results_df['Dataset'] == dataset_name) &
                   (results_df['Sample Size'] == dataset_percentage) &
                   (results_df['Model'] == model_name), 'Precision'] = precision
    results_df.loc[(results_df['Seed'] == seed) &
                   (results_df['Dataset'] == dataset_name) &
                   (results_df['Sample Size'] == dataset_percentage) &
                   (results_df['Model'] == model_name), 'Recall'] = recall
    results_df.loc[(results_df['Seed'] == seed) &
                   (results_df['Dataset'] == dataset_name) &
                   (results_df['Sample Size'] == dataset_percentage) &
                   (results_df['Model'] == model_name), 'F1'] = f1
    results_df.loc[(results_df['Seed'] == seed) &
                   (results_df['Dataset'] == dataset_name) &
                   (results_df['Sample Size'] == dataset_percentage) &
                   (results_df['Model'] == model_name), 'MCC'] = mcc

    # exporting the results
    results_df.to_csv(f"results/results_stage_ml_eval.csv", index=False)

    training_number += 1

    # cleaning up
    del model, y_pred, x_train, x_test, y_train, y_test, X, y, df, full_df
    gc.collect()
