"""
Script for dataset description
"""

import gc
import os
import warnings

import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning

from src.utils.dataset_utils import get_dataset_sample, preprocess_dataset, limit_dataset_size
from src.exp_setup import dataset_list, test_size
from src.utils.dataset_utils import describe_dataset_using_pymfe

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


# loading the results dataset to fill the dataset descriptions
df_name = f"results/results_stage_ml_tune.csv"
# df_name = f"results/results_stage_ml_tune_small_test.csv"
results_df = pd.read_csv(df_name)

# removing columns that are not useful for the description (all but 'Seed', 'Dataset', 'Sample Size')
results_df.drop(columns=[col for col in results_df.columns if col not in ['Seed', 'Dataset', 'Sample Size']], inplace=True)
# removing duplicated rows
results_df.drop_duplicates(inplace=True)

# counter
training_number = 1
number_of_tunings = results_df.shape[0]

print(f"Number of descriptions to be done: {number_of_tunings}")

# iterating over the results_df_to_iterate to read from each row the Seed,Dataset,Sample Size,Model, and Model Parameters
for index, row in results_df.copy().iterrows():
    seed = row['Seed']
    dataset_name = row['Dataset']
    dataset_percentage = row['Sample Size']

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
    full_df = limit_dataset_size(full_df, class_name, seed)

    # splitting the dataset
    df = get_dataset_sample(full_df, seed, dataset_percentage, class_name, test_size)

    print(f"\033[92m\nStarted execution with {seed},{dataset_name},{dataset_percentage} ({df.shape[0]} rows)\033[0m")

    # codify & prepare the dataset
    print("Codifying & preparing dataset ...")
    df = preprocess_dataset(df)

    attr_value: dict = describe_dataset_using_pymfe(df, class_name, seed)

    # saving dataset descriptions
    for k, v in attr_value.items():
        results_df.loc[(results_df['Seed'] == seed) &
                       (results_df['Dataset'] == dataset_name) &
                       (results_df['Sample Size'] == dataset_percentage), k] = v

    # exporting the results
    results_df.to_csv(f"./results/results_stage_dataset_description.csv", index=False)

    training_number += 1

    # cleaning up
    del df
    gc.collect()
