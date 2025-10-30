"""
Script to find the best ML model parameters for each dataset and sample size
"""

import os
import warnings
from multiprocessing import Pool

from src.exp_setup import dataset_list, seed_list, dataset_percentage_list, model_list, test_size
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning
from sklearn.model_selection import train_test_split

from src.utils.dataset_utils import get_dataset_sample, preprocess_dataset, limit_dataset_size
from src.utils.model_utils import create_models, reset_seed

results_header = ['Seed', 'Dataset', 'Sample Size', 'Model', 'Model Parameters']

number_of_tunings = len(dataset_percentage_list) * len(model_list)

def run_exp(setup: list):
    try:
        # Ignore warnings
        os.environ['PYTHONWARNINGS'] = 'ignore'
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

        results = []

        setup_name = setup[0]
        seed = setup[1]
        dataset_setup = setup[2]

        tuning_number = 1
        dataset_name = dataset_setup[0]
        useful_columns = dataset_setup[1]
        class_name = dataset_setup[2]

        # Loading the dataset
        dataset_folder = f"./datasets_others/{dataset_name}"
        csv_files = [file for file in os.listdir(dataset_folder) if file.endswith('.csv')]
        if not csv_files:
            return
        full_df = pd.read_csv(f"{dataset_folder}/{csv_files[0]}")

        if useful_columns:
            full_df.drop(columns=useful_columns, inplace=True)

        # Limiting dataset size
        full_df = limit_dataset_size(full_df, class_name, seed)

        for dataset_percentage in dataset_percentage_list:
            df = get_dataset_sample(full_df, seed, dataset_percentage, class_name, test_size)

            if df is None:
                tuning_number += len(model_list)
                continue

            df = preprocess_dataset(df)
            # df[class_name] = LabelEncoder().fit_transform(df[class_name])

            X = df.drop(class_name, axis=1)
            y = df[class_name]

            x_train, x_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=seed, stratify=y
            )

            reset_seed(seed)
            tuned_models = create_models(seed, x_train, y_train)

            for (model_name, model_parameters_desc) in tuned_models:
                results.append([seed, dataset_name, dataset_percentage, model_name, model_parameters_desc])
                results_df = pd.DataFrame(results, columns=results_header)
                results_df.to_csv(f"results/results_stage_ml_tune_setup_{setup_name}.csv", index=False)
                tuning_number += 1

    except Exception as e:
        print(f"Error in setup {setup}: {e}")

def main():
    setups_tu_parallelize = []
    for seed in seed_list:
        dataset_no = 0
        for dataset_setup in dataset_list:
            dataset_no += 1
            setups_tu_parallelize.append([f"seed_{seed}_dataset_{dataset_no}", seed, dataset_setup])

    # Determine the number of processes. It's often beneficial to set this to the number of CPU cores.
    max_workers = os.cpu_count() or 4  # Fallback to 4 if os.cpu_count() returns None

    with Pool(processes=max_workers) as pool:
        pool.map(run_exp, setups_tu_parallelize)

    # contenting all csv files into one
    all_files = os.listdir("results")
    all_files = [file for file in all_files if file.startswith("results_stage_ml_tune_setup_")]
    all_files = sorted(all_files)
    all_results = pd.concat([pd.read_csv(f"results/{file}") for file in all_files], ignore_index=True)
    all_results.to_csv("results/results_stage_ml_tune.csv", index=False)

    # removing individual csv files
    for file in all_files:
        os.remove(f"results/{file}")

if __name__ == "__main__":
    main()
