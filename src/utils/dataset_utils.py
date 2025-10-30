"""
This file contains auxiliary methods sample, preprocess and generate the dataset description
"""

import numpy as np
from pandas import DataFrame
from pymfe.mfe import MFE


def limit_dataset_size(df: DataFrame, class_name: str, seed: int, max_size: int = 1000000) -> DataFrame:
    total_rows = df.shape[0]

    if total_rows > max_size:
        print(f" - Limiting dataset size to {max_size} rows")

        # Calculate the sampling size for each class proportionally
        class_counts = df[class_name].value_counts()
        sampling_sizes = (class_counts / total_rows * max_size).round().astype(int)

        # Sample each class according to the calculated sampling size
        df_sampled = df.groupby(class_name).apply(
            lambda x: x.sample(n=min(len(x), sampling_sizes[x.name]), random_state=seed)).reset_index(drop=True)

        # Adjust in case rounding leads to a mismatch in the final number of samples
        excess = df_sampled.shape[0] - max_size
        if excess > 0:
            df_sampled = df_sampled.sample(n=max_size, random_state=seed)

        return df_sampled
    else:
        return df

def get_dataset_sample(df: DataFrame, seed: int, sample_percent: float, class_name: str, test_size: float) -> DataFrame:
    # for train test split is necessary to ensure the least populated class in y has only > 1 member
    # and the test set has at least 1 member

    # Calculate the number of samples for the least populated class after sampling
    min_class_samples_after_sampling = df[class_name].value_counts().min() * sample_percent

    # Calculate the number of samples for the least populated class after sampling
    min_class_samples_after_sampling_and_split = min_class_samples_after_sampling * test_size

    # If the least populated class will have at least 2 members after sampling, perform the sampling
    if min_class_samples_after_sampling_and_split >= 1:
        return df.groupby(class_name, group_keys=False).apply(
            lambda x: x.sample(frac=sample_percent, random_state=seed)).sort_index()
    else:
        return None


def preprocess_dataset(df: DataFrame) -> DataFrame:
    # replacing infinite values by the maximum allowed value
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if np.any(np.isinf(df[numeric_columns])):
        print(" - Replacing infinite values by the maximum allowed value")
        df[numeric_columns] = df[numeric_columns].replace([np.inf, -np.inf], np.nan)

    # encoding all no numerical columns
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for column in df.columns:
        if not df[column].dtype.kind in ['i', 'f']:
            print(f" - Encoding column {column}")
            df[column] = le.fit_transform(df[column].astype(str))

    # replacing missing values by mean
    if df.isnull().any().any():
        print(" - Replacing missing values by mean")
        df.fillna(df.mean(), inplace=True)

    return df

# set with general dataset characteristics
dataset_description_header = ["class_ent", "eq_num_attr", "gravity", "inst_to_attr",
                              "nr_attr", "nr_bin", "nr_class", "nr_cor_attr", "nr_inst", "nr_norm",
                              "nr_outliers", "ns_ratio"]
# cat_to_num, num_to_cat, and nr_cat  was omitted because since the dataset was codified its value are 0
# nr_num was omitted because it is the same as nr_attr

def describe_dataset_using_pymfe(df: DataFrame, class_name: str, random_state: int) -> dict:
    X = df.drop(class_name, axis=1).values
    y = df[class_name].values
    y = ['label ' + str(val) for val in y]

    mfe = MFE(groups=["general", "info-theory", "statistical"])
    mfe.fit(X, y)
    ft = mfe.extract()

    attr_value = {}

    for attr in dataset_description_header:
        if attr in ft[0]:
            # attr_values.append(ft[1][ft[0].index(attr)])
            attr_value[attr] = ft[1][ft[0].index(attr)]
        else:
            raise Exception(f"Attribute {attr} not found in the extracted features")

    return attr_value
