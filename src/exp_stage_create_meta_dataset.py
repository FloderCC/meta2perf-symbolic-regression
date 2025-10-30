"""
Script to run the experiment.
"""
import ast
import math
import numpy as np
import pandas as pd

dataset_description_header= ["class_ent", "eq_num_attr", "gravity", "inst_to_attr",
                              "nr_attr", "nr_bin", "nr_class", "nr_cor_attr", "nr_inst", "nr_norm",
                              "nr_outliers", "ns_ratio"]

# opening the dataset with the ML results (results_stage_ml_eval)
df_ml_results = pd.read_csv("results/results_stage_ml_eval.csv")

# opening the dataset with the meta-features (results_stage_dataset_description)
df_meta_features = pd.read_csv("./results/results_stage_dataset_description.csv")


df_meta_dataset = pd.merge(
    df_ml_results,
    df_meta_features,
    on=["Seed", "Dataset", "Sample Size"],
    how="inner"
)

# /// Adding Model's related columns \\\

### adding the Model Type column ###

def encode_model_type(value):
    if value == "Linear":
        return 0
    if value == "Probabilistic":
        return 1
    if value == "Tree-Based":
        return 2
    if value == "Ensemble":
        return 3
    if value == "Neural Network":
        return 4
    raise ValueError(f"Model type {value} not recognized.")

def get_model_type_given_model(model):
    """
    Returns the model type based on the model name.
    """
    if model in ["LR", "Ridge", "SGD", "Perceptron", "LinearSVC"]:
        return encode_model_type("Linear")
    if model in ["GaussianNB", "BernoulliNB"]:
        return encode_model_type("Probabilistic")
    if model in ["DT", "ExtraTree"]:
        return encode_model_type("Tree-Based")
    if model in ["RF", "ExtraTrees", "AdaBoost", "HistGradientBoosting", "Bagging"]:
        return encode_model_type("Ensemble")
    if model in ["MLP", "DNN"]:
        return encode_model_type("Neural Network")
    raise ValueError(f"Model {model} not recognized for model type calculation.")

df_meta_dataset["Model Type"] = df_meta_dataset["Model"].apply(get_model_type_given_model)

### adding the training complexity column ###

def encode_training_complexity(value):
    """
    Encodes the complexity value into a numerical representation.
    O(m) -> 0, O(n * m) -> 1, O(n * m * log(m)) -> 2, O(n * m^2) -> 3
    """
    if value == "O(m)":
        return 0
    if value == "O(n * m)":
        return 1
    if value == "O(n * m * log(m))":
        return 2
    if value == "O(n * m^2)":
        return 3
    raise ValueError(f"Complexity {value} not recognized.")

def get_training_complexity_given_model(model):
    if model in ["LR", "SGD", "Perceptron", "LinearSVC", "GaussianNB", "BernoulliNB", "AdaBoost"]:
        return encode_training_complexity("O(n * m)")
    if model == "Ridge":
        return encode_training_complexity("O(n * m^2)")
    if model in ["DT", "ExtraTree"]:
        return encode_training_complexity("O(n * m * log(m))")
    if model in ["RF", "ExtraTrees"]:
        return encode_training_complexity("O(n * m * log(m))")
    if model == "HistGradientBoosting":
        return encode_training_complexity("O(m)")
    if model == "Bagging":
        return encode_training_complexity("O(n * m * log(m))")
    if model in ["MLP", "DNN"]:
        return encode_training_complexity("O(n * m)")
    raise ValueError(f"Model {model} not recognized for training complexity calculation.")

df_meta_dataset["Training Complexity"] = df_meta_dataset["Model"].apply(get_training_complexity_given_model)

### adding the prediction complexity column ###

def encode_prediction_complexity(value):
    """
    Encodes the complexity value into a numerical representation.
    O(log(m)) -> 0, O(m) -> 1, O(n * m) -> 2
    """
    if value == "O(log(m))":
        return 0
    if value == "O(m)":
        return 1
    if value == "O(n * m)":
        return 2
    raise ValueError(f"Complexity {value} not recognized.")

def get_prediction_complexity_given_model(model):
    if model in ["LR", "Ridge", "SGD", "Perceptron", "LinearSVC", "GaussianNB", "BernoulliNB", "AdaBoost"]:
        return encode_prediction_complexity("O(m)")
    if model in ["DT", "ExtraTree", "RF", "ExtraTrees", "HistGradientBoosting", "Bagging"]:
        return encode_prediction_complexity("O(log(m))")
    if model in ["MLP", "DNN"]:
        return encode_prediction_complexity("O(n * m)")
    raise ValueError(f"Model {model} not recognized for prediction complexity calculation.")

df_meta_dataset["Prediction Complexity"] = df_meta_dataset["Model"].apply(get_prediction_complexity_given_model)

### adding the column regularization the number of parameters ###

def encode_regularization(value):
    """
    Encodes the regularization type into a numerical representation.
    None -> 0, Laplace -> 1, l1 -> 2, l2 -> 3, ElasticNet -> 4,
    """
    if value is None:
        return 0
    if value == "laplace":
        return 1
    if value == "l1":
        return 2
    if value == "l2":
        return 3
    if value == "elasticnet":
        return 4
    raise ValueError(f"Regularization {value} not recognized.")

def get_regularization_number_of_parameters_given_model(model, hyperparameters):
    # parsing hyperparameters from string to a dictionary)
    hyperparameters_as_dict = ast.literal_eval("{" + hyperparameters + "}")

    if model in ["LR", "LinearSVC", "Perceptron"]:
        return encode_regularization(hyperparameters_as_dict["penalty"])
    if model in ["Ridge", "BernoulliNB"]:
        return encode_regularization("l2" if hyperparameters_as_dict["alpha"] != 0.1 else None)
    if model in ["SGD"]:
        return encode_regularization("l2") # the default value is l2
    if model in ["DT", "ExtraTree", "GaussianNB", "Bagging", "RF", "ExtraTrees", "AdaBoost"]: # they do not have explicit regularization parameters
        return encode_regularization(None)
    if model in ["HistGradientBoosting"]:
        return encode_regularization("l2" if hyperparameters_as_dict["l2_regularization"] != 0.0 else None)
    if model in ["MLP", "DNN"]:
        return encode_regularization("l2" if hyperparameters_as_dict["alpha"] != 0.0001 else None) # the possible values are 0.0001, 0.001, 0.01
    raise ValueError(f"Model {model} not recognized for regularization calculation.")

df_meta_dataset["Regularization"] = df_meta_dataset.apply(
    lambda row: get_regularization_number_of_parameters_given_model(
        row["Model"],
        row["Model Parameters"]
    ),
    axis=1
)

### adding the Robust to Outliers column ###

def encode_robust_to_outliers(value):
    """
    Encodes the robustness to outliers into a numerical representation.
    No -> 0, Medium -> 1, Yes -> 2
    """
    if value == "No":
        return 0
    if value == "Medium":
        return 1
    if value == "Yes":
        return 2
    raise ValueError(f"Robustness to outliers {value} not recognized.")

def get_robust_to_outliers_given_model(model):
    if model in ["LR", "Ridge", "LinearSVC", "GaussianNB", "BernoulliNB", "AdaBoost", "MLP", "DNN"]:
        return encode_robust_to_outliers("No")
    if model in ["SGD", "Perceptron"]:
        return encode_robust_to_outliers("Medium")
    if model in ["DT", "ExtraTree", "RF", "ExtraTrees", "HistGradientBoosting", "Bagging"]:
        return encode_robust_to_outliers("Yes")
    raise ValueError(f"Model {model} not recognized for robustness to outliers calculation.")

df_meta_dataset["Robust to Outliers"] = df_meta_dataset["Model"].apply(get_robust_to_outliers_given_model)

### adding the Representational Capacity column ###

def encode_representational_capacity(value):
    """
    Encodes the representational capacity into a numerical representation.
    Low -> 0, Medium -> 1, High -> 2, Very High -> 3
    """
    if value == "Low":
        return 0
    if value == "Medium":
        return 1
    if value == "High":
        return 2
    if value == "Very High":
        return 3
    raise ValueError(f"Representational capacity {value} not recognized.")
def get_representational_capacity_given_model(model):
    if model in ["LR", "Ridge", "GaussianNB", "BernoulliNB"]:
        return encode_representational_capacity("Low")
    if model in ["SGD", "Perceptron", "LinearSVC", "DT", "ExtraTree", "AdaBoost", "Bagging"]:
        return encode_representational_capacity("Medium")
    if model in ["RF", "ExtraTrees", "HistGradientBoosting"]:
        return encode_representational_capacity("High")
    if model in ["MLP", "DNN"]:
        return encode_representational_capacity("Very High")
    raise ValueError(f"Model {model} not recognized for representational capacity calculation.")
df_meta_dataset["Representational Capacity"] = df_meta_dataset["Model"].apply(get_representational_capacity_given_model)

### adding the Processing Units Number column ###

def get_processing_units_number_given_model_and_hyperparameters(model, hyperparameters, no_class, no_features):
    """
    Returns the number of processing units based on the model name and its hyperparameters.
    """
    # parsing hyperparameters from string to a dictionary)

    hyperparameters_as_dict = ast.literal_eval("{" + hyperparameters + "}")

    if model in ["LR", "Ridge", "LinearSVC", "SGD", "Perceptron", "BernoulliNB"]: # no_class * (#features + 1)
        return no_class * no_features + 1

    if model in ["DT", "ExtraTree"]: # 2^max_depth - 1
        tree_p_n_u = 2 ** int(hyperparameters_as_dict["max_depth"]) - 1
        return tree_p_n_u

    if model in ["RF", "ExtraTrees"]: # n_estimators × 2^max_depth - 1
        tree_p_n_u = int(hyperparameters_as_dict["n_estimators"]) * (2 ** int(hyperparameters_as_dict["max_depth"]) - 1)
        return tree_p_n_u

    if model == "HistGradientBoosting": # max_iter× (2^max_depth - 1):
        tree_p_n_u = int(hyperparameters_as_dict["max_iter"]) * (2 ** int(hyperparameters_as_dict["max_depth"]) - 1)
        return tree_p_n_u

    if model == "AdaBoost": # n_estimators * 3
        tree_p_n_u = int(hyperparameters_as_dict["n_estimators"]) * (2 ** 1 - 1)
        return tree_p_n_u

    if model == "Bagging": # n_estimators * 2^160
        tree_p_n_u = int(hyperparameters_as_dict["n_estimators"]) * 2 ** 160
        return tree_p_n_u

    if model == "GaussianNB": # class * #features * 2
        return no_class * no_features * 2

    if model in ["MLP", "DNN"]: # sum(hidden_layer_sizes)
        return sum(hyperparameters_as_dict["hidden_layer_sizes"])

    raise ValueError(f"Model {model} not recognized for processing units calculation.")

df_meta_dataset_2 = df_meta_dataset.copy()

df_meta_dataset["Processing Units Number"] = df_meta_dataset.apply(
    lambda row: math.log(get_processing_units_number_given_model_and_hyperparameters(
        row["Model"],
        row["Model Parameters"],
        row["nr_class"],
        row["nr_attr"]
    )),
    axis=1
)

### formating the meta-dataset ###
setup_columns = ["Seed", "Dataset", "Sample Size", "Model"]
dataset_description_columns = ["class_ent", "eq_num_attr", "gravity", "inst_to_attr",
                              "nr_attr", "nr_bin", "nr_class", "nr_cor_attr", "nr_inst", "nr_norm",
                              "nr_outliers", "ns_ratio"]
model_description_columns = ["Processing Units Number", "Model Type", "Training Complexity", "Prediction Complexity",
                 "Regularization", "Robust to Outliers", "Representational Capacity"] # maybe also Number of Iterations

target_columns = ["MCC"]

columns_to_save = setup_columns + dataset_description_columns + model_description_columns + target_columns

df_meta_dataset = df_meta_dataset[columns_to_save].copy()

df_meta_dataset.to_csv("./results/meta_dataset.csv", index=False)






