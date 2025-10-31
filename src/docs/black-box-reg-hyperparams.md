# Hyperparameter search space explored for the black-box regressors

| Regressor | Hyperparameters (explored values) |
|-----------|-----------------------------------|
| **Linear Regression** | No tunable hyperparameters (default configuration). |
| **Decision Tree (DT)** | **max_depth**: {8, 16, 64, 128} <br> **min_samples_split**: {5, 10} <br> **min_samples_leaf**: {1, 2, 5} <br> **max_features**: {`sqrt`, `log2`} |
| **Random Forest (RF)** | **n_estimators**: {50, 100} <br> **max_depth**: {8, 16, 64, 128} <br> **min_samples_split**: {2, 5} <br> **min_samples_leaf**: {1, 2} <br> **max_features**: {`sqrt`, `log2`, 0.5} <br> **bootstrap**: {True, False} |
| **Gradient Boosting (GB)** | **n_estimators**: {100, 300} <br> **learning_rate**: {0.03, 0.1} <br> **max_depth**: {3, 5} <br> **subsample**: {1.0, 0.8} |
| **MLP Regressor** | **hidden_layer_sizes**: {(50,), (100,), (200,)} <br> **activation**: {relu, tanh} <br> **alpha**: {0.0001, 0.001} <br> **max_iter**: 1000 (fixed) |


## Selected hyperparameters for each model, for both Non-normalized and Normalized meta-datasets:

| Model                    | Non-normalized meta-dataset                                                                 | Normalized meta-dataset                                                                   |
|--------------------------|--------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| LinearRegression         | Default parameters (no tuning)                                                             | Default parameters (no tuning)                                                            |
| DecisionTreeRegressor    | `max_depth`: 8, `max_features`: sqrt, `min_samples_leaf`: 1, `min_samples_split`: 5        | `max_depth`: 8, `max_features`: sqrt, `min_samples_leaf`: 1, `min_samples_split`: 5       |
| RandomForestRegressor    | `bootstrap`: False, `max_depth`: 8, `max_features`: sqrt, `min_samples_leaf`: 1, `min_samples_split`: 2, `n_estimators`: 100 | `bootstrap`: False, `max_depth`: 8, `max_features`: sqrt, `min_samples_leaf`: 2, `min_samples_split`: 5, `n_estimators`: 100 |
| GradientBoostingRegressor| `learning_rate`: 0.03, `max_depth`: 5, `n_estimators`: 100, `subsample`: 1.0              | `learning_rate`: 0.03, `max_depth`: 5, `n_estimators`: 100, `subsample`: 1.0             |
| MLPRegressor             | `activation`: tanh, `alpha`: 0.001, `hidden_layer_sizes`: (200,)                           | `activation`: relu, `alpha`: 0.0001, `hidden_layer_sizes`: (200,)                        |