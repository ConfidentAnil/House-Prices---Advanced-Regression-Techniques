import numpy as np
import time
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import make_scorer, mean_squared_error



def parameter_grids(model):
    """
    Returns the possible values of the hyperparameters used by a specific model 
    for hyperparameter tuning.

    Parameters:
    model (sklearn estimator): The machine learning model to be trained..

    Returns:
    dict: Parameter Grid.
    """
    # RandomForest
    rf_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2'],
        'bootstrap': [True, False]
    }

    # XGBoost
    xgb_param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 6, 9],
        'min_child_weight': [1, 5, 10],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2],
        'lambda': [0, 0.1, 1.0],  # L2 regularization term.
        'alpha': [0, 0.1, 1.0]    # L1 regularization term.
    }

    # Gradient Boosting
    gb_param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 6, 9],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'subsample': [0.8, 0.9, 1.0],
        'criterion': ['friedman_mse', 'squared_error']
    }

    try:
        if model.__class__.__name__ == "RandomForestRegressor":
            return rf_param_grid
        elif model.__class__.__name__ == "XGBRegressor":
            return xgb_param_grid
        elif model.__class__.__name__ == "GradientBoostingRegressor":
            return gb_param_grid
    except:
        pass # Not sure what to do for stacked algorithms



def log_rmse(y_true, y_pred):
    """
    Calculate the Root Mean Squared Error (RMSE) between the logarithm of the predicted value and
    the logarithm of the observed sales price.

    Parameters:
    y_true (array-like): The actual observed sales prices.
    y_pred (array-like): The predicted sales prices.

    Returns:
    float: The RMSE between the logarithm of the predicted and observed sales prices.
    """
    # Ensure the inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Compute the log of the values
    log_y_true = np.log(y_true)
    log_y_pred = np.log(y_pred)
    
    # Calculate the RMSE
    rmse = np.sqrt(mean_squared_error(log_y_true, log_y_pred))
    
    return rmse



def train_and_evaluate_model(model, X, y):
    """
    Trains and evaluates a machine learning model.

    Parameters:
    model (sklearn estimator): The machine learning model to be trained.
    X (pd.DataFrame or np.ndarray): The feature matrix.
    y (pd.Series or np.ndarray): The target vector.

    Returns:
    None: Prints the log-RMSE of the model on the test set and the time taken for training and evaluation.
    """
    start = time.time()

    to_print = "Training " + model.__class__.__name__
    gap = (75-len(to_print))//2
    print("*"*gap, to_print, "*"*gap)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluation
    y_pred = model.predict(X_test)
    error = log_rmse(y_pred, y_test)
    
    # Calculating time taken
    time_taken = round(time.time() - start, 6)
    print(f"Model training completed in {time_taken} seconds.\n\n")

    return [error, time_taken]



def hyperparameter_tuning_with_grid_search(model, X, y):
    """
    Perform hyperparameter tuning using GridSearchCV with a custom scoring metric.

    Parameters:
    model (sklearn estimator): The machine learning model to tune.
    param_grid (dict): Dictionary with parameters names (str) as keys and lists of parameter values as values.
    X (array-like): Features of the data.
    y (array-like): Target variable of the data.

    Returns:
    dict: Best parameters found by GridSearchCV.
    """
    start = time.time()
    to_print = "Hyperparameter Tuning for " + model.__class__.__name__
    gap = (75-len(to_print))//2
    print("*"*gap, to_print, "*"*gap)

    # Get the dict for parameter grid
    param_grid = parameter_grids(model)

    # Create a custom scorer
    log_rmse_scorer = make_scorer(log_rmse, greater_is_better=False)
    
    grid_search = GridSearchCV(model, 
                               param_grid=param_grid, 
                               cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
                               scoring=log_rmse_scorer,  # Use the custom log_rmse scorer
                               return_train_score=True,
                               verbose=2, n_jobs=-1)  # Provides more detailed output
    grid_search.fit(X, y)

    time_taken = round(time.time() - start, 6)
    print(f"Hyperparameter tuning completed in {time_taken} seconds.\n")

    return [grid_search.best_params_, time_taken]



def hyperparameter_tuning_with_random_search(model, X, y):
    """
    Perform hyperparameter tuning using RandomSearchCV with a custom scoring metric.

    Parameters:
    model (sklearn estimator): The machine learning model to tune.
    param_grid (dict): Dictionary with parameters names (str) as keys and lists of parameter values as values.
    X (array-like): Features of the data.
    y (array-like): Target variable of the data.

    Returns:
    dict: Best parameters found by RandomSearchCV.
    """
    start = time.time()
    to_print = "Hyperparameter Tuning for " + model.__class__.__name__
    gap = (75-len(to_print))//2
    print("*"*gap, to_print, "*"*gap)
    
    # Get the dict for parameter grid
    param_grid = parameter_grids(model)

    # Create a custom scorer
    log_rmse_scorer = make_scorer(log_rmse, greater_is_better=False)

    random_search = RandomizedSearchCV(model, 
                                       param_distributions= param_grid, 
                                       cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42), 
                                       scoring=log_rmse_scorer,  # Use the custom log_rmse scorer
                                       return_train_score=True,
                                       verbose=2, n_jobs=-1)  # Provides more detailed output
    random_search.fit(X, y)

    time_taken = round(time.time() - start, 6)
    print(f"Hyperparameter tuning completed in {time_taken} seconds.\n")
    
    return [random_search.best_params_, time_taken]