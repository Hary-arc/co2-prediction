import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb

def train_models(X, y, random_state=42):
    """
    Train multiple regression models on the given data.
    
    Parameters:
    -----------
    X : pandas.DataFrame or numpy.ndarray
        Feature matrix
    y : pandas.Series or numpy.ndarray
        Target vector
    random_state : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    dict
        Dictionary containing trained models
    """
    models = {
        'Decision Tree': DecisionTreeRegressor(random_state=random_state),
        'Random Forest': RandomForestRegressor(random_state=random_state),
        'XGBoost': xgb.XGBRegressor(random_state=random_state)
    }
    
    # Train each model
    for name, model in models.items():
        models[name] = model.fit(X, y)
    
    return models

def tune_hyperparameters(X, y, model_type='Random Forest', random_state=42):
    """
    Tune hyperparameters for a specific model using GridSearchCV.
    
    Parameters:
    -----------
    X : pandas.DataFrame or numpy.ndarray
        Feature matrix
    y : pandas.Series or numpy.ndarray
        Target vector
    model_type : str, optional
        Type of model to tune - 'Decision Tree', 'Random Forest', or 'XGBoost'
    random_state : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    sklearn estimator
        Best model after hyperparameter tuning
    dict
        Best hyperparameters
    """
    if model_type == 'Decision Tree':
        model = DecisionTreeRegressor(random_state=random_state)
        param_grid = {
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    elif model_type == 'Random Forest':
        model = RandomForestRegressor(random_state=random_state)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    elif model_type == 'XGBoost':
        model = xgb.XGBRegressor(random_state=random_state)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.8, 1.0]
        }
    else:
        raise ValueError("Unsupported model type. Use 'Decision Tree', 'Random Forest', or 'XGBoost'.")
    
    # Perform grid search
    grid_search = GridSearchCV(
        model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1
    )
    grid_search.fit(X, y)
    
    return grid_search.best_estimator_, grid_search.best_params_

def evaluate_model(model, X, y, cv=5):
    """
    Evaluate a model using cross-validation.
    
    Parameters:
    -----------
    model : sklearn estimator
        Model to evaluate
    X : pandas.DataFrame or numpy.ndarray
        Feature matrix
    y : pandas.Series or numpy.ndarray
        Target vector
    cv : int, optional
        Number of cross-validation folds
        
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    # Cross-validation scores
    cv_r2 = cross_val_score(model, X, y, cv=cv, scoring='r2')
    cv_mse = -cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
    cv_mae = -cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
    
    # Calculate RMSE from MSE
    cv_rmse = np.sqrt(cv_mse)
    
    return {
        'R2': np.mean(cv_r2),
        'RMSE': np.mean(cv_rmse),
        'MAE': np.mean(cv_mae),
        'R2_std': np.std(cv_r2),
        'RMSE_std': np.std(cv_rmse),
        'MAE_std': np.std(cv_mae)
    }

def predict_emissions(model, X):
    """
    Generate CO2 emissions predictions using the trained model.
    
    Parameters:
    -----------
    model : sklearn estimator
        Trained model
    X : pandas.DataFrame or numpy.ndarray
        Feature matrix
        
    Returns:
    --------
    numpy.ndarray
        Predicted CO2 emissions
    """
    return model.predict(X)

def get_feature_importance(model, feature_names):
    """
    Extract feature importance from the model.
    
    Parameters:
    -----------
    model : sklearn estimator
        Trained model
    feature_names : list
        List of feature names
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with feature importances
    """
    # Check model type and extract feature importance
    if isinstance(model, (RandomForestRegressor, DecisionTreeRegressor)):
        importance = model.feature_importances_
    elif isinstance(model, xgb.XGBRegressor):
        importance = model.feature_importances_
    else:
        raise ValueError("Unsupported model type for feature importance extraction")
    
    # Create a DataFrame with feature names and importance
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    return importance_df

def get_uncertainty_intervals(model, X, method='bootstrap', n_iterations=100, confidence=0.95):
    """
    Generate prediction uncertainty intervals.
    
    Parameters:
    -----------
    model : sklearn estimator
        Trained model
    X : pandas.DataFrame or numpy.ndarray
        Feature matrix
    method : str, optional
        Method for uncertainty estimation - 'bootstrap' or 'quantile' (for XGBoost only)
    n_iterations : int, optional
        Number of bootstrap iterations
    confidence : float, optional
        Confidence level for the intervals
        
    Returns:
    --------
    tuple
        Lower and upper bounds for predictions
    """
    # Get point predictions
    predictions = model.predict(X)
    
    if method == 'bootstrap':
        # Bootstrap predictions
        bootstrap_predictions = np.zeros((X.shape[0], n_iterations))
        
        for i in range(n_iterations):
            # Sample with replacement from training data (assuming model has training data)
            if hasattr(model, 'estimators_'):  # RandomForest
                # For RandomForest, use a subset of trees for each bootstrap iteration
                indices = np.random.choice(len(model.estimators_), size=len(model.estimators_), replace=True)
                bootstrap_pred = np.mean([model.estimators_[idx].predict(X) for idx in indices], axis=0)
            else:
                # For other models, we'd need the training data
                # This is a simplified version
                noise = np.random.normal(0, 0.1 * np.std(predictions), size=predictions.shape)
                bootstrap_pred = predictions + noise
            
            bootstrap_predictions[:, i] = bootstrap_pred
        
        # Calculate confidence intervals
        alpha = (1 - confidence) / 2
        lower_bounds = np.quantile(bootstrap_predictions, alpha, axis=1)
        upper_bounds = np.quantile(bootstrap_predictions, 1 - alpha, axis=1)
        
    elif method == 'quantile' and isinstance(model, xgb.XGBRegressor):
        # For XGBoost, use quantile regression (simplified)
        # In practice, you'd train separate models for lower and upper quantiles
        lower_bounds = predictions * 0.9  # Simplified approximation
        upper_bounds = predictions * 1.1  # Simplified approximation
        
    else:
        # Simple heuristic method
        prediction_std = np.std(predictions) * 0.1  # Assume 10% of std as uncertainty
        z_score = 1.96  # For 95% confidence
        
        lower_bounds = predictions - z_score * prediction_std
        upper_bounds = predictions + z_score * prediction_std
    
    return lower_bounds, upper_bounds
