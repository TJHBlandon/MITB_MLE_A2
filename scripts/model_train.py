import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, PredefinedSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, fbeta_score, roc_auc_score, classification_report
from typing import Tuple, Dict, Any, Optional


def create_train_val_test_split(
    X: pd.DataFrame, 
    y: pd.DataFrame, 
    test_size: float = 0.2, 
    val_size: float = 0.2, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create train/validation/test split with stratification.
    
    Args:
        X: Features dataframe
        y: Target dataframe with 'label' column
        test_size: Proportion for test set (default 0.2)
        val_size: Proportion for validation set (default 0.2)
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    print("Preparing data with train/validation/test split...")
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        shuffle=True, 
        stratify=y['label']
    )
    
    #divide remaining data into train and validation
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_adjusted,
        random_state=random_state,
        shuffle=True,
        stratify=y_temp['label']
    )
    
    print(f"Data split sizes:")
    print(f"  Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Validation set: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def prepare_and_scale_features(
    X_train: pd.DataFrame, 
    X_val: pd.DataFrame, 
    X_test: pd.DataFrame, 
    X_oot: Optional[pd.DataFrame] = None,
    id_columns: list = ['customer_id', 'snapshot_date']
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], StandardScaler]:
    """
    Prepare features by removing ID columns and applying scaling.
    
    Args:
        X_train, X_val, X_test: Feature dataframes
        X_oot: Optional out-of-time dataset
        id_columns: List of ID columns to remove
        
    Returns:
        Scaled feature arrays and fitted scaler
    """
    print("Preparing and scaling features...")
    
    scaler = StandardScaler()
    
    # Transform data into numpy array
    X_train_arr = X_train.drop(columns=id_columns, errors='ignore').values
    X_val_arr = X_val.drop(columns=id_columns, errors='ignore').values
    X_test_arr = X_test.drop(columns=id_columns, errors='ignore').values
    
    # Apply scaling
    X_train_arr = scaler.fit_transform(X_train_arr)
    X_val_arr = scaler.transform(X_val_arr)
    X_test_arr = scaler.transform(X_test_arr)
    
    X_oot_arr = None
    if X_oot is not None:
        X_oot_arr = X_oot.drop(columns=id_columns, errors='ignore').values
        X_oot_arr = scaler.transform(X_oot_arr)
    
    print(f"Feature dimensions: {X_train_arr.shape[1]} features")
    
    return X_train_arr, X_val_arr, X_test_arr, X_oot_arr, scaler


def extract_target_arrays(
    y_train: pd.DataFrame, 
    y_val: pd.DataFrame, 
    y_test: pd.DataFrame, 
    y_oot: Optional[pd.DataFrame] = None,
    target_column: str = 'label'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Extract target variables as numpy arrays.
    
    Args:
        y_train, y_val, y_test: Target dataframes
        y_oot: Optional out-of-time target dataframe
        target_column: Name of target column
        
    Returns:
        Target arrays
    """
    y_train_arr = y_train[target_column].values
    y_val_arr = y_val[target_column].values
    y_test_arr = y_test[target_column].values
    
    print(f"Class distribution in training set: {np.bincount(y_train_arr)}")
    print(f"Class distribution in validation set: {np.bincount(y_val_arr)}")
    print(f"Class distribution in test set: {np.bincount(y_test_arr)}")
    
    y_oot_arr = None
    if y_oot is not None:
        y_oot_arr = y_oot[target_column].values
        print(f"Class distribution in OOT set: {np.bincount(y_oot_arr)}")
    
    return y_train_arr, y_val_arr, y_test_arr, y_oot_arr


def train_baseline_xgboost(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    X_val: np.ndarray, 
    y_val: np.ndarray,
    **xgb_params
) -> xgb.XGBClassifier:
    """
    Train a baseline XGBoost model with default parameters.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data for monitoring
        **xgb_params: Additional XGBoost parameters
        
    Returns:
        Trained XGBoost model
    """
    print("Training baseline XGBoost model...")
    
    # Default parameters
    default_params = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'random_state': 42,
        'tree_method': 'hist',
        'enable_categorical': False
    }
    
    default_params.update(xgb_params)
    
    # Initialize and train model
    model = xgb.XGBClassifier(**default_params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )
    
    return model


def evaluate_model(
    model: xgb.XGBClassifier, 
    X: np.ndarray, 
    y: np.ndarray, 
    threshold: float = 0.5,
    dataset_name: str = "Dataset"
) -> Dict[str, float]:
    """
    Evaluate model performance with multiple metrics.
    
    Args:
        model: Trained XGBoost model
        X: Features
        y: True labels
        threshold: Classification threshold
        dataset_name: Name for printing results
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Get predictions
    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_pred_proba > threshold).astype(int)
    
    # Calculate metrics
    f1 = f1_score(y, y_pred)
    f15 = fbeta_score(y, y_pred, beta=1.5)
    auc = roc_auc_score(y, y_pred_proba)
    
    results = {
        'f1_score': f1,
        'f1.5_score': f15,
        'auc_score': auc,
        'threshold': threshold
    }
    
    print(f"\n{dataset_name} Results:")
    print(f"F1 Score: {f1:.4f}")
    print(f"F1.5 Score: {f15:.4f}")
    print(f"AUC Score: {auc:.4f}")
    
    return results


def get_hyperparameter_grid() -> Dict[str, list]:
    """
    Get comprehensive hyperparameter grid for XGBoost tuning.
    
    Returns:
        Dictionary with hyperparameter ranges
    """
    return {
        'n_estimators': [100, 200, 300],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1, 0.15],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9],
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [1, 2, 5],
        'min_child_weight': [1, 3, 5]
    }


def tune_hyperparameters(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    X_val: np.ndarray, 
    y_val: np.ndarray,
    param_grid: Optional[Dict[str, list]] = None,
    scoring: str = 'f1',
    n_jobs: int = -1,
    verbose: int = 1
) -> GridSearchCV:
    """
    Perform hyperparameter tuning using GridSearchCV with predefined train/val split.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        param_grid: Parameter grid for search (if None, uses default)
        scoring: Scoring metric for optimization
        n_jobs: Number of parallel jobs
        verbose: Verbosity level
        
    Returns:
        Fitted GridSearchCV object
    """
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING WITH GRID SEARCH")
    print("="*60)
    
    if param_grid is None:
        param_grid = get_hyperparameter_grid()
    
    # Base model for grid search
    base_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        random_state=42,
        tree_method='hist'
    )
    
    # Create predefined split indices
    test_fold = np.full(len(X_train), -1)  # Training data gets -1
    val_fold = np.full(len(X_val), 0)      # Validation data gets 0
    split_index = np.concatenate([test_fold, val_fold])
    
    # Combine train and validation data
    X_train_val = np.vstack([X_train, X_val])
    y_train_val = np.concatenate([y_train, y_val])
    
    # Create predefined split
    ps = PredefinedSplit(test_fold=split_index)
    
    # Grid Search
    print("Running Grid Search (this may take a while)...")
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=ps,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose,
        return_train_score=True
    )
    
    grid_search.fit(X_train_val, y_train_val)
    
    print("Grid Search completed!")
    print(f"Best {scoring} Score (Validation): {grid_search.best_score_:.4f}")
    print(f"Best Parameters: {grid_search.best_params_}")
    
    return grid_search


def optimize_threshold_for_f1(y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[float, float]:
    """
    Find optimal threshold for F1 score.
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities
        
    Returns:
        Optimal threshold and corresponding F1 score
    """
    thresholds = np.arange(0.1, 0.9, 0.01)
    f1_scores = []
    
    for threshold in thresholds:
        y_pred_thresh = (y_proba > threshold).astype(int)
        f1_thresh = f1_score(y_true, y_pred_thresh)
        f1_scores.append(f1_thresh)
    
    optimal_idx = np.argmax(f1_scores)
    return thresholds[optimal_idx], f1_scores[optimal_idx]


def get_feature_importance(
    model: xgb.XGBClassifier, 
    feature_names: Optional[list] = None,
    top_k: int = 10
) -> pd.DataFrame:
    """
    Extract and display feature importance from trained model.
    
    Args:
        model: Trained XGBoost model
        feature_names: List of feature names (if None, creates generic names)
        top_k: Number of top features to display
        
    Returns:
        DataFrame with feature importance
    """
    feature_importance = model.feature_importances_
    
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(len(feature_importance))]
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop {top_k} Most Important Features:")
    print(importance_df.head(top_k).to_string(index=False))
    
    return importance_df


def print_hyperparameter_guide():
    """
    Print comprehensive hyperparameter guide for XGBoost 3.0.2.
    """
    print("\n" + "="*60)
    print("COMPREHENSIVE HYPERPARAMETER GUIDE FOR XGBoost 3.0.2")
    print("="*60)
    
    hyperparameter_guide = {
        'n_estimators': {
            'range': [50, 100, 200, 500, 1000],
            'description': 'Number of boosting rounds (trees)',
            'tuning_tip': 'Start with 100-200, increase if underfitting'
        },
        'max_depth': {
            'range': [3, 4, 5, 6, 7, 8, 9, 10],
            'description': 'Maximum depth of trees',
            'tuning_tip': '3-6 for small datasets, 6-10 for larger datasets'
        },
        'learning_rate': {
            'range': [0.01, 0.05, 0.1, 0.15, 0.2, 0.3],
            'description': 'Step size shrinkage (eta)',
            'tuning_tip': 'Lower values need more n_estimators'
        },
        'subsample': {
            'range': [0.6, 0.7, 0.8, 0.9, 1.0],
            'description': 'Fraction of samples used for each tree',
            'tuning_tip': '0.8-0.9 often works well, prevents overfitting'
        },
        'colsample_bytree': {
            'range': [0.6, 0.7, 0.8, 0.9, 1.0],
            'description': 'Fraction of features used for each tree',
            'tuning_tip': '0.8-0.9 good for preventing overfitting'
        },
        'reg_alpha': {
            'range': [0, 0.1, 0.5, 1, 2, 5],
            'description': 'L1 regularization term',
            'tuning_tip': 'Use when you want feature selection'
        },
        'reg_lambda': {
            'range': [1, 1.5, 2, 3, 5, 10],
            'description': 'L2 regularization term',
            'tuning_tip': 'Default is 1, increase to reduce overfitting'
        },
        'min_child_weight': {
            'range': [1, 3, 5, 7, 10],
            'description': 'Minimum sum of instance weight in a child',
            'tuning_tip': 'Higher values prevent overfitting'
        }
    }
    
    print("Key hyperparameters and recommended ranges:")
    for param, info in hyperparameter_guide.items():
        print(f"\n{param}:")
        print(f"  Range: {info['range']}")
        print(f"  Description: {info['description']}")
        print(f"  Tip: {info['tuning_tip']}")


def run_complete_pipeline(
    X: pd.DataFrame, 
    y: pd.DataFrame,
    X_oot: Optional[pd.DataFrame] = None,
    y_oot: Optional[pd.DataFrame] = None,
    test_size: float = 0.2,
    val_size: float = 0.2,
    param_grid: Optional[Dict[str, list]] = None,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Run the complete XGBoost training pipeline.
    
    Args:
        X, y: Main dataset features and targets
        X_oot, y_oot: Optional out-of-time dataset
        test_size, val_size: Split proportions
        param_grid: Custom parameter grid for tuning
        random_state: Random seed
        
    Returns:
        Dictionary containing all results and trained models
    """
    results = {}
    
    #Create train/val/test split
    X_train, X_val, X_test, y_train, y_val, y_test = create_train_val_test_split(
        X, y, test_size, val_size, random_state
    )
    
    #Prepare and scale features
    X_train_arr, X_val_arr, X_test_arr, X_oot_arr, scaler = prepare_and_scale_features(
        X_train, X_val, X_test, X_oot
    )
    
    #Extract target arrays
    y_train_arr, y_val_arr, y_test_arr, y_oot_arr = extract_target_arrays(
        y_train, y_val, y_test, y_oot
    )
    
    #Train baseline model
    baseline_model = train_baseline_xgboost(X_train_arr, y_train_arr, X_val_arr, y_val_arr)
    baseline_results = evaluate_model(baseline_model, X_val_arr, y_val_arr, dataset_name="Baseline Model (Validation)")
    
    #Hyperparameter tuning
    grid_search = tune_hyperparameters(X_train_arr, y_train_arr, X_val_arr, y_val_arr, param_grid)
    
    #Retrain best model and evaluate on test set
    best_model = grid_search.best_estimator_
    best_model.fit(X_train_arr, y_train_arr, eval_set=[(X_val_arr, y_val_arr)], verbose=False)
    
    # Step 7: Optimize threshold
    y_pred_proba_val = best_model.predict_proba(X_val_arr)[:, 1]
    optimal_threshold, optimal_f1_val = optimize_threshold_for_f1(y_val_arr, y_pred_proba_val)
    
    print(f"\nOptimal threshold for F1 (found on validation set): {optimal_threshold:.3f}")
    print(f"Optimal F1 score on validation set: {optimal_f1_val:.4f}")
    
    # Final evaluation on test set
    test_results = evaluate_model(best_model, X_test_arr, y_test_arr, optimal_threshold, "Final Test Set")
    
    #Feature importance
    feature_names = [col for col in X.columns if col not in ['customer_id', 'snapshot_date']]
    importance_df = get_feature_importance(best_model, feature_names)
    
    #Detailed classification report
    y_pred_final_test = (best_model.predict_proba(X_test_arr)[:, 1] > optimal_threshold).astype(int)
    print("\nDetailed Classification Report (Test Set):")
    print(classification_report(y_test_arr, y_pred_final_test))
    
    # Compile results
    results = {
        'best_model': best_model,
        'scaler': scaler,
        'optimal_threshold': optimal_threshold,
        'baseline_results': baseline_results,
        'test_results': test_results,
        'best_params': grid_search.best_params_,
        'feature_importance': importance_df,
        'data_splits': {
            'X_train': X_train_arr, 'X_val': X_val_arr, 'X_test': X_test_arr,
            'y_train': y_train_arr, 'y_val': y_val_arr, 'y_test': y_test_arr
        }
    }
    
    # if X_oot_arr is not None:
    #     # oot_results = evaluate_model(best_model, X_oot_arr, y_oot_arr, optimal_threshold, "Out-of-Time")
    #     # results['oot_results'] = oot_results
    #     results['data_splits']['X_oot'] = X_oot_arr
    #     results['data_splits']['y_oot'] = y_oot_arr
    
    print(f"\nPipeline completed!")
    print(f"Best model and results saved in returned dictionary")
    
    return results