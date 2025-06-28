import os
import glob
import joblib
import json
from typing import Dict, Any, Optional
import tempfile
from datetime import datetime


def read_gold_table(table, gold_db, spark):
    """
    Helper function to read all partitions of a gold table
    """
    folder_path = os.path.join(gold_db, table)
    files_list = [os.path.join(folder_path, os.path.basename(f)) for f in glob.glob(os.path.join(folder_path, '*'))]
    df = spark.read.option("header", "true").parquet(*files_list)
    return df


def data_check(features, labels):
    """
    To check if data for feature and label is consistent
    """
    print(features.count())
    print(labels.count())
    if features.count() == labels.count():

        X_df = features.toPandas().sort_values(by='customer_id')
        y_df = labels.toPandas().sort_values(by='customer_id')
        return (X_df, y_df)
    return False    


def save_model_to_temp_filesystem(
    results: Dict[str, Any], 
    run_id: Optional[str] = None,
    X_oot: Optional[Any] = None,
    y_oot: Optional[Any] = None
) -> Dict[str, str]:
    """
    Save XGBoost model and OOT datasets to temporary directory with proper permissions.
    
    Args:
        results: Dictionary containing model and other objects
        run_id: Optional run ID for unique naming
        X_oot: Out-of-time features dataset
        y_oot: Out-of-time target dataset
        
    Returns:
        Dictionary with file paths (JSON serializable)
    """
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Use /tmp which typically has write permissions
    model_dir = f"/tmp/airflow_models_{run_id}"
    
    # Create directory with proper permissions
    try:
        os.makedirs(model_dir, mode=0o755, exist_ok=True)
        print(f"Created model directory: {model_dir}")
    except PermissionError as e:
        print(f"Failed to create {model_dir}, falling back to tempfile: {e}")
        # Fallback to system temp directory
        model_dir = tempfile.mkdtemp(prefix=f"airflow_models_{run_id}_")
        print(f"Using temporary directory: {model_dir}")
    
    # Define file paths
    model_path = os.path.join(model_dir, f"xgb_model_{run_id}.joblib")
    scaler_path = os.path.join(model_dir, f"scaler_{run_id}.joblib")
    results_path = os.path.join(model_dir, f"results_{run_id}.json")
    
    # OOT dataset paths
    X_oot_path = None
    y_oot_path = None
    if X_oot is not None:
        X_oot_path = os.path.join(model_dir, f"X_oot_{run_id}.joblib")
    if y_oot is not None:
        y_oot_path = os.path.join(model_dir, f"y_oot_{run_id}.joblib")
    
    try:
        # Save model and scaler using joblib
        joblib.dump(results['best_model'], model_path)
        joblib.dump(results['scaler'], scaler_path)
        print(f"Saved model to: {model_path}")
        print(f"Saved scaler to: {scaler_path}")
        
        # Save OOT datasets if provided
        if X_oot is not None:
            joblib.dump(X_oot, X_oot_path)
            print(f"Saved X_oot to: {X_oot_path}")
            
        if y_oot is not None:
            joblib.dump(y_oot, y_oot_path)
            print(f"Saved y_oot to: {y_oot_path}")
        
        # Save serializable results (include OOT results if available)
        serializable_results = {
            'optimal_threshold': results['optimal_threshold'],
            'baseline_results': results['baseline_results'],
            'test_results': results['test_results'],
            'best_params': results['best_params'],
            'feature_importance': results['feature_importance'].to_dict('records') if 'feature_importance' in results else None
        }
        
        # Add OOT results if available
        if 'oot_results' in results:
            serializable_results['oot_results'] = results['oot_results']
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"Saved results to: {results_path}")
        
        # Verify files were created
        files_to_verify = [model_path, scaler_path, results_path]
        if X_oot_path:
            files_to_verify.append(X_oot_path)
        if y_oot_path:
            files_to_verify.append(y_oot_path)
            
        for path in files_to_verify:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Failed to create file: {path}")
            print(f"Verified file exists: {path} ({os.path.getsize(path)} bytes)")
        
        # Return paths
        return_dict = {
            'model_path': model_path,
            'scaler_path': scaler_path,
            'results_path': results_path,
            'model_dir': model_dir,
            'run_id': run_id,
            'has_oot_data': X_oot is not None or y_oot is not None
        }
        
        if X_oot_path:
            return_dict['X_oot_path'] = X_oot_path
        if y_oot_path:
            return_dict['y_oot_path'] = y_oot_path
            
        return return_dict
        
    except Exception as e:
        print(f"Error saving to filesystem: {e}")
        # Clean up on failure
        try:
            import shutil
            if os.path.exists(model_dir):
                shutil.rmtree(model_dir)
        except:
            pass
        raise

def load_model_from_filesystem(file_paths: Dict[str, str]) -> Dict[str, Any]:

    """
    Load XGBoost model and related objects from filesystem.
    
    Args:
        file_paths: Dictionary with file paths from XCom
        
    Returns:
        Dictionary with loaded objects
    """
    # Load model and scaler
    model = joblib.load(file_paths['model_path'])
    scaler = joblib.load(file_paths['scaler_path'])
    
    # Load results
    with open(file_paths['results_path'], 'r') as f:
        results = json.load(f)
    
    return {
        'best_model': model,
        'scaler': scaler,
        **results
    }

def save_model_to_local_directory(
    results: Dict[str, Any], 
    run_id: Optional[str] = None,
    X_oot: Optional[Any] = None,
    y_oot: Optional[Any] = None,
    model_dir: str = "/opt/airflow/models"
) -> Dict[str, str]:
    """
    Save XGBoost model and OOT datasets to local mounted directory.
    
    Args:
        results: Dictionary containing model and other objects
        run_id: Optional run ID for unique naming
        X_oot: Out-of-time features dataset
        y_oot: Out-of-time target dataset
        model_dir: Local directory path for saving models
        
    Returns:
        Dictionary with file paths (JSON serializable)
    """
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create model directory with run-specific subdirectory
    full_model_dir = os.path.join(model_dir, f"run_{run_id}")
    
    try:
        # Create directory with proper permissions
        os.makedirs(full_model_dir, mode=0o755, exist_ok=True)
        print(f"Created model directory: {full_model_dir}")
        
        # Test write permissions by creating a test file
        test_file = os.path.join(full_model_dir, "test_write.tmp")
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        print(f"✅ Write permissions confirmed for: {full_model_dir}")
        
    except PermissionError as e:
        print(f"❌ Permission denied for {full_model_dir}: {e}")
        # Fallback to temp directory
        full_model_dir = tempfile.mkdtemp(prefix=f"airflow_models_{run_id}_")
        print(f"⚠️ Falling back to temp directory: {full_model_dir}")
    except Exception as e:
        print(f"❌ Failed to create directory {full_model_dir}: {e}")
        # Fallback to temp directory
        full_model_dir = tempfile.mkdtemp(prefix=f"airflow_models_{run_id}_")
        print(f"⚠️ Falling back to temp directory: {full_model_dir}")
    
    # Define file paths
    model_path = os.path.join(full_model_dir, f"xgb_model_{run_id}.joblib")
    scaler_path = os.path.join(full_model_dir, f"scaler_{run_id}.joblib")
    results_path = os.path.join(full_model_dir, f"results_{run_id}.json")
    
    # OOT dataset paths
    X_oot_path = None
    y_oot_path = None
    if X_oot is not None:
        X_oot_path = os.path.join(full_model_dir, f"X_oot_{run_id}.joblib")
    if y_oot is not None:
        y_oot_path = os.path.join(full_model_dir, f"y_oot_{run_id}.joblib")
    
    try:
        # Save model and scaler using joblib
        joblib.dump(results['best_model'], model_path)
        joblib.dump(results['scaler'], scaler_path)
        print(f"✅ Saved model to: {model_path}")
        print(f"✅ Saved scaler to: {scaler_path}")
        
        # Save OOT datasets if provided
        if X_oot is not None:
            joblib.dump(X_oot, X_oot_path)
            print(f"✅ Saved X_oot to: {X_oot_path}")
            
        if y_oot is not None:
            joblib.dump(y_oot, y_oot_path)
            print(f"✅ Saved y_oot to: {y_oot_path}")
        
        # Save serializable results (include OOT results if available)
        serializable_results = {
            'optimal_threshold': results['optimal_threshold'],
            'baseline_results': results['baseline_results'],
            'test_results': results['test_results'],
            'best_params': results['best_params'],
            'feature_importance': results['feature_importance'].to_dict('records') if 'feature_importance' in results else None,
            'model_directory': full_model_dir,
            'storage_method': 'local_directory'
        }
        
        # Add OOT results if available
        if 'oot_results' in results:
            serializable_results['oot_results'] = results['oot_results']
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"✅ Saved results to: {results_path}")
        
        # Verify files were created and get their sizes
        files_to_verify = [model_path, scaler_path, results_path]
        if X_oot_path:
            files_to_verify.append(X_oot_path)
        if y_oot_path:
            files_to_verify.append(y_oot_path)
            
        for path in files_to_verify:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Failed to create file: {path}")
            file_size = os.path.getsize(path)
            print(f"✅ Verified file: {os.path.basename(path)} ({file_size:,} bytes)")
        
        # Return paths
        return_dict = {
            'model_path': model_path,
            'scaler_path': scaler_path,
            'results_path': results_path,
            'model_dir': full_model_dir,
            'run_id': run_id,
            'has_oot_data': X_oot is not None or y_oot is not None,
            'storage_method': 'local_directory',
            'base_model_dir': model_dir
        }
        
        if X_oot_path:
            return_dict['X_oot_path'] = X_oot_path
        if y_oot_path:
            return_dict['y_oot_path'] = y_oot_path
            
        print(f"🎉 Successfully saved all files to local directory: {full_model_dir}")
        return return_dict
        
    except Exception as e:
        print(f"❌ Error saving to local directory: {e}")
        # Clean up on failure
        try:
            import shutil
            if os.path.exists(full_model_dir):
                shutil.rmtree(full_model_dir)
                print(f"🧹 Cleaned up failed directory: {full_model_dir}")
        except:
            pass
        raise