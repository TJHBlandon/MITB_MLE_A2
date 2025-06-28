import os
import gc
import sys
import numpy as np
import pandas as pd
import joblib
import json
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from airflow import DAG
from airflow.decorators import task, dag
from airflow.operators.python import PythonOperator

from pyspark.sql import SparkSession
from tqdm import tqdm

# Add project root to PYTHONPATH so imports work
import sys
sys.path.extend(["/opt/airflow/scripts", "/opt/airflow/utils"])

# Data processing imports
from utils.process_bronze_table import process_bronze_table
from utils.process_silver_table import process_silver_table
from utils.process_gold_table import process_gold_table

# Model training imports
import scripts.model_train as model_train
import scripts.utilities as utils


def create_isolated_spark_session():
    """Memory-optimized Spark session"""
    return (SparkSession.builder
        .master("local[2]")
        .appName(f"AirflowTask-{os.getpid()}")
        
        # MEMORY OPTIMIZATION - Reduced settings
        .config("spark.driver.memory", "1g")
        .config("spark.driver.maxResultSize", "512m")  
        .config("spark.executor.memory", "1g")
        .config("spark.executor.cores", "1")
        
        # CRITICAL MEMORY FIXES
        .config("spark.storage.memoryFraction", "0.2")
        .config("spark.storage.level", "MEMORY_AND_DISK_SER")
        .config("spark.sql.shuffle.partitions", "10")
        .config("spark.default.parallelism", "2")
        
        # EXISTING CONFIGS
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.sql.adaptive.enabled", "false")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "false")
        .config("spark.dynamicAllocation.enabled", "false")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.catalogImplementation", "in-memory")
        .config("spark.driver.host", "localhost")
        .config("spark.driver.bindAddress", "0.0.0.0")
        .getOrCreate())


def force_cleanup_spark():
    """Forcefully cleanup all Spark resources"""
    try:
        spark = SparkSession.getActiveSession()
        if spark is not None:
            try:
                spark.sparkContext.stop()
            except:
                pass
            try:
                spark.stop()
            except:
                pass
        
        gc.collect()
        SparkSession._instantiatedSession = None
        SparkSession._activeSession = None
        
    except Exception as e:
        print(f"Warning during Spark cleanup: {e}")


def generate_first_of_month_dates(start_date_str, end_date_str):
    """Generate list of dates to process"""
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    first_of_month_dates = []
    current_date = datetime(start_date.year, start_date.month, 1)

    while current_date <= end_date:
        first_of_month_dates.append(current_date.strftime("%Y-%m-%d"))
        
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)

    return first_of_month_dates


def ensure_directory_exists(directory_path):
    """Ensure directory exists and has proper permissions"""
    try:
        os.makedirs(directory_path, mode=0o755, exist_ok=True)
        print(f"✅ Directory ready: {directory_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to create directory {directory_path}: {e}")
        return False


default_args = {
    "owner": "data-team",
    "retries": 1,
    "retry_delay": timedelta(minutes=3),
}


@dag(
    dag_id="integrated_end_to_end_pipeline",
    description="Complete end-to-end pipeline: Data Processing + Model Training + Predictions + Monitoring",
    start_date=datetime(2023, 1, 1),
    schedule_interval="0 0 1 1/3 *",  # Run quarterly
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["end-to-end", "data-processing", "model-training", "predictions", "monitoring"]
)
def integrated_end_to_end_pipeline():
    
    @task
    def initialize_pipeline():
        """Initialize the complete end-to-end pipeline"""
        print("🚀 Initializing End-to-End Pipeline...")
        
        # Configuration for both data processing and model training
        config = {
            # Data processing configuration
            "data_processing": {
                "start_date": "2023-01-01",
                "end_date": "2024-12-01",
                "data_directory": "/opt/airflow/data",
                "bronze_directory": "/opt/airflow/datamart/bronze",
                "silver_directory": "/opt/airflow/datamart/silver", 
                "gold_directory": "/opt/airflow/datamart/gold",
                "source_files": {
                    "clickstream": "/opt/airflow/data/feature_clickstream.csv",
                    "attributes": "/opt/airflow/data/features_attributes.csv",
                    "financials": "/opt/airflow/data/features_financials.csv",
                    "lms": "/opt/airflow/data/lms_loan_daily.csv"
                }
            },
            # Model training configuration
            "model_training": {
                "train_test_period_months": 12,
                "oot_data_range": 3,
                "model_train_date": datetime(2025, 1, 1, 0, 0, 0),
                "model_directory": "/opt/airflow/models",
                "prediction_directory": "/opt/airflow/prediction"
            }
        }
        
        # Calculate model training dates
        model_config = config["model_training"]
        model_config["oot_end_date"] = (model_config['model_train_date'] - timedelta(days=1)).date()
        model_config["oot_start_date"] = (model_config['model_train_date'] - relativedelta(months=3)).date()
        model_config["model_data_end_date"] = (model_config["oot_start_date"] - timedelta(days=1))
        model_config["model_data_start_date"] = (model_config["oot_start_date"] - relativedelta(months=12))
        
        # Generate processing dates for data processing
        data_config = config["data_processing"]
        processing_dates = generate_first_of_month_dates(data_config["start_date"], data_config["end_date"])
        data_config["processing_dates"] = processing_dates
        
        print(f"📅 Data processing: {len(processing_dates)} months from {data_config['start_date']} to {data_config['end_date']}")
        print(f"🎯 Model training period: {model_config['model_data_start_date']} to {model_config['model_data_end_date']}")
        print(f"🔮 OOT period: {model_config['oot_start_date']} to {model_config['oot_end_date']}")
        
        # Setup all required directories
        all_directories = [
            data_config["bronze_directory"],
            data_config["silver_directory"],
            data_config["gold_directory"],
            model_config["model_directory"],
            model_config["prediction_directory"]
        ]
        
        all_directories_ready = True
        for directory in all_directories:
            if not ensure_directory_exists(directory):
                all_directories_ready = False
        
        if not all_directories_ready:
            raise RuntimeError("Failed to setup required directories")
        
        # Verify source data files exist
        missing_files = []
        for table_name, file_path in data_config["source_files"].items():
            if not os.path.exists(file_path):
                missing_files.append(f"{table_name}: {file_path}")
        
        if missing_files:
            print(f"⚠️ Warning: Missing source files:")
            for missing in missing_files:
                print(f"   - {missing}")
        else:
            print("✅ All source data files found")
        
        return {
            "status": "initialized",
            "config": config,
            "data_processing_dates_count": len(processing_dates),
            "missing_files": missing_files,
            "pipeline_type": "end_to_end"
        }
    
    @task
    def process_bronze_tables(**context):
        """Process all bronze tables"""
        print("📋 Starting bronze table processing...")
        
        spark = None
        try:
            # Get configuration
            ti = context['task_instance']
            init_result = ti.xcom_pull(task_ids='initialize_pipeline')
            data_config = init_result["config"]["data_processing"]
            
            # Create Spark session
            spark = create_isolated_spark_session()
            spark.sparkContext.setLogLevel("ERROR")
            print("Spark session created for bronze processing")
            
            bronze_directory = data_config["bronze_directory"]
            processing_dates = data_config["processing_dates"]
            source_files = data_config["source_files"]
            
            # Process each table type
            tables_to_process = ["clickstream", "attributes", "financials", "lms"]
            
            bronze_results = {
                "processed_tables": [],
                "total_partitions": 0,
                "errors": []
            }
            
            for table_name in tables_to_process:
                source_file = source_files.get(table_name)
                
                if not source_file or not os.path.exists(source_file):
                    error_msg = f"Source file not found for {table_name}: {source_file}"
                    print(f"❌ {error_msg}")
                    bronze_results["errors"].append(error_msg)
                    continue
                
                print(f"\n📋 Processing bronze table: {table_name}")
                
                table_partitions = 0
                table_errors = []
                
                # Process each date partition
                for date_str in tqdm(processing_dates, desc=f"Processing {table_name}"):
                    try:
                        process_bronze_table(table_name, source_file, bronze_directory, date_str, spark)
                        table_partitions += 1
                    except Exception as e:
                        error_msg = f"Failed to process {table_name} for {date_str}: {str(e)}"
                        table_errors.append(error_msg)
                
                bronze_results["processed_tables"].append({
                    "table_name": table_name,
                    "partitions_processed": table_partitions,
                    "partitions_failed": len(table_errors),
                    "errors": table_errors
                })
                
                bronze_results["total_partitions"] += table_partitions
                bronze_results["errors"].extend(table_errors)
                
                print(f"✅ Completed {table_name}: {table_partitions} partitions processed")
            
            print(f"\n🎉 Bronze processing completed! Total partitions: {bronze_results['total_partitions']}")
            return bronze_results
            
        except Exception as e:
            print(f"❌ Bronze processing failed: {str(e)}")
            raise
        finally:
            if spark:
                force_cleanup_spark()
    
    @task
    def process_silver_tables(**context):
        """Process all silver tables"""
        print("🥈 Starting silver table processing...")
        
        spark = None
        try:
            # Get configuration
            ti = context['task_instance']
            init_result = ti.xcom_pull(task_ids='initialize_pipeline')
            bronze_result = ti.xcom_pull(task_ids='process_bronze_tables')
            
            data_config = init_result["config"]["data_processing"]
            
            # Create Spark session
            spark = create_isolated_spark_session()
            spark.sparkContext.setLogLevel("ERROR")
            
            bronze_directory = data_config["bronze_directory"]
            silver_directory = data_config["silver_directory"]
            processing_dates = data_config["processing_dates"]
            
            # Get successfully processed bronze tables
            bronze_tables = [table["table_name"] for table in bronze_result["processed_tables"] 
                           if table["partitions_processed"] > 0]
            
            silver_results = {
                "processed_tables": [],
                "total_partitions": 0,
                "errors": []
            }
            
            for table_name in bronze_tables:
                print(f"\n🥈 Processing silver table: {table_name}")
                
                table_partitions = 0
                table_errors = []
                
                for date_str in tqdm(processing_dates, desc=f"Processing {table_name}"):
                    try:
                        process_silver_table(table_name, bronze_directory, silver_directory, date_str, spark)
                        table_partitions += 1
                    except Exception as e:
                        error_msg = f"Failed to process {table_name} for {date_str}: {str(e)}"
                        table_errors.append(error_msg)
                
                silver_results["processed_tables"].append({
                    "table_name": table_name,
                    "partitions_processed": table_partitions,
                    "partitions_failed": len(table_errors),
                    "errors": table_errors
                })
                
                silver_results["total_partitions"] += table_partitions
                silver_results["errors"].extend(table_errors)
                
                print(f"✅ Completed {table_name}: {table_partitions} partitions processed")
            
            print(f"\n🎉 Silver processing completed! Total partitions: {silver_results['total_partitions']}")
            return silver_results
            
        except Exception as e:
            print(f"❌ Silver processing failed: {str(e)}")
            raise
        finally:
            if spark:
                force_cleanup_spark()
    
    @task
    def process_gold_tables(**context):
        """Process gold tables (feature and label stores)"""
        print("🥇 Starting gold table processing...")
        
        spark = None
        try:
            # Get configuration
            ti = context['task_instance']
            init_result = ti.xcom_pull(task_ids='initialize_pipeline')
            
            data_config = init_result["config"]["data_processing"]
            
            # Create Spark session
            spark = create_isolated_spark_session()
            spark.sparkContext.setLogLevel("ERROR")
            
            silver_directory = data_config["silver_directory"]
            gold_directory = data_config["gold_directory"]
            processing_dates = data_config["processing_dates"]
            
            print(f"Processing gold tables from: {silver_directory}")
            
            # Process gold tables
            X, y = process_gold_table(silver_directory, gold_directory, processing_dates, spark)
            
            # Get dataset information
            X_count = X.count()
            y_count = y.count()
            X_columns = len(X.columns)
            y_columns = len(y.columns)
            
            print(f"\n🎯 Gold processing completed!")
            print(f"Feature dataset (X): {X_count:,} rows, {X_columns} columns")
            print(f"Label dataset (y): {y_count:,} rows, {y_columns} columns")
            
            gold_results = {
                "status": "completed",
                "feature_dataset": {
                    "row_count": X_count,
                    "column_count": X_columns,
                    "path": f"{gold_directory}/feature_store"
                },
                "label_dataset": {
                    "row_count": y_count,
                    "column_count": y_columns,
                    "path": f"{gold_directory}/label_store"
                },
                "gold_directory": gold_directory
            }
            
            print("✅ Gold tables ready for model training!")
            return gold_results
            
        except Exception as e:
            print(f"❌ Gold processing failed: {str(e)}")
            raise
        finally:
            if spark:
                force_cleanup_spark()
    
    @task
    def model_training(**context):
        """Model training using processed gold tables"""
        print("🤖 Starting model training...")
        
        spark = None
        try:
            # Get configuration
            ti = context['task_instance']
            init_result = ti.xcom_pull(task_ids='initialize_pipeline')
            gold_result = ti.xcom_pull(task_ids='process_gold_tables')
            
            model_config = init_result["config"]["model_training"]
            gold_directory = gold_result["gold_directory"]
            
            # Create Spark session for reading gold tables
            spark = create_isolated_spark_session()
            spark.sparkContext.setLogLevel("ERROR")
            
            # Load data from gold tables
            X_spark = utils.read_gold_table('feature_store', gold_directory, spark)
            y_spark = utils.read_gold_table('label_store', gold_directory, spark)
            
            if X_spark and y_spark:
                model_data = utils.data_check(X_spark, y_spark)
            
            features, labels = model_data
            
            # Filter data based on model training dates
            date_mask = (labels['snapshot_date'] >= model_config['model_data_start_date']) & \
                        (labels['snapshot_date'] <= model_config['model_train_date'].date())
            
            y_model_df = labels[date_mask]
            X_model_df = features[np.isin(features['customer_id'], y_model_df['customer_id'].unique())]
            
            # Split training and OOT data
            y_train = y_model_df[y_model_df['snapshot_date'] <= model_config['model_data_end_date']]
            X_train = X_model_df[np.isin(X_model_df['customer_id'], y_train['customer_id'].unique())]
            
            y_oot = y_model_df[(y_model_df['snapshot_date'] >= model_config['oot_start_date']) & 
                              (y_model_df['snapshot_date'] <= model_config['oot_end_date'])]
            X_oot = X_model_df[np.isin(X_model_df['customer_id'], y_oot['customer_id'].unique())]
            
            print(f"📊 Training data: {len(X_train)} samples")
            print(f"🔮 OOT data: {len(X_oot)} samples")
            
            # Train model
            results = model_train.run_complete_pipeline(X_train, y_train, X_oot, y_oot)
            
            # Save model to local directory
            model_files = utils.save_model_to_local_directory(
                results, 
                run_id=context['run_id'],
                X_oot=X_oot, 
                y_oot=y_oot,
                model_dir=model_config["model_directory"]
            )
            
            print("✅ Model training completed and saved!")
            return model_files
                
        except Exception as e:
            print(f"❌ Model training failed: {str(e)}")
            raise
        finally:
            if spark:
                force_cleanup_spark()
    
    @task
    def model_predict(**context):
        """Make predictions using trained model"""
        print("🔮 Starting model prediction...")
        
        try:
            # Get configuration and model files
            ti = context['task_instance']
            init_result = ti.xcom_pull(task_ids='initialize_pipeline')
            file_paths = ti.xcom_pull(task_ids='model_training')
            
            model_config = init_result["config"]["model_training"]
            
            if not file_paths:
                raise ValueError("No model files received from training task")
            
            # Load model components
            model = joblib.load(file_paths['model_path'])
            scaler = joblib.load(file_paths['scaler_path'])
            
            with open(file_paths['results_path'], 'r') as f:
                results = json.load(f)
            optimal_threshold = results['optimal_threshold']
            
            print(f"✅ Loaded model, threshold: {optimal_threshold}")
            
            # Load OOT datasets
            X_oot = joblib.load(file_paths['X_oot_path'])
            y_oot = joblib.load(file_paths['y_oot_path']) if 'y_oot_path' in file_paths else None
            
            # Prepare features for prediction
            id_columns = ['customer_id', 'snapshot_date']
            X_oot_features = X_oot.copy()
            
            prediction_metadata = {}
            for col in id_columns:
                if col in X_oot.columns:
                    prediction_metadata[col] = X_oot[col].tolist()
                    X_oot_features = X_oot_features.drop(columns=[col])
            
            # Make predictions
            X_oot_scaled = scaler.transform(X_oot_features)
            predictions_proba = model.predict_proba(X_oot_scaled)[:, 1]
            predictions = (predictions_proba > optimal_threshold).astype(int)
            
            print(f"✅ Generated {len(predictions)} predictions")
            print(f"Positive rate: {predictions.mean()*100:.1f}%")
            
            # Create predictions dataset
            predictions_df_data = {
                'prediction_binary': predictions,
                'prediction_proba': predictions_proba,
                'threshold_used': [optimal_threshold] * len(predictions),
                'model_run_id': [file_paths['run_id']] * len(predictions),
                'prediction_timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')] * len(predictions)
            }
            
            # Add metadata
            for col, values in prediction_metadata.items():
                predictions_df_data[col] = values
            
            if y_oot is not None:
                if hasattr(y_oot, 'columns'):
                    predictions_df_data['actual_label'] = y_oot['label'].tolist()
                else:
                    predictions_df_data['actual_label'] = y_oot.tolist()
            
            predictions_df = pd.DataFrame(predictions_df_data)
            
            # Save predictions
            predictions_dir = model_config["prediction_directory"]
            run_id = file_paths['run_id']
            predictions_path = os.path.join(predictions_dir, f"predictions_{run_id}.joblib")
            predictions_csv_path = os.path.join(predictions_dir, f"predictions_{run_id}.csv")
            
            os.makedirs(predictions_dir, exist_ok=True)
            joblib.dump(predictions_df, predictions_path)
            predictions_df.to_csv(predictions_csv_path, index=False)
            
            print(f"✅ Saved predictions to: {predictions_path}")
            
            # Evaluate predictions
            evaluation_results = {}
            if y_oot is not None:
                from sklearn.metrics import f1_score, fbeta_score, roc_auc_score
                
                y_oot_labels = y_oot['label'].values if hasattr(y_oot, 'columns') else y_oot
                
                oot_f1 = f1_score(y_oot_labels, predictions)
                oot_f15 = fbeta_score(y_oot_labels, predictions, beta=1.5)
                oot_auc = roc_auc_score(y_oot_labels, predictions_proba)
                
                print(f"🎯 OOT Performance: F1={oot_f1:.4f}, AUC={oot_auc:.4f}")
                
                evaluation_results = {
                    'oot_f1': oot_f1,
                    'oot_f15': oot_f15,
                    'oot_auc': oot_auc,
                    'has_evaluation': True
                }
            
            return {
                'predictions': predictions.tolist(),
                'predictions_proba': predictions_proba.tolist(),
                'threshold_used': optimal_threshold,
                'num_predictions': len(predictions),
                'positive_rate': float(predictions.mean()),
                'predictions_path': predictions_path,
                'predictions_csv_path': predictions_csv_path,
                'run_id': file_paths['run_id'],
                'prediction_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'model_path': file_paths['model_path'],
                'scaler_path': file_paths['scaler_path'],
                **evaluation_results
            }
            
        except Exception as e:
            print(f"❌ Prediction task failed: {str(e)}")
            raise
    
    @task
    def model_monitoring(**context):
        """Monitor model performance and data quality"""
        print("📊 Starting model monitoring...")
        
        try:
            # Get prediction results
            ti = context['task_instance']
            init_result = ti.xcom_pull(task_ids='initialize_pipeline')
            prediction_results = ti.xcom_pull(task_ids='model_predict')
            
            model_config = init_result["config"]["model_training"]
            
            # Load predictions dataset
            predictions_df = joblib.load(prediction_results['predictions_path'])
            print(f"✅ Analyzing {predictions_df.shape[0]} predictions")
            
            # Basic analysis
            total_predictions = len(predictions_df)
            positive_predictions = predictions_df['prediction_binary'].sum()
            positive_rate = predictions_df['prediction_binary'].mean()
            avg_proba = predictions_df['prediction_proba'].mean()
            
            print(f"📈 Prediction Summary:")
            print(f"   Total: {total_predictions:,}")
            print(f"   Positive: {positive_predictions:,} ({positive_rate*100:.2f}%)")
            print(f"   Avg Probability: {avg_proba:.4f}")
            
            # Probability distribution
            prob_percentiles = np.percentile(predictions_df['prediction_proba'], [10, 25, 50, 75, 90])
            
            monitoring_results = {
                'run_id': prediction_results['run_id'],
                'monitoring_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_predictions': total_predictions,
                'positive_predictions': int(positive_predictions),
                'positive_rate': float(positive_rate),
                'avg_prediction_proba': float(avg_proba),
                'threshold_used': float(predictions_df['threshold_used'].iloc[0]),
                'prob_percentiles': {
                    'p10': float(prob_percentiles[0]),
                    'p25': float(prob_percentiles[1]),
                    'p50': float(prob_percentiles[2]),
                    'p75': float(prob_percentiles[3]),
                    'p90': float(prob_percentiles[4])
                }
            }
            
            # Performance evaluation if labels available
            if 'actual_label' in predictions_df.columns:
                from sklearn.metrics import (
                    f1_score, fbeta_score, roc_auc_score, precision_score, 
                    recall_score, confusion_matrix
                )
                
                actual_labels = predictions_df['actual_label'].values
                predicted_labels = predictions_df['prediction_binary'].values
                predicted_probas = predictions_df['prediction_proba'].values
                
                precision = precision_score(actual_labels, predicted_labels)
                recall = recall_score(actual_labels, predicted_labels)
                f1 = f1_score(actual_labels, predicted_labels)
                f15 = fbeta_score(actual_labels, predicted_labels, beta=1.5)
                auc = roc_auc_score(actual_labels, predicted_probas)
                cm = confusion_matrix(actual_labels, predicted_labels)
                
                print(f"🎯 Performance Metrics:")
                print(f"   Precision: {precision:.4f}, Recall: {recall:.4f}")
                print(f"   F1: {f1:.4f}, AUC: {auc:.4f}")
                
                monitoring_results.update({
                    'has_performance_metrics': True,
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1_score': float(f1),
                    'f15_score': float(f15),
                    'auc_score': float(auc),
                    'confusion_matrix': {
                        'tn': int(cm[0,0]), 'fp': int(cm[0,1]),
                        'fn': int(cm[1,0]), 'tp': int(cm[1,1])
                    }
                })
                
                # Drift detection
                drift_alerts = []
                if prediction_results.get('has_evaluation'):
                    training_auc = prediction_results.get('oot_auc', 0)
                    auc_drift = abs(auc - training_auc)
                    
                    if auc_drift > 0.05:
                        drift_alerts.append(f"HIGH AUC DRIFT: {auc_drift:.4f}")
                    
                    print(f"🚨 Drift Analysis: AUC drift = {auc_drift:.4f}")
                
                monitoring_results['drift_alerts'] = drift_alerts
            
            # Data quality checks
            missing_checks = {}
            for col in predictions_df.columns:
                missing_count = predictions_df[col].isnull().sum()
                if missing_count > 0:
                    missing_checks[col] = int(missing_count)
            
            extreme_probs = {
                'very_low': int((predictions_df['prediction_proba'] < 0.01).sum()),
                'very_high': int((predictions_df['prediction_proba'] > 0.99).sum())
            }
            
            monitoring_results.update({
                'data_quality': {
                    'missing_values': missing_checks,
                    'extreme_probabilities': extreme_probs
                }
            })
            
            # Save monitoring results
            monitoring_path = os.path.join(
                model_config["prediction_directory"], 
                f"monitoring_results_{prediction_results['run_id']}.json"
            )
            
            with open(monitoring_path, 'w') as f:
                json.dump(monitoring_results, f, indent=2)
            
            print(f"✅ Monitoring results saved to: {monitoring_path}")
            
            return {
                'monitoring_status': 'completed',
                'monitoring_path': monitoring_path,
                'total_predictions': monitoring_results['total_predictions'],
                'positive_rate': monitoring_results['positive_rate'],
                'data_quality_issues': len(missing_checks) > 0 or sum(extreme_probs.values()) > 0,
                'performance_available': monitoring_results.get('has_performance_metrics', False),
                'drift_alerts': monitoring_results.get('drift_alerts', []),
                'current_f1': monitoring_results.get('f1_score'),
                'current_auc': monitoring_results.get('auc_score')
            }
            
        except Exception as e:
            print(f"❌ Model monitoring failed: {str(e)}")
            raise
    
    @task
    def pipeline_completion_summary(**context):
        """Generate comprehensive pipeline completion summary"""
        print("📋 Generating pipeline completion summary...")
        
        try:
            # Get results from all pipeline stages
            ti = context['task_instance']
            init_result = ti.xcom_pull(task_ids='initialize_pipeline')
            bronze_result = ti.xcom_pull(task_ids='process_bronze_tables')
            silver_result = ti.xcom_pull(task_ids='process_silver_tables')
            gold_result = ti.xcom_pull(task_ids='process_gold_tables')
            training_result = ti.xcom_pull(task_ids='model_training')
            prediction_result = ti.xcom_pull(task_ids='model_predict')
            monitoring_result = ti.xcom_pull(task_ids='model_monitoring')
            
            # Create comprehensive summary
            pipeline_summary = {
                'pipeline_type': 'end_to_end_integrated',
                'completion_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'run_id': training_result.get('run_id') if training_result else 'unknown',
                'overall_status': 'completed_successfully'
            }
            
            # Data Processing Summary
            data_processing_summary = {
                'bronze_layer': {
                    'tables_processed': len(bronze_result["processed_tables"]),
                    'total_partitions': bronze_result["total_partitions"],
                    'errors': len(bronze_result["errors"])
                },
                'silver_layer': {
                    'tables_processed': len(silver_result["processed_tables"]),
                    'total_partitions': silver_result["total_partitions"],
                    'errors': len(silver_result["errors"])
                },
                'gold_layer': {
                    'feature_rows': gold_result["feature_dataset"]["row_count"],
                    'feature_columns': gold_result["feature_dataset"]["column_count"],
                    'label_rows': gold_result["label_dataset"]["row_count"],
                    'label_columns': gold_result["label_dataset"]["column_count"],
                    'status': gold_result["status"]
                }
            }
            
            # Model Training Summary
            model_training_summary = {
                'model_saved': training_result is not None,
                'model_directory': training_result.get('model_dir') if training_result else None,
                'storage_method': training_result.get('storage_method') if training_result else None,
                'has_oot_data': training_result.get('has_oot_data', False) if training_result else False
            }
            
            # Prediction Summary
            prediction_summary = {
                'predictions_generated': prediction_result is not None,
                'total_predictions': prediction_result.get('num_predictions', 0) if prediction_result else 0,
                'positive_rate': prediction_result.get('positive_rate', 0) if prediction_result else 0,
                'threshold_used': prediction_result.get('threshold_used', 0) if prediction_result else 0,
                'predictions_saved': prediction_result.get('predictions_path') is not None if prediction_result else False
            }
            
            if prediction_result and prediction_result.get('has_evaluation'):
                prediction_summary.update({
                    'oot_performance': {
                        'f1': prediction_result.get('oot_f1'),
                        'f15': prediction_result.get('oot_f15'),
                        'auc': prediction_result.get('oot_auc')
                    }
                })
            
            # Monitoring Summary
            monitoring_summary = {
                'monitoring_completed': monitoring_result is not None,
                'monitoring_status': monitoring_result.get('monitoring_status') if monitoring_result else 'failed',
                'data_quality_issues': monitoring_result.get('data_quality_issues', True) if monitoring_result else True,
                'performance_available': monitoring_result.get('performance_available', False) if monitoring_result else False,
                'drift_alerts_count': len(monitoring_result.get('drift_alerts', [])) if monitoring_result else 0
            }
            
            if monitoring_result and monitoring_result.get('performance_available'):
                monitoring_summary.update({
                    'current_performance': {
                        'f1': monitoring_result.get('current_f1'),
                        'auc': monitoring_result.get('current_auc')
                    }
                })
            
            # Compile all summaries
            pipeline_summary.update({
                'data_processing': data_processing_summary,
                'model_training': model_training_summary,
                'predictions': prediction_summary,
                'monitoring': monitoring_summary
            })
            
            # Identify issues across the entire pipeline
            pipeline_issues = []
            
            # Data processing issues
            if bronze_result["total_partitions"] == 0:
                pipeline_issues.append("No bronze tables processed successfully")
            if silver_result["total_partitions"] == 0:
                pipeline_issues.append("No silver tables processed successfully")
            if gold_result["feature_dataset"]["row_count"] == 0:
                pipeline_issues.append("No feature data generated")
            
            # Model training issues
            if not training_result:
                pipeline_issues.append("Model training failed")
            
            # Prediction issues
            if not prediction_result:
                pipeline_issues.append("Prediction generation failed")
            elif prediction_result.get('num_predictions', 0) == 0:
                pipeline_issues.append("No predictions generated")
            
            # Monitoring issues
            if not monitoring_result:
                pipeline_issues.append("Model monitoring failed")
            elif monitoring_result.get('data_quality_issues'):
                pipeline_issues.append("Data quality issues detected")
            
            # Drift alerts
            if monitoring_result and monitoring_result.get('drift_alerts'):
                pipeline_issues.append(f"{len(monitoring_result.get('drift_alerts'))} model drift alerts")
            
            # Update overall status based on issues
            if pipeline_issues:
                pipeline_summary['overall_status'] = 'completed_with_issues'
                pipeline_summary['issues'] = pipeline_issues
            
            # Print comprehensive summary
            print(f"\n🎉 END-TO-END PIPELINE COMPLETION SUMMARY")
            print("=" * 60)
            print(f"Overall Status: {pipeline_summary['overall_status']}")
            print(f"Run ID: {pipeline_summary['run_id']}")
            print(f"Completion Time: {pipeline_summary['completion_timestamp']}")
            
            print(f"\n📊 DATA PROCESSING RESULTS:")
            print(f"  Bronze: {data_processing_summary['bronze_layer']['total_partitions']} partitions processed")
            print(f"  Silver: {data_processing_summary['silver_layer']['total_partitions']} partitions processed")
            print(f"  Gold: {data_processing_summary['gold_layer']['feature_rows']:,} feature rows, {data_processing_summary['gold_layer']['label_rows']:,} label rows")
            
            print(f"\n🤖 MODEL TRAINING RESULTS:")
            print(f"  Model Saved: {'✅' if model_training_summary['model_saved'] else '❌'}")
            print(f"  OOT Data Available: {'✅' if model_training_summary['has_oot_data'] else '❌'}")
            
            print(f"\n🔮 PREDICTION RESULTS:")
            print(f"  Predictions Generated: {prediction_summary['total_predictions']:,}")
            print(f"  Positive Rate: {prediction_summary['positive_rate']*100:.2f}%")
            
            if prediction_summary.get('oot_performance'):
                oot_perf = prediction_summary['oot_performance']
                print(f"  OOT F1: {oot_perf['f1']:.4f}, AUC: {oot_perf['auc']:.4f}")
            
            print(f"\n📈 MONITORING RESULTS:")
            print(f"  Monitoring Status: {monitoring_summary['monitoring_status']}")
            print(f"  Performance Metrics: {'✅' if monitoring_summary['performance_available'] else '❌'}")
            print(f"  Data Quality: {'⚠️ Issues' if monitoring_summary['data_quality_issues'] else '✅ Good'}")
            print(f"  Drift Alerts: {monitoring_summary['drift_alerts_count']}")
            
            if pipeline_issues:
                print(f"\n⚠️ PIPELINE ISSUES ({len(pipeline_issues)}):")
                for i, issue in enumerate(pipeline_issues, 1):
                    print(f"  {i}. {issue}")
            else:
                print(f"\n🎉 PIPELINE COMPLETED SUCCESSFULLY WITH NO ISSUES!")
            
            # Save comprehensive summary
            try:
                summary_path = os.path.join(
                    init_result["config"]["model_training"]["prediction_directory"],
                    f"pipeline_summary_{pipeline_summary['run_id']}.json"
                )
                with open(summary_path, 'w') as f:
                    json.dump(pipeline_summary, f, indent=2)
                print(f"\n📄 Pipeline summary saved to: {summary_path}")
                pipeline_summary['summary_path'] = summary_path
            except Exception as e:
                print(f"⚠️ Could not save pipeline summary: {e}")
            
            return pipeline_summary
            
        except Exception as e:
            print(f"❌ Pipeline summary generation failed: {str(e)}")
            return {
                'overall_status': 'summary_generation_failed',
                'error': str(e),
                'completion_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

    # Define the complete task flow
    init_task = initialize_pipeline()
    bronze_task = process_bronze_tables()
    silver_task = process_silver_tables()
    gold_task = process_gold_tables()
    training_task = model_training()
    prediction_task = model_predict()
    monitoring_task = model_monitoring()
    summary_task = pipeline_completion_summary()
    
    # Set task dependencies for end-to-end pipeline
    init_task >> bronze_task >> silver_task >> gold_task >> training_task >> prediction_task >> monitoring_task >> summary_task

# Instantiate the DAG
dag_instance = integrated_end_to_end_pipeline()