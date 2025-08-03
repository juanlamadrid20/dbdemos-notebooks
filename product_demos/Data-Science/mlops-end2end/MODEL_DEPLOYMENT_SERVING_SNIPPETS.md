# MLOps Model Deployment & Serving Code Snippets

This document contains reusable code snippets for model deployment and serving in MLOps workflows using Databricks. Covers batch inference, real-time endpoints, A/B testing, and monitoring patterns.

## Table of Contents

- [Batch Inference](#batch-inference)
- [Online Feature Tables](#online-feature-tables)
- [Real-time Model Serving Endpoints](#real-time-model-serving-endpoints)
- [A/B Testing & Traffic Management](#ab-testing--traffic-management)
- [Model Monitoring & Drift Detection](#model-monitoring--drift-detection)
- [Inference Tables & Prediction Capture](#inference-tables--prediction-capture)
- [Automated Alerting & Retraining](#automated-alerting--retraining)

---

## Batch Inference

### Basic Batch Inference with MLflow UDF

```python
import mlflow
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository

def run_batch_inference(model_name, model_alias, input_table, output_table=None):
    """
    Run batch inference using MLflow Spark UDF
    
    Args:
        model_name: Full Unity Catalog model name (catalog.schema.model)
        model_alias: Model alias (Champion, Challenger, Production)
        input_table: Input table with features
        output_table: Optional output table to save predictions
    """
    
    # Load input data
    inference_df = spark.read.table(input_table)
    
    # Load model as Spark UDF
    model_uri = f"models:/{model_name}@{model_alias}"
    champion_model = mlflow.pyfunc.spark_udf(
        spark, 
        model_uri=model_uri
        # Use env_manager="virtualenv" for dependency isolation if needed
    )
    
    # Generate predictions
    predictions_df = inference_df.withColumn(
        'predictions', 
        champion_model(*champion_model.metadata.get_input_schema().input_names())
    )
    
    # Save predictions if output table specified
    if output_table:
        predictions_df.write.mode("overwrite").saveAsTable(output_table)
    
    return predictions_df

# Usage
model_name = f"{catalog}.{db}.mlops_churn"
predictions = run_batch_inference(
    model_name=model_name,
    model_alias="Champion", 
    input_table="customer_features",
    output_table="churn_predictions"
)

display(predictions)
```

### Advanced Batch Inference with Feature Store

```python
from databricks.feature_engineering import FeatureEngineeringClient

def batch_inference_with_feature_store(model_name, model_alias, label_table, 
                                      result_type="string"):
    """
    Batch inference with automatic feature lookup from Feature Store
    
    Args:
        model_name: Full Unity Catalog model name
        model_alias: Model alias to use
        label_table: Table with customer IDs and timestamps for feature lookup
        result_type: Expected output type for predictions
    """
    
    fe = FeatureEngineeringClient()
    
    # Load customer IDs for scoring
    inference_df = spark.table(label_table)
    
    # Model URI
    model_uri = f"models:/{model_name}@{model_alias}"
    
    # Batch score with automatic feature lookup
    predictions_df = fe.score_batch(
        df=inference_df, 
        model_uri=model_uri, 
        result_type=result_type
    )
    
    return predictions_df

# Usage
predictions = batch_inference_with_feature_store(
    model_name=f"{catalog}.{db}.advanced_mlops_churn",
    model_alias="Champion",
    label_table="customer_ids_table",
    result_type="string"
)

display(predictions)
```

### Model Requirements Management

```python
def install_model_requirements(model_name, model_alias):
    """Download and install model requirements for inference"""
    
    # Download model requirements
    requirements_path = ModelsArtifactRepository(
        f"models:/{model_name}@{model_alias}"
    ).download_artifacts(artifact_path="requirements.txt")
    
    # Install requirements (use in notebook cell with %pip)
    return requirements_path

# Usage in notebook
# requirements_path = install_model_requirements(model_name, "Champion")
# %pip install --quiet -r $requirements_path
# dbutils.library.restartPython()
```

---

## Online Feature Tables

### Create Online Feature Table

```python
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    OnlineTable, OnlineTableSpec, OnlineTableSpecTriggeredSchedulingPolicy
)

def create_online_feature_table(source_table, online_table_name, 
                               primary_keys, timeseries_key=None):
    """
    Create online feature table for low-latency serving
    
    Args:
        source_table: Source offline feature table
        online_table_name: Name for the online table
        primary_keys: List of primary key columns
        timeseries_key: Optional timeseries column for time-series features
    """
    
    w = WorkspaceClient()
    
    # Enable Change Data Feed on source table
    spark.sql(f"ALTER TABLE {source_table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
    
    # Create online table specification
    online_table_spec = OnlineTableSpec(
        primary_key_columns=primary_keys,
        timeseries_key=timeseries_key,
        source_table_full_name=source_table,
        run_triggered=OnlineTableSpecTriggeredSchedulingPolicy.from_dict(
            {"triggered": "true"}
        ),
    )
    
    online_table = OnlineTable.from_dict({
        "name": online_table_name,
        "spec": online_table_spec.as_dict(),
    })
    
    # Create and wait for online table
    w.online_tables.create_and_wait(table=online_table)
    
    # Wait for table to be ready
    ready_state = w.online_tables.get(online_table_name).status.detailed_state.ONLINE_NO_PENDING_UPDATE
    current_state = w.online_tables.get(online_table_name).status.detailed_state
    
    while current_state != ready_state:
        ol_table = w.online_tables.get(online_table_name)
        current_state = ol_table.status.detailed_state
        time.sleep(30)
    
    print(f"Online table {online_table_name} is ready!")
    
    return w.online_tables.get(online_table_name)

# Usage
online_table = create_online_feature_table(
    source_table=f"{catalog}.{db}.churn_feature_table",
    online_table_name=f"{catalog}.{db}.churn_feature_table_online",
    primary_keys=["customer_id"],
    timeseries_key="transaction_ts"
)
```

### Online Table Management

```python
def manage_online_table(online_table_name, action="status"):
    """
    Manage online feature table operations
    
    Args:
        online_table_name: Full name of online table
        action: 'status', 'refresh', 'delete'
    """
    
    w = WorkspaceClient()
    
    if action == "status":
        try:
            table_info = w.online_tables.get(online_table_name)
            print(f"Table Status: {table_info.status.detailed_state}")
            return table_info
        except Exception as e:
            print(f"Table not found or error: {e}")
            return None
    
    elif action == "refresh":
        # Trigger refresh (useful when source data changes)
        try:
            table_info = w.online_tables.get(online_table_name)
            w.pipelines.start_update(
                pipeline_id=table_info.spec.pipeline_id, 
                full_refresh=True
            )
            print(f"Refresh triggered for {online_table_name}")
        except Exception as e:
            print(f"Refresh failed: {e}")
    
    elif action == "delete":
        try:
            w.online_tables.delete(online_table_name)
            print(f"Deleted online table: {online_table_name}")
        except Exception as e:
            print(f"Delete failed: {e}")

# Usage
manage_online_table(f"{catalog}.{db}.churn_feature_table_online", "status")
manage_online_table(f"{catalog}.{db}.churn_feature_table_online", "refresh")
```

---

## Real-time Model Serving Endpoints

### Create Model Serving Endpoint

```python
from databricks.sdk.service.serving import EndpointCoreConfigInput, EndpointTag
from mlflow import MlflowClient

def create_serving_endpoint(endpoint_name, model_name, model_alias="Champion",
                          workload_size="Small", auto_capture_config=None):
    """
    Create real-time model serving endpoint
    
    Args:
        endpoint_name: Name for the serving endpoint
        model_name: Full Unity Catalog model name
        model_alias: Model alias to serve
        workload_size: Compute size (Small, Medium, Large)
        auto_capture_config: Optional inference logging config
    """
    
    w = WorkspaceClient()
    client = MlflowClient()
    
    # Get model version from alias
    model_version = client.get_model_version_by_alias(model_name, model_alias).version
    served_model_name = model_name.split('.')[-1]
    
    # Configure endpoint
    endpoint_config_dict = {
        "served_models": [{
            "model_name": model_name,
            "model_version": model_version,
            "scale_to_zero_enabled": True,
            "workload_size": workload_size,
        }],
        "traffic_config": {
            "routes": [{
                "served_model_name": f"{served_model_name}-{model_version}",
                "traffic_percentage": 100
            }]
        }
    }
    
    # Add auto-capture config if provided
    if auto_capture_config:
        endpoint_config_dict["auto_capture_config"] = auto_capture_config
    
    endpoint_config = EndpointCoreConfigInput.from_dict(endpoint_config_dict)
    
    try:
        # Create endpoint
        w.serving_endpoints.create(
            name=endpoint_name,
            config=endpoint_config,
            tags=[EndpointTag.from_dict({"key": "environment", "value": "production"})]
        )
        print(f"Creating endpoint {endpoint_name} with model {model_name}@{model_alias}")
        
    except Exception as e:
        if f"Endpoint with name '{endpoint_name}' already exists" in str(e):
            print(f"Endpoint exists, updating with new model version")
            w.serving_endpoints.update_config(
                name=endpoint_name,
                served_models=endpoint_config.served_models,
                traffic_config=endpoint_config.traffic_config
            )
        else:
            raise e
    
    # Wait for endpoint to be ready
    from datetime import timedelta
    endpoint = w.serving_endpoints.wait_get_serving_endpoint_not_updating(
        endpoint_name, 
        timeout=timedelta(minutes=120)
    )
    
    assert endpoint.state.config_update.value == "NOT_UPDATING" and \
           endpoint.state.ready.value == "READY", "Endpoint not ready"
    
    print(f"✅ Endpoint {endpoint_name} is ready!")
    return endpoint

# Usage with inference logging
auto_capture_config = {
    "catalog_name": catalog,
    "schema_name": db,
    "table_name_prefix": "churn_served"
}

endpoint = create_serving_endpoint(
    endpoint_name="churn_prediction_endpoint",
    model_name=f"{catalog}.{db}.mlops_churn",
    model_alias="Champion",
    workload_size="Small",
    auto_capture_config=auto_capture_config
)
```

### Query Serving Endpoint

```python
def query_serving_endpoint(endpoint_name, input_data, format_type="dataframe_records"):
    """
    Query serving endpoint with input data
    
    Args:
        endpoint_name: Name of the serving endpoint
        input_data: Input data for prediction
        format_type: Input format ('dataframe_records', 'dataframe_split')
    """
    
    w = WorkspaceClient()
    
    if format_type == "dataframe_records":
        response = w.serving_endpoints.query(
            name=endpoint_name, 
            dataframe_records=input_data
        )
    else:
        response = w.serving_endpoints.query(
            name=endpoint_name,
            inputs=input_data
        )
    
    return response.predictions

# Usage
input_data = [
    {"customer_id": "0002-ORFBO", "transaction_ts": "2024-09-10"},
    {"customer_id": "0003-MKNFE", "transaction_ts": "2024-09-10"}
]

predictions = query_serving_endpoint("churn_prediction_endpoint", input_data)
print(f"Predictions: {predictions}")
```

### Get Model Input Example

```python
def get_model_input_example(model_name, model_version):
    """Get input example from registered model"""
    from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
    from mlflow.models import Model
    
    # Download model artifacts
    repo = ModelsArtifactRepository(f"models:/{model_name}/{model_version}")
    model_path = repo.download_artifacts("")
    
    # Load input example
    model = Model.load(model_path)
    input_example = model.load_input_example(model_path)
    
    if input_example is not None:
        return input_example.to_dict(orient='records')
    else:
        return None

# Usage
input_example = get_model_input_example(model_name, model_version)
if input_example:
    predictions = query_serving_endpoint("churn_prediction_endpoint", input_example)
```

---

## A/B Testing & Traffic Management

### Multi-Model Endpoint with Traffic Splitting

```python
def setup_ab_testing_endpoint(endpoint_name, model_configs, auto_capture_config=None):
    """
    Setup A/B testing endpoint with multiple model versions
    
    Args:
        endpoint_name: Name of serving endpoint
        model_configs: List of dicts with model_name, alias, traffic_percentage
        auto_capture_config: Optional inference logging config
    
    Example model_configs:
    [
        {"model_name": "catalog.db.model", "alias": "Champion", "traffic_percentage": 80},
        {"model_name": "catalog.db.model", "alias": "Challenger", "traffic_percentage": 20}
    ]
    """
    
    w = WorkspaceClient()
    client = MlflowClient()
    
    served_models = []
    routes = []
    
    for config in model_configs:
        model_name = config["model_name"]
        alias = config["alias"] 
        traffic_percentage = config["traffic_percentage"]
        
        # Get model version
        model_version = client.get_model_version_by_alias(model_name, alias).version
        served_model_name = f"{model_name.split('.')[-1]}-{alias.lower()}-{model_version}"
        
        # Add served model
        served_models.append({
            "model_name": model_name,
            "model_version": model_version,
            "scale_to_zero_enabled": True,
            "workload_size": "Small",
            "name": served_model_name
        })
        
        # Add traffic route
        routes.append({
            "served_model_name": served_model_name,
            "traffic_percentage": traffic_percentage
        })
    
    # Validate traffic percentages sum to 100
    total_traffic = sum(route["traffic_percentage"] for route in routes)
    assert total_traffic == 100, f"Traffic percentages must sum to 100, got {total_traffic}"
    
    endpoint_config_dict = {
        "served_models": served_models,
        "traffic_config": {"routes": routes}
    }
    
    if auto_capture_config:
        endpoint_config_dict["auto_capture_config"] = auto_capture_config
    
    endpoint_config = EndpointCoreConfigInput.from_dict(endpoint_config_dict)
    
    try:
        w.serving_endpoints.create(
            name=endpoint_name,
            config=endpoint_config,
            tags=[EndpointTag.from_dict({"key": "test_type", "value": "ab_testing"})]
        )
        print(f"Created A/B testing endpoint: {endpoint_name}")
        
    except Exception as e:
        if f"Endpoint with name '{endpoint_name}' already exists" in str(e):
            print(f"Updating existing endpoint with A/B configuration")
            w.serving_endpoints.update_config(
                name=endpoint_name,
                served_models=endpoint_config.served_models,
                traffic_config=endpoint_config.traffic_config
            )
        else:
            raise e
    
    return endpoint_config

# Usage
model_configs = [
    {
        "model_name": f"{catalog}.{db}.mlops_churn",
        "alias": "Champion", 
        "traffic_percentage": 80
    },
    {
        "model_name": f"{catalog}.{db}.mlops_churn",
        "alias": "Challenger", 
        "traffic_percentage": 20
    }
]

ab_endpoint = setup_ab_testing_endpoint(
    endpoint_name="churn_ab_testing_endpoint",
    model_configs=model_configs,
    auto_capture_config={
        "catalog_name": catalog,
        "schema_name": db,
        "table_name_prefix": "churn_ab_test"
    }
)
```

### Traffic Management Functions

```python
def update_traffic_split(endpoint_name, new_traffic_config):
    """
    Update traffic split for existing endpoint
    
    Args:
        endpoint_name: Name of serving endpoint
        new_traffic_config: New traffic configuration
    """
    
    w = WorkspaceClient()
    
    # Get current endpoint config
    current_endpoint = w.serving_endpoints.get(endpoint_name)
    
    # Update traffic config
    w.serving_endpoints.update_config(
        name=endpoint_name,
        served_models=current_endpoint.config.served_models,
        traffic_config=new_traffic_config
    )
    
    print(f"Updated traffic split for {endpoint_name}")

def promote_challenger_to_production(endpoint_name, challenger_alias="Challenger"):
    """
    Promote challenger model to 100% traffic after successful A/B test
    """
    
    w = WorkspaceClient()
    current_endpoint = w.serving_endpoints.get(endpoint_name)
    
    # Find challenger model in served models
    challenger_model = None
    for model in current_endpoint.config.served_models:
        if challenger_alias.lower() in model.name.lower():
            challenger_model = model
            break
    
    if challenger_model:
        # Update traffic to send 100% to challenger
        new_traffic_config = {
            "routes": [{
                "served_model_name": challenger_model.name,
                "traffic_percentage": 100
            }]
        }
        
        update_traffic_split(endpoint_name, new_traffic_config)
        print(f"Promoted {challenger_alias} model to 100% traffic")
    else:
        raise ValueError(f"No {challenger_alias} model found in endpoint")

# Usage
# Gradually increase challenger traffic
new_traffic = {
    "routes": [
        {"served_model_name": "mlops_churn-champion-1", "traffic_percentage": 60},
        {"served_model_name": "mlops_churn-challenger-2", "traffic_percentage": 40}
    ]
}

update_traffic_split("churn_ab_testing_endpoint", new_traffic)
```

---

## Model Monitoring & Drift Detection

### Setup Lakehouse Monitoring

```python
from databricks.sdk.service.catalog import (
    MonitorInferenceLog, MonitorInferenceLogProblemType, MonitorMetric, MonitorMetricType
)
from pyspark.sql.types import DoubleType, StructField

def create_inference_monitor(table_name, baseline_table_name, output_schema_name,
                           prediction_col="prediction", label_col=None, 
                           timestamp_col="inference_timestamp", model_id_col="model_version",
                           custom_metrics=None, slicing_exprs=None):
    """
    Create comprehensive inference monitoring setup
    
    Args:
        table_name: Full name of inference table to monitor
        baseline_table_name: Baseline/training data table
        output_schema_name: Schema to store monitoring results
        prediction_col: Column containing model predictions
        label_col: Column containing ground truth labels (optional)
        timestamp_col: Timestamp column for time-series analysis
        model_id_col: Column identifying model version
        custom_metrics: List of custom business metrics
        slicing_exprs: List of slicing expressions for segmented analysis
    """
    
    w = WorkspaceClient()
    
    # Define custom business metrics if not provided
    if custom_metrics is None:
        custom_metrics = [
            MonitorMetric(
                type=MonitorMetricType.CUSTOM_METRIC_TYPE_AGGREGATE,
                name="expected_loss",
                input_columns=[":table"],
                definition="""avg(CASE
                WHEN {{prediction_col}} != {{label_col}} AND {{label_col}} = 'Yes' THEN -monthly_charges
                ELSE 0 END
                )""",
                output_data_type=StructField("output", DoubleType()).json()
            )
        ]
    
    # Default slicing expressions if not provided
    if slicing_exprs is None:
        slicing_exprs = ["senior_citizen='Yes'", "contract"]
    
    try:
        monitor_info = w.quality_monitors.create(
            table_name=table_name,
            inference_log=MonitorInferenceLog(
                problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_CLASSIFICATION,
                prediction_col=prediction_col,
                timestamp_col=timestamp_col,
                granularities=["1 day"],
                model_id_col=model_id_col,
                label_col=label_col,
            ),
            assets_dir="/tmp/monitoring",
            output_schema_name=output_schema_name,
            baseline_table_name=baseline_table_name,
            slicing_exprs=slicing_exprs,
            custom_metrics=custom_metrics
        )
        
        print(f"✅ Monitor created for {table_name}")
        
    except Exception as e:
        if "already exist" in str(e).lower():
            print(f"Monitor already exists, retrieving info")
            monitor_info = w.quality_monitors.get(table_name=table_name)
        else:
            raise e
    
    # Wait for monitor to be active
    while monitor_info.status.value == "MONITOR_STATUS_PENDING":
        monitor_info = w.quality_monitors.get(table_name=table_name)
        time.sleep(10)
    
    print(f"Monitor status: {monitor_info.status.value}")
    return monitor_info

# Usage
monitor_info = create_inference_monitor(
    table_name=f"{catalog}.{db}.churn_inference_table",
    baseline_table_name=f"{catalog}.{db}.churn_baseline",
    output_schema_name=f"{catalog}.{db}",
    prediction_col="prediction",
    label_col="churn",
    timestamp_col="inference_timestamp",
    model_id_col="model_version"
)
```

### Monitor Management & Refresh

```python
def manage_monitor(table_name, action="refresh", wait_for_completion=True):
    """
    Manage monitor operations
    
    Args:
        table_name: Full name of monitored table
        action: 'refresh', 'status', 'delete'
        wait_for_completion: Whether to wait for refresh completion
    """
    
    w = WorkspaceClient()
    
    if action == "refresh":
        # Trigger monitor refresh
        refresh_info = w.quality_monitors.run_refresh(table_name=table_name)
        print(f"Triggered refresh for {table_name}")
        
        if wait_for_completion:
            from databricks.sdk.service.catalog import MonitorRefreshInfoState
            
            while refresh_info.state in (MonitorRefreshInfoState.PENDING, MonitorRefreshInfoState.RUNNING):
                refresh_info = w.quality_monitors.get_refresh(
                    table_name=table_name, 
                    refresh_id=refresh_info.refresh_id
                )
                print(f"Refresh status: {refresh_info.state}")
                time.sleep(30)
            
            print(f"✅ Refresh completed: {refresh_info.state}")
        
        return refresh_info
    
    elif action == "status":
        monitor_info = w.quality_monitors.get(table_name=table_name)
        print(f"Monitor Status: {monitor_info.status}")
        print(f"Drift Table: {monitor_info.drift_metrics_table_name}")
        print(f"Profile Table: {monitor_info.profile_metrics_table_name}")
        return monitor_info
    
    elif action == "delete":
        w.quality_monitors.delete(table_name=table_name, purge_artifacts=True)
        print(f"Deleted monitor for {table_name}")

# Usage
manage_monitor(f"{catalog}.{db}.churn_inference_table", "refresh")
monitor_info = manage_monitor(f"{catalog}.{db}.churn_inference_table", "status")
```

---

## Inference Tables & Prediction Capture

### Create Inference Table with Metadata

```python
def create_inference_table_with_metadata(predictions_df, model_name, model_alias, 
                                       table_name, include_timestamp=True):
    """
    Create inference table with model metadata for monitoring
    
    Args:
        predictions_df: DataFrame with predictions
        model_name: Full model name
        model_alias: Model alias used
        table_name: Name for inference table
        include_timestamp: Whether to add inference timestamp
    """
    
    from datetime import datetime, timedelta
    import pyspark.sql.functions as F
    from mlflow import MlflowClient
    
    client = MlflowClient()
    
    # Get model version info
    model_version = int(client.get_model_version_by_alias(model_name, model_alias).version)
    
    # Add metadata columns
    inference_df = predictions_df.withColumn("model_name", F.lit(model_name)) \
                                .withColumn("model_version", F.lit(model_version)) \
                                .withColumn("model_alias", F.lit(model_alias))
    
    if include_timestamp:
        # Add inference timestamp (can be offset for demo purposes)
        inference_df = inference_df.withColumn(
            "inference_timestamp", 
            F.lit(datetime.now() - timedelta(days=2))  # Offset for demo
        )
    
    # Save inference table
    inference_df.write.mode("overwrite").saveAsTable(table_name)
    
    # Enable Change Data Feed for monitoring
    spark.sql(f"ALTER TABLE {table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
    
    print(f"✅ Created inference table: {table_name}")
    print(f"   Model: {model_name}@{model_alias} (v{model_version})")
    print(f"   Rows: {inference_df.count()}")
    
    return inference_df

# Usage
predictions_with_metadata = create_inference_table_with_metadata(
    predictions_df=predictions,
    model_name=f"{catalog}.{db}.mlops_churn",
    model_alias="Champion",
    table_name=f"{catalog}.{db}.churn_inference_table"
)

display(predictions_with_metadata)
```

### Merge Online and Offline Inference Data

```python
def create_unified_inference_table(offline_table, online_table, 
                                 label_table, output_table):
    """
    Create unified inference table combining online and offline predictions
    
    Args:
        offline_table: Batch inference results
        online_table: Real-time inference results (from auto-capture)
        label_table: Ground truth labels when available
        output_table: Combined inference table name
    """
    
    # Union offline and online inference data
    offline_df = spark.table(offline_table)
    
    try:
        online_df = spark.table(online_table)
        # Union the tables if both exist
        combined_df = offline_df.unionByName(online_df, allowMissingColumns=True)
    except:
        print("Online table not found, using offline data only")
        combined_df = offline_df
    
    # Join with labels if available
    try:
        labels_df = spark.table(label_table)
        # Join on customer_id and timestamp
        inference_with_labels = combined_df.join(
            labels_df, 
            on=["customer_id", "transaction_ts"], 
            how="left"
        )
    except:
        print("Label table not found, proceeding without labels")
        inference_with_labels = combined_df
    
    # Remove any split columns and save
    final_df = inference_with_labels.drop("split") if "split" in inference_with_labels.columns else inference_with_labels
    
    final_df.write.mode("overwrite").saveAsTable(output_table)
    
    # Enable Change Data Feed
    spark.sql(f"ALTER TABLE {output_table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
    
    print(f"✅ Created unified inference table: {output_table}")
    return final_df

# Usage
unified_inference = create_unified_inference_table(
    offline_table=f"{catalog}.{db}.churn_offline_inference",
    online_table=f"{catalog}.{db}.churn_served_requests",
    label_table=f"{catalog}.{db}.churn_label_table",
    output_table=f"{catalog}.{db}.churn_unified_inference"
)
```

---

## Automated Alerting & Retraining

### Drift Detection and Violation Counting

```python
def detect_model_drift_violations(monitor_table_name, drift_thresholds=None, 
                                 performance_thresholds=None, model_id="*"):
    """
    Detect drift violations from monitoring tables and count violations
    
    Args:
        monitor_table_name: Base name of monitored table
        drift_thresholds: Dict with drift metric thresholds
        performance_thresholds: Dict with performance thresholds
        model_id: Model ID to filter on
    """
    
    if drift_thresholds is None:
        drift_thresholds = {
            "prediction_drift": 0.2,
            "label_drift": 0.2
        }
    
    if performance_thresholds is None:
        performance_thresholds = {
            "f1_score": 0.4,
            "expected_loss": 30
        }
    
    w = WorkspaceClient()
    
    # Get monitoring table names
    monitor_info = w.quality_monitors.get(table_name=monitor_table_name)
    drift_table_name = monitor_info.drift_metrics_table_name
    profile_table_name = monitor_info.profile_metrics_table_name
    
    # Query performance metrics
    performance_query = f"""
    SELECT
      window.start as time,
      f1_score.macro AS f1_score,
      expected_loss,
      Model_Version AS model_id
    FROM {profile_table_name}
    WHERE
      window.start >= "2024-06-01"
      AND log_type = "INPUT"
      AND column_name = ":table"
      AND slice_key is null
      AND slice_value is null
      AND Model_Version = '{model_id}'
    ORDER BY window.start
    """
    
    performance_df = spark.sql(performance_query)
    
    # Query drift metrics
    drift_query = f"""
    SELECT
      window.start AS time,
      column_name,
      js_distance AS drift_metric,
      Model_Version AS model_id
    FROM {drift_table_name}
    WHERE
      column_name IN ('prediction', 'churn')
      AND window.start >= "2024-06-01"
      AND slice_key is null
      AND slice_value is null
      AND Model_Version = '{model_id}'
      AND drift_type = "CONSECUTIVE"
    ORDER BY window.start
    """
    
    drift_df = spark.sql(drift_query)
    
    # Count violations
    performance_violations = performance_df.filter(
        (F.col("f1_score") < performance_thresholds["f1_score"]) & 
        (F.abs(F.col("expected_loss")) > performance_thresholds["expected_loss"])
    ).count()
    
    drift_violations = 0
    if not drift_df.isEmpty():
        # Pivot drift metrics
        drift_pivoted = drift_df.groupBy("time", "model_id") \
                               .pivot("column_name") \
                               .agg(F.first("drift_metric"))
        
        drift_violations = drift_pivoted.filter(
            (F.col("churn") > drift_thresholds["label_drift"]) & 
            (F.col("prediction") > drift_thresholds["prediction_drift"])
        ).count()
    
    total_violations = performance_violations + drift_violations
    
    violation_summary = {
        "performance_violations": performance_violations,
        "drift_violations": drift_violations,
        "total_violations": total_violations,
        "thresholds_used": {
            "drift": drift_thresholds,
            "performance": performance_thresholds
        }
    }
    
    print(f"📊 Violation Summary:")
    print(f"   Performance violations: {performance_violations}")
    print(f"   Drift violations: {drift_violations}")
    print(f"   Total violations: {total_violations}")
    
    return violation_summary

# Usage
violations = detect_model_drift_violations(
    monitor_table_name=f"{catalog}.{db}.churn_inference_table",
    drift_thresholds={"prediction_drift": 0.19, "label_drift": 0.19},
    performance_thresholds={"f1_score": 0.5, "expected_loss": 30},
    model_id="*"
)
```

### Automated Retraining Trigger

```python
def setup_retraining_workflow(violation_threshold=1, notification_config=None):
    """
    Setup automated retraining workflow based on violations
    
    Args:
        violation_threshold: Number of violations to trigger retraining
        notification_config: Dict with notification settings
    """
    
    def trigger_retraining_job(job_name, violation_count):
        """Trigger retraining job if violations exceed threshold"""
        
        if violation_count >= violation_threshold:
            print(f"🚨 Violations ({violation_count}) exceed threshold ({violation_threshold})")
            print(f"🔄 Triggering retraining job: {job_name}")
            
            # Set task value for workflow branching
            dbutils.jobs.taskValues.set(
                key='all_violations_count', 
                value=violation_count
            )
            
            # In a Databricks Job, this would trigger the retraining task
            # w.jobs.run_now(job_id=retraining_job_id)
            
            return True
        else:
            print(f"✅ Violations ({violation_count}) below threshold ({violation_threshold})")
            return False
    
    def send_notification(violation_count, channel="slack"):
        """Send notification about model performance"""
        
        if notification_config and channel in notification_config:
            # Implementation would depend on your notification system
            # Examples: Slack webhook, email, PagerDuty, etc.
            message = f"Model violations detected: {violation_count}"
            print(f"📧 Sending {channel} notification: {message}")
            
            # Example Slack webhook call:
            # requests.post(notification_config[channel]['webhook_url'], 
            #              json={"text": message})
    
    return trigger_retraining_job, send_notification

# Usage
trigger_retraining, send_notification = setup_retraining_workflow(
    violation_threshold=2,
    notification_config={
        "slack": {"webhook_url": "https://hooks.slack.com/..."},
        "email": {"recipients": ["ml-team@company.com"]}
    }
)

# In your monitoring workflow
violations_summary = detect_model_drift_violations(
    f"{catalog}.{db}.churn_inference_table"
)

# Check if retraining should be triggered
should_retrain = trigger_retraining("churn_model_training_job", violations_summary["total_violations"])

if should_retrain:
    send_notification(violations_summary["total_violations"], "slack")
```

### Workflow Integration Patterns

```python
def create_monitoring_workflow_config():
    """
    Create configuration for automated monitoring workflow
    """
    
    workflow_config = {
        "schedule": "0 0 * * *",  # Daily at midnight
        "tasks": [
            {
                "task_key": "refresh_monitor",
                "notebook_task": {"notebook_path": "/path/to/refresh_monitor_notebook"},
                "depends_on": []
            },
            {
                "task_key": "detect_violations", 
                "notebook_task": {"notebook_path": "/path/to/drift_detection_notebook"},
                "depends_on": [{"task_key": "refresh_monitor"}]
            },
            {
                "task_key": "check_violations",
                "condition_task": {
                    "op": "GREATER_THAN",
                    "left": "{{tasks.detect_violations.values.all_violations_count}}",
                    "right": "1"
                },
                "depends_on": [{"task_key": "detect_violations"}]
            },
            {
                "task_key": "trigger_retraining",
                "run_job_task": {"job_id": "retraining_job_id"},
                "depends_on": [{"task_key": "check_violations"}]
            },
            {
                "task_key": "send_alert",
                "notebook_task": {"notebook_path": "/path/to/alert_notebook"},
                "depends_on": [{"task_key": "check_violations"}]
            }
        ]
    }
    
    return workflow_config

# Usage
workflow_config = create_monitoring_workflow_config()
print("📋 Monitoring Workflow Configuration:")
print(json.dumps(workflow_config, indent=2))
```

---

## Best Practices

### 1. Environment Management

```python
def setup_deployment_environment(environment="production"):
    """Setup environment-specific configurations"""
    
    env_configs = {
        "development": {
            "workload_size": "Small",
            "scale_to_zero": True,
            "auto_capture": False
        },
        "staging": {
            "workload_size": "Medium", 
            "scale_to_zero": True,
            "auto_capture": True
        },
        "production": {
            "workload_size": "Large",
            "scale_to_zero": False,
            "auto_capture": True
        }
    }
    
    return env_configs.get(environment, env_configs["development"])
```

### 2. Model Version Management

```python
def promote_model_across_environments(model_name, source_alias, target_alias):
    """Promote model from one alias to another"""
    
    client = MlflowClient()
    
    # Get source model version
    source_model = client.get_model_version_by_alias(model_name, source_alias)
    
    # Promote to target alias
    client.set_registered_model_alias(
        name=model_name,
        alias=target_alias,
        version=source_model.version
    )
    
    print(f"✅ Promoted {model_name} v{source_model.version} from {source_alias} to {target_alias}")
```

### 3. Monitoring Best Practices

```python
def monitoring_best_practices_checklist():
    """Checklist for model monitoring setup"""
    
    checklist = [
        "✅ Enable Change Data Feed on inference tables",
        "✅ Set up baseline table with training data distribution", 
        "✅ Define business-relevant custom metrics",
        "✅ Configure appropriate slicing dimensions",
        "✅ Set up automated refresh schedule",
        "✅ Define violation thresholds based on business impact",
        "✅ Configure alerting and notification channels",
        "✅ Set up automated retraining triggers",
        "✅ Monitor both technical and business metrics",
        "✅ Regular review of monitoring dashboards"
    ]
    
    for item in checklist:
        print(item)
```

---

## Usage Examples

### Complete Deployment Pipeline

```python
# 1. Setup online features
online_table = create_online_feature_table(
    source_table=f"{catalog}.{db}.churn_features",
    online_table_name=f"{catalog}.{db}.churn_features_online",
    primary_keys=["customer_id"],
    timeseries_key="transaction_ts"
)

# 2. Create serving endpoint with inference capture
endpoint = create_serving_endpoint(
    endpoint_name="churn_production_endpoint",
    model_name=f"{catalog}.{db}.mlops_churn",
    model_alias="Champion",
    auto_capture_config={
        "catalog_name": catalog,
        "schema_name": db, 
        "table_name_prefix": "churn_prod"
    }
)

# 3. Setup monitoring
monitor = create_inference_monitor(
    table_name=f"{catalog}.{db}.churn_inference_table",
    baseline_table_name=f"{catalog}.{db}.churn_baseline",
    output_schema_name=f"{catalog}.{db}"
)

# 4. Setup automated violation detection
violations = detect_model_drift_violations(
    f"{catalog}.{db}.churn_inference_table"
)

# 5. Configure retraining triggers
trigger_retraining, send_notification = setup_retraining_workflow(violation_threshold=2)
should_retrain = trigger_retraining("retraining_job", violations["total_violations"])
```

This provides a complete framework for model deployment, serving, and monitoring in production MLOps workflows using Databricks.