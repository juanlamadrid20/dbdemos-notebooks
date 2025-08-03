# Model Deployment and Serving Best Practices for MLOps on Databricks

This guide captures model deployment and serving best practices from the Databricks MLOps end-to-end demo, covering batch inference, real-time serving endpoints, A/B testing, monitoring, and drift detection.

## Overview

The project demonstrates a comprehensive deployment pipeline:
- **Quickstart**: Simple batch inference with Spark UDFs
- **Advanced**: Feature Store integration, online tables, real-time serving, and monitoring

Both approaches leverage Unity Catalog models with Champion/Challenger patterns and include automated monitoring for production readiness.

## Key Dependencies

```python
# Core deployment packages
%pip install mlflow==2.22.0
%pip install databricks-sdk==0.40.0
%pip install databricks-feature-engineering==0.12.1  # Advanced only
```

## Batch Inference Patterns

### 1. Simple Batch Inference (Quickstart)

**Best Practice**: Use Spark UDFs for distributed batch processing with Unity Catalog models.

```python
import mlflow
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository

# Dynamic dependency installation from model artifacts
requirements_path = ModelsArtifactRepository(
    f"models:/{catalog}.{db}.mlops_churn@Champion"
).download_artifacts(artifact_path="requirements.txt")

%pip install --quiet -r $requirements_path
dbutils.library.restartPython()

# Load model as Spark UDF
champion_model = mlflow.pyfunc.spark_udf(
    spark, 
    model_uri=f"models:/{catalog}.{db}.mlops_churn@Champion"
)

# Batch scoring
inference_df = spark.read.table("mlops_churn_inference")
preds_df = inference_df.withColumn(
    'predictions', 
    champion_model(*champion_model.metadata.get_input_schema().input_names())
)

display(preds_df)
```

**Key Benefits**:
- Automatic dependency management from model artifacts
- Distributed processing across Spark cluster
- Unity Catalog alias support for model lifecycle management

### 2. Feature Store Batch Inference (Advanced)

**Best Practice**: Use Feature Engineering client for automatic feature lookup and on-demand computation.

```python
from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()

# Load customer IDs for scoring
inference_df = spark.read.table("advanced_churn_cust_ids")

# Model URI with alias
model_uri = f"models:/{catalog}.{db}.advanced_mlops_churn@Champion"

# Batch score with automatic feature lookup
preds_df = fe.score_batch(
    df=inference_df, 
    model_uri=model_uri, 
    result_type="string"
)

display(preds_df)
```

**Key Benefits**:
- Automatic feature lookup from feature tables
- On-demand feature computation using Unity Catalog functions
- Point-in-time correctness for time-series features

### 3. Inference Tracking for Monitoring

**Best Practice**: Save predictions with metadata for downstream monitoring.

```python
from mlflow import MlflowClient
from datetime import datetime, timedelta

client = MlflowClient()
model_version = int(client.get_model_version_by_alias(
    name=model_name, alias="Champion"
).version)

# Add tracking metadata
offline_inference_df = preds_df.withColumn("model_name", F.lit(model_name)) \
                              .withColumn("model_version", F.lit(model_version)) \
                              .withColumn("model_alias", F.lit("Champion")) \
                              .withColumn("inference_timestamp", F.lit(datetime.now()))

# Save for monitoring
offline_inference_df.write.mode("overwrite") \
                    .saveAsTable("advanced_churn_offline_inference")
```

## Real-Time Serving Architecture

### 1. Online Feature Tables

**Best Practice**: Enable low-latency feature serving with online tables.

#### Enable Change Data Feed
```sql
-- Essential for efficient online table updates
ALTER TABLE advanced_churn_feature_table 
SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
```

#### Create Online Table via API
```python
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    OnlineTable, OnlineTableSpec, OnlineTableSpecTriggeredSchedulingPolicy
)

w = WorkspaceClient()

# Define online table specification
churn_features_online_store_spec = OnlineTableSpec(
    primary_key_columns=["customer_id"],
    timeseries_key="transaction_ts",
    source_table_full_name=f"{catalog}.{db}.advanced_churn_feature_table",
    run_triggered=OnlineTableSpecTriggeredSchedulingPolicy.from_dict(
        {"triggered": "true"}
    ),
)

churn_features_online_table = OnlineTable.from_dict({
    "name": f"{catalog}.{db}.advanced_churn_feature_table_online_table",
    "spec": churn_features_online_store_spec.as_dict(),
})

# Create and wait for completion
w.online_tables.create_and_wait(table=churn_features_online_table)
```

#### Monitor Online Table Status
```python
# Check readiness
online_table_status = w.online_tables.get(
    f"{catalog}.{db}.advanced_churn_feature_table_online_table"
)

ready_state = online_table_status.status.detailed_state.ONLINE_NO_PENDING_UPDATE
current_state = online_table_status.status.detailed_state

while current_state != ready_state:
    online_table_status = w.online_tables.get(
        f"{catalog}.{db}.advanced_churn_feature_table_online_table"
    )
    current_state = online_table_status.status.detailed_state
    time.sleep(30)

print("Online table is ready.")
```

### 2. Model Serving Endpoints

**Best Practice**: Deploy models with A/B testing capability and auto-scaling.

#### Alias Management for A/B Testing
```python
# Promote Champion to Production for serving
endpoint_name = "advanced_mlops_churn_ep"
model_name = f"{catalog}.{db}.advanced_mlops_churn"
model_version = client.get_model_version_by_alias(
    name=model_name, alias="Champion"
).version

# Set Production alias
client.set_registered_model_alias(
    name=model_name, 
    alias="Production", 
    version=model_version
)
```

#### Create Serving Endpoint
```python
from databricks.sdk.service.serving import EndpointCoreConfigInput, EndpointTag

# Parse model name for serving
served_model_name = model_name.split('.')[-1]

# Define endpoint configuration
endpoint_config_dict = {
    "served_models": [{
        "model_name": model_name,
        "model_version": model_version,
        "scale_to_zero_enabled": True,
        "workload_size": "Small",
    }],
    "traffic_config": {
        "routes": [{
            "served_model_name": f"{served_model_name}-{model_version}",
            "traffic_percentage": 100
        }]
    },
    "auto_capture_config": {
        "catalog_name": catalog,
        "schema_name": db,
        "table_name_prefix": "advanced_churn_served"
    }
}

endpoint_config = EndpointCoreConfigInput.from_dict(endpoint_config_dict)

# Create endpoint with monitoring
try:
    w.serving_endpoints.create(
        name=endpoint_name,
        config=endpoint_config,
        tags=[EndpointTag.from_dict({"key": "dbdemos", "value": "advanced_mlops_churn"})]
    )
except Exception as e:
    if f"Endpoint with name '{endpoint_name}' already exists" in str(e):
        # Update existing endpoint
        w.serving_endpoints.update_config(
            name=endpoint_name,
            served_models=endpoint_config.served_models,
            traffic_config=endpoint_config.traffic_config
        )
```

#### Wait for Endpoint Readiness
```python
from datetime import timedelta

# Wait for deployment completion
endpoint = w.serving_endpoints.wait_get_serving_endpoint_not_updating(
    endpoint_name, 
    timeout=timedelta(minutes=120)
)

assert (endpoint.state.config_update.value == "NOT_UPDATING" and 
        endpoint.state.ready.value == "READY"), "Endpoint not ready or failed"
```

### 3. Real-Time Inference Testing

**Best Practice**: Test endpoints with proper input formatting and error handling.

```python
# Test with customer IDs and timestamps
dataframe_records = [
    {"customer_id": "0002-ORFBO", "transaction_ts": "2024-09-10"},
    {"customer_id": "0003-MKNFE", "transaction_ts": "2024-09-10"}
]

# Query endpoint
import time
time.sleep(60)  # Allow endpoint to fully initialize

response = w.serving_endpoints.query(
    name=endpoint_name, 
    dataframe_records=dataframe_records
)
print("Churn predictions:", response.predictions)
```

## A/B Testing and Traffic Management

### 1. Multi-Version Deployment

**Best Practice**: Deploy multiple model versions with traffic splitting for A/B testing.

```python
# Example: 80% Production, 20% Champion for testing
endpoint_config_ab = {
    "served_models": [
        {
            "model_name": model_name,
            "model_version": production_version,
            "scale_to_zero_enabled": True,
            "workload_size": "Small",
        },
        {
            "model_name": model_name,
            "model_version": champion_version,
            "scale_to_zero_enabled": True,
            "workload_size": "Small",
        }
    ],
    "traffic_config": {
        "routes": [
            {
                "served_model_name": f"{served_model_name}-{production_version}",
                "traffic_percentage": 80
            },
            {
                "served_model_name": f"{served_model_name}-{champion_version}",
                "traffic_percentage": 20
            }
        ]
    }
}
```

### 2. Alias Strategy for Lifecycle Management

**Best Practice**: Use a three-alias system for comprehensive lifecycle management.

```python
# Three-tier alias system:
# - @Challenger: New model under validation
# - @Champion: Validated model ready for production
# - @Production: Live model serving production traffic

# Promotion workflow
def promote_model_lifecycle(model_name, new_version):
    """Complete model promotion workflow"""
    
    # Step 1: Register new model as Challenger
    client.set_registered_model_alias(
        name=model_name, alias="Challenger", version=new_version
    )
    
    # Step 2: After validation, promote to Champion
    client.set_registered_model_alias(
        name=model_name, alias="Champion", version=new_version
    )
    
    # Step 3: After A/B testing, promote to Production
    client.set_registered_model_alias(
        name=model_name, alias="Production", version=new_version
    )
```

## Model Monitoring and Drift Detection

### 1. Lakehouse Monitoring Setup

**Best Practice**: Implement comprehensive monitoring for inference tables.

```python
from databricks.sdk.service.catalog import (
    MonitorInferenceLog, MonitorInferenceLogProblemType, MonitorMetric, MonitorMetricType
)

# Create custom business metrics
expected_loss_metric = [
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

# Create monitor
monitor_info = w.quality_monitors.create(
    table_name=f"{catalog}.{db}.advanced_churn_inference_table",
    inference_log=MonitorInferenceLog(
        problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_CLASSIFICATION,
        prediction_col="prediction",
        timestamp_col="inference_timestamp",
        granularities=["1 day"],
        model_id_col="model_version",
        label_col="churn",
    ),
    assets_dir=f"{os.getcwd()}/monitoring",
    output_schema_name=f"{catalog}.{db}",
    baseline_table_name=f"{catalog}.{db}.advanced_churn_baseline",
    slicing_exprs=["senior_citizen='Yes'", "contract"],
    custom_metrics=expected_loss_metric
)
```

### 2. Drift Detection and Alerting

**Best Practice**: Implement automated drift detection with business-relevant thresholds.

```python
def detect_model_drift(catalog, db, metric="f1_score.macro", drift="js_distance"):
    """Detect model performance and data drift"""
    
    # Get monitor table names
    monitor_info = w.quality_monitors.get(
        table_name=f"{catalog}.{db}.advanced_churn_inference_table"
    )
    drift_table_name = monitor_info.drift_metrics_table_name
    profile_table_name = monitor_info.profile_metrics_table_name
    
    # Performance metrics query
    performance_metrics_df = spark.sql(f"""
    SELECT
        window.start as time,
        {metric} AS performance_metric,
        expected_loss,
        Model_Version AS `Model Id`
    FROM {profile_table_name}
    WHERE
        window.start >= "2024-06-01"
        AND log_type = "INPUT"
        AND column_name = ":table"
        AND slice_key is null
        AND slice_value is null
    ORDER BY window.start
    """)
    
    # Drift metrics query
    drift_metrics_df = spark.sql(f"""
    SELECT
        window.start AS time,
        column_name,
        {drift} AS drift_metric,
        Model_Version AS `Model Id`
    FROM {drift_table_name}
    WHERE
        column_name IN ('prediction', 'churn')
        AND window.start >= "2024-06-01"
        AND slice_key is null
        AND slice_value is null
        AND drift_type = "CONSECUTIVE"
    ORDER BY window.start
    """)
    
    return performance_metrics_df, drift_metrics_df
```

### 3. Violation Detection and Automated Actions

**Best Practice**: Define business-relevant thresholds and automate responses.

```python
def count_violations(performance_df, drift_df):
    """Count violations based on business thresholds"""
    
    # Business thresholds
    PERFORMANCE_THRESHOLD = 0.5
    EXPECTED_LOSS_THRESHOLD = 30
    DRIFT_THRESHOLD = 0.19
    
    # Join metrics
    all_metrics_df = performance_df
    if not drift_df.isEmpty():
        # Pivot drift metrics
        unstacked_drift_df = (
            drift_df.groupBy("time", "`Model Id`")
            .pivot("column_name")
            .agg(first("drift_metric"))
            .orderBy("time")
        )
        all_metrics_df = performance_df.join(
            unstacked_drift_df, on=["time", "Model Id"], how="inner"
        )
    
    # Count performance violations
    performance_violations = all_metrics_df.where(
        (col("performance_metric") < PERFORMANCE_THRESHOLD) & 
        (abs(col("expected_loss")) > EXPECTED_LOSS_THRESHOLD)
    ).count()
    
    # Count drift violations
    drift_violations = 0
    if not drift_df.isEmpty():
        drift_violations = all_metrics_df.where(
            (col("churn") > DRIFT_THRESHOLD) & 
            (col("prediction") > DRIFT_THRESHOLD)
        ).count()
    
    total_violations = performance_violations + drift_violations
    
    # Set task value for workflow branching
    dbutils.jobs.taskValues.set(
        key='all_violations_count', 
        value=total_violations
    )
    
    return total_violations
```

### 4. Automated Workflow Integration

**Best Practice**: Use Databricks Workflows with conditional logic for automated responses.

```python
# In drift detection notebook - set task values
dbutils.jobs.taskValues.set(key='all_violations_count', value=violation_count)

# In workflow configuration
workflow_config = {
    "tasks": [
        {
            "task_key": "Drift_detection",
            "notebook_task": {
                "notebook_path": "08_drift_detection"
            }
        },
        {
            "task_key": "Check_Violations",
            "depends_on": [{"task_key": "Drift_detection"}],
            "condition_task": {
                "op": "GREATER_THAN",
                "left": "{{tasks.Drift_detection.values.all_violations_count}}",
                "right": "0"
            }
        },
        {
            "task_key": "Model_training",
            "depends_on": [{"task_key": "Check_Violations", "outcome": "true"}],
            "notebook_task": {
                "notebook_path": "02_automl_champion"
            }
        }
    ]
}
```

## Production Deployment Checklist

### 1. Model Readiness
- [ ] Model validation passed (accuracy, business metrics)
- [ ] Model documentation complete
- [ ] Dependency requirements captured
- [ ] Feature lineage documented

### 2. Infrastructure Setup
- [ ] Online tables created and synced
- [ ] Serving endpoint deployed and tested
- [ ] Auto-scaling configured appropriately
- [ ] Monitoring tables created

### 3. Testing and Validation
- [ ] Batch inference tested
- [ ] Real-time endpoint tested
- [ ] A/B testing configuration validated
- [ ] Error handling verified

### 4. Monitoring and Alerting
- [ ] Lakehouse monitoring configured
- [ ] Custom business metrics defined
- [ ] Drift detection thresholds set
- [ ] Automated alert workflows configured

## Performance Optimization Tips

### 1. Batch Inference
- Use appropriate Spark cluster sizing for data volume
- Leverage Delta Lake caching for repeated reads
- Consider partitioning for large inference datasets
- Use broadcast joins for small feature tables

### 2. Real-Time Serving
- Choose appropriate workload size based on concurrency needs
- Enable scale-to-zero for cost optimization
- Use online tables for low-latency feature access
- Monitor endpoint metrics and adjust scaling

### 3. Feature Store
- Enable Change Data Feed for efficient online table updates
- Use appropriate primary keys and time-series keys
- Consider feature caching strategies
- Monitor feature freshness

## Security and Governance

### 1. Access Control
- Use Unity Catalog permissions for model access
- Implement role-based access for serving endpoints
- Secure online tables with appropriate permissions
- Audit model usage and access patterns

### 2. Model Governance
- Maintain clear alias semantics (Challenger/Champion/Production)
- Document model versions and changes
- Track model lineage and dependencies
- Implement approval workflows for production deployment

### 3. Data Privacy
- Ensure PII handling compliance in feature tables
- Implement data masking where required
- Monitor data access and usage
- Maintain audit trails for compliance

## Summary of Best Practices

### Deployment Strategy
1. **Gradual rollout**: Use Challenger → Champion → Production progression
2. **A/B testing**: Deploy multiple versions with traffic splitting
3. **Automated validation**: Implement comprehensive testing before promotion
4. **Dependency management**: Capture and install model requirements automatically

### Serving Architecture  
5. **Feature stores**: Use online tables for low-latency feature access
6. **Auto-scaling**: Enable scale-to-zero and appropriate workload sizing
7. **Multi-version support**: Deploy multiple model versions simultaneously
8. **Error handling**: Implement robust error handling and fallback strategies

### Monitoring and Maintenance
9. **Comprehensive monitoring**: Track performance, drift, and business metrics
10. **Automated alerting**: Set business-relevant thresholds and automated responses
11. **Drift detection**: Monitor both data and prediction drift continuously
12. **Workflow automation**: Use conditional logic for automated retraining

### Operational Excellence
13. **Infrastructure as code**: Use APIs for reproducible deployments
14. **Cost optimization**: Leverage auto-scaling and scale-to-zero features
15. **Security**: Implement proper access controls and governance
16. **Documentation**: Maintain comprehensive model and deployment documentation

This framework provides a production-ready deployment and serving strategy for ML models on Databricks, ensuring reliability, scalability, and maintainability in production environments.