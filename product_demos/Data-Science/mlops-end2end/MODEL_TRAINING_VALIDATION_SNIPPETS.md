# MLOps Model Training & Validation Code Snippets

This document contains reusable code snippets for model training, registration, validation, and lifecycle management in MLOps workflows using Databricks and Unity Catalog. AutoML patterns are excluded and will be covered in a separate guide.

## Table of Contents

- [Model Registration & Lifecycle Management](#model-registration--lifecycle-management)
- [Best Run Selection](#best-run-selection)
- [Model Documentation & Metadata](#model-documentation--metadata)
- [Model Validation Framework](#model-validation-framework)
- [Business Metrics Evaluation](#business-metrics-evaluation)
- [Champion-Challenger Pattern](#champion-challenger-pattern)
- [Advanced Validation with Feature Store](#advanced-validation-with-feature-store)

---

## Model Registration & Lifecycle Management

### Basic Model Registration to Unity Catalog

```python
import mlflow
from mlflow import MlflowClient

# Initialize MLflow client
client = MlflowClient()

# Model configuration
model_name = f"{catalog}.{db}.mlops_churn"
run_id = "your_experiment_run_id"

# Register model to Unity Catalog
model_details = mlflow.register_model(
    f"runs:/{run_id}/sklearn_model",  # or /model for other frameworks
    model_name
)

print(f"Model registered: {model_details.name}, Version: {model_details.version}")
```

### Model Aliasing for Lifecycle Management

```python
# Set model aliases for different lifecycle stages
def set_model_alias(model_name, alias, version):
    """Set model alias (Champion, Challenger, Baseline, etc.)"""
    client.set_registered_model_alias(
        name=model_name,
        alias=alias,
        version=version
    )
    print(f"Model {model_name} version {version} set as {alias}")

# Usage examples
set_model_alias(model_name, "Challenger", model_details.version)
set_model_alias(model_name, "Champion", model_details.version)  # After validation
```

### Model Description and Documentation

```python
# Add comprehensive model description
client.update_registered_model(
    name=model_name,
    description="This model predicts customer churn using features from the training table. "
               "It is used to power the Telco Churn Dashboard in DB SQL."
)

# Add version-specific details
best_score = 0.85  # Your model's performance metric
version_desc = (
    f"This model version has an F1 validation metric of {round(best_score, 4)*100}%. "
    f"Follow the link to its training run for more details."
)

client.update_model_version(
    name=model_name,
    version=model_details.version,
    description=version_desc
)
```

---

## Best Run Selection

### Programmatic Best Run Selection

```python
def find_best_model_run(experiment_name_pattern=None, experiment_id=None, metric="test_f1_score"):
    """
    Find the best model run from MLflow experiments
    
    Args:
        experiment_name_pattern: Pattern to match experiment names
        experiment_id: Specific experiment ID (alternative to pattern)
        metric: Metric to use for selecting best run
    """
    
    if experiment_id is None:
        # Find experiment by pattern
        experiments = mlflow.search_experiments(
            filter_string=f"name LIKE '{experiment_name_pattern}%'",
            order_by=["last_update_time DESC"]
        )
        experiment_id = experiments[0].experiment_id
    
    # Search for best run
    best_model = mlflow.search_runs(
        experiment_ids=experiment_id,
        order_by=[f"metrics.{metric} DESC"],
        max_results=1,
        filter_string="status = 'FINISHED'"  # Add run_name filter if needed
    )
    
    return best_model

# Usage
xp_path = f"/Users/{current_user}/dbdemos_mlops"
best_model = find_best_model_run(f"{xp_path}/dbdemos_automl")
run_id = best_model.iloc[0]['run_id']
```

### Alternative: Load Experiment as Spark DataFrame

```python
# Load entire experiment for analysis
experiment_df = spark.read.format("mlflow-experiment").load(experiment_id)
display(experiment_df)

# Filter and analyze runs
best_runs = experiment_df.filter("status = 'FINISHED'") \
                        .orderBy(F.desc("metrics.test_f1_score")) \
                        .limit(5)
```

---

## Model Documentation & Metadata

### Model Tagging for Metadata Management

```python
def add_model_metadata(model_name, model_version, metrics_dict, tags_dict=None):
    """Add comprehensive metadata to model version"""
    
    # Add performance metrics as tags
    for metric_name, metric_value in metrics_dict.items():
        client.set_model_version_tag(
            name=model_name,
            version=model_version,
            key=metric_name,
            value=f"{round(metric_value, 4)}"
        )
    
    # Add custom tags
    if tags_dict:
        for tag_key, tag_value in tags_dict.items():
            client.set_model_version_tag(
                name=model_name,
                version=model_version,
                key=tag_key,
                value=str(tag_value)
            )

# Usage
metrics = {
    "f1_score": 0.85,
    "precision": 0.82,
    "recall": 0.88
}

tags = {
    "model_type": "classification",
    "training_framework": "sklearn",
    "data_version": "v2.1"
}

add_model_metadata(model_name, model_details.version, metrics, tags)
```

---

## Model Validation Framework

### Comprehensive Model Validation Pipeline

```python
class ModelValidator:
    """Comprehensive model validation framework"""
    
    def __init__(self, model_name, model_alias="Challenger"):
        self.model_name = model_name
        self.model_alias = model_alias
        self.client = MlflowClient()
        self.model_details = self.client.get_model_version_by_alias(model_name, model_alias)
        self.model_version = int(self.model_details.version)
        
    def validate_description(self, min_length=20):
        """Validate model has adequate description"""
        description = self.model_details.description
        
        if not description or len(description) <= min_length:
            has_description = False
            print(f"Please add detailed model description ({min_length}+ chars)")
        else:
            has_description = True
            
        self._tag_result("has_description", has_description)
        return has_description
    
    def validate_performance_metric(self, metric_name="test_f1_score", champion_alias="Champion"):
        """Compare performance against champion model"""
        challenger_run_id = self.model_details.run_id
        challenger_metric = mlflow.get_run(challenger_run_id).data.metrics[metric_name]
        
        try:
            champion_model = self.client.get_model_version_by_alias(self.model_name, champion_alias)
            champion_metric = mlflow.get_run(champion_model.run_id).data.metrics[metric_name]
            
            print(f'Champion {metric_name}: {champion_metric}. Challenger {metric_name}: {challenger_metric}.')
            metric_passed = challenger_metric >= champion_metric
        except:
            print("No Champion found. Accepting as first model.")
            metric_passed = True
            
        self._tag_result(f"metric_{metric_name}_passed", metric_passed)
        return metric_passed
    
    def validate_prediction_capability(self, test_data_table):
        """Test model can make predictions on production data"""
        try:
            # Load model and test prediction
            model_uri = f"models:/{self.model_name}@{self.model_alias}"
            model_udf = mlflow.pyfunc.spark_udf(spark, model_uri=model_uri)
            
            test_df = spark.table(test_data_table).limit(10)
            predictions = test_df.withColumn(
                'predictions', 
                model_udf(*model_udf.metadata.get_input_schema().input_names())
            )
            
            prediction_count = predictions.count()
            predicts = prediction_count > 0
            
        except Exception as e:
            print(f"Prediction test failed: {e}")
            predicts = False
            
        self._tag_result("predicts", predicts)
        return predicts
    
    def validate_artifacts(self):
        """Check if model has associated artifacts"""
        import os
        import tempfile
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                run_info = self.client.get_run(self.model_details.run_id)
                local_path = mlflow.artifacts.download_artifacts(
                    run_id=run_info.info.run_id, 
                    dst_path=temp_dir
                )
                
                has_artifacts = bool(os.listdir(local_path))
                
                if not has_artifacts:
                    print("No artifacts found. Consider adding visualizations or data profiling.")
                else:
                    print(f"Artifacts found: {os.listdir(local_path)}")
                    
        except Exception as e:
            print(f"Artifact validation failed: {e}")
            has_artifacts = False
            
        self._tag_result("has_artifacts", has_artifacts)
        return has_artifacts
    
    def _tag_result(self, key, value):
        """Helper to tag validation results"""
        self.client.set_model_version_tag(
            name=self.model_name,
            version=str(self.model_version),
            key=key,
            value=str(value)
        )
    
    def run_all_validations(self, test_data_table):
        """Run complete validation suite"""
        results = {
            "description": self.validate_description(),
            "performance": self.validate_performance_metric(),
            "prediction": self.validate_prediction_capability(test_data_table),
            "artifacts": self.validate_artifacts()
        }
        
        print(f"\nValidation Results for {self.model_name} v{self.model_version}:")
        for check, passed in results.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {check}: {status}")
            
        return all(results.values())

# Usage
validator = ModelValidator(f"{catalog}.{db}.mlops_churn", "Challenger")
all_passed = validator.run_all_validations("mlops_churn_training")
```

---

## Business Metrics Evaluation

### Champion vs Challenger Business Impact Analysis

```python
import pandas as pd
import plotly.express as px
from sklearn.metrics import confusion_matrix

def calculate_business_metrics(model_alias, validation_df, model_name):
    """Calculate business impact using confusion matrix and cost analysis"""
    
    # Business cost configuration
    cost_of_customer_churn = 2000  # Lost customer value
    cost_of_discount = 500         # Retention offer cost
    
    # Cost matrix
    costs = {
        'true_negative': 0,                           # No churn, no discount
        'false_negative': cost_of_customer_churn,     # Missed churn - lost customer
        'true_positive': cost_of_customer_churn - cost_of_discount,  # Prevented churn
        'false_positive': -cost_of_discount          # Unnecessary discount
    }
    
    # Make predictions
    model_uri = f"models:/{model_name}@{model_alias}"
    model_udf = mlflow.pyfunc.spark_udf(spark, model_uri=model_uri)
    
    predictions_df = validation_df.withColumn(
        'predictions',
        model_udf(*model_udf.metadata.get_input_schema().input_names())
    ).toPandas()
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(
        predictions_df['churn'], 
        predictions_df['predictions']
    ).ravel()
    
    # Calculate total business value
    total_value = (
        tn * costs['true_negative'] + 
        fp * costs['false_positive'] + 
        fn * costs['false_negative'] + 
        tp * costs['true_positive']
    )
    
    return {
        'model_alias': model_alias,
        'true_positive': tp,
        'true_negative': tn,
        'false_positive': fp,
        'false_negative': fn,
        'business_value': total_value
    }

def compare_model_business_impact(model_name, validation_table):
    """Compare business impact between Champion and Challenger models"""
    
    validation_df = spark.table(validation_table).filter("split='validate'")
    
    try:
        # Get metrics for both models
        champion_metrics = calculate_business_metrics("Champion", validation_df, model_name)
        challenger_metrics = calculate_business_metrics("Challenger", validation_df, model_name)
        
        # Create comparison visualization
        comparison_data = pd.DataFrame([champion_metrics, challenger_metrics])
        
        fig = px.bar(
            comparison_data, 
            x='model_alias', 
            y='business_value',
            color='model_alias',
            title='Champion vs Challenger - Business Value Comparison',
            labels={'business_value': 'Potential Revenue Impact ($)'}
        )
        
        fig.show()
        
        # Print detailed comparison
        value_diff = challenger_metrics['business_value'] - champion_metrics['business_value']
        print(f"Business Value Comparison:")
        print(f"  Champion: ${champion_metrics['business_value']:,.2f}")
        print(f"  Challenger: ${challenger_metrics['business_value']:,.2f}")
        print(f"  Improvement: ${value_diff:,.2f}")
        
        return challenger_metrics['business_value'] > champion_metrics['business_value']
        
    except Exception as e:
        print(f"Champion model not found. Evaluating Challenger only: {e}")
        challenger_metrics = calculate_business_metrics("Challenger", validation_df, model_name)
        print(f"Challenger Business Value: ${challenger_metrics['business_value']:,.2f}")
        return True  # Accept if no champion exists

# Usage
business_improvement = compare_model_business_impact(
    f"{catalog}.{db}.mlops_churn", 
    "mlops_churn_training"
)
```

---

## Champion-Challenger Pattern

### Complete Promotion Workflow

```python
def promote_challenger_to_champion(model_name, validation_table=None, 
                                 required_validations=None):
    """
    Complete workflow to promote Challenger to Champion after validation
    
    Args:
        model_name: Full model name in Unity Catalog
        validation_table: Table for business metrics evaluation
        required_validations: List of required validation tags
    """
    
    if required_validations is None:
        required_validations = ["has_description", "metric_f1_passed", "predicts"]
    
    client = MlflowClient()
    
    # Get Challenger model details
    try:
        challenger_model = client.get_model_version_by_alias(model_name, "Challenger")
        model_version = int(challenger_model.version)
    except Exception as e:
        raise Exception(f"No Challenger model found: {e}")
    
    # Check validation results
    model_tags = client.get_model_version(model_name, model_version).tags
    
    validation_results = {}
    for validation in required_validations:
        passed = model_tags.get(validation, "False") == "True"
        validation_results[validation] = passed
        print(f"{validation}: {'✓ PASS' if passed else '✗ FAIL'}")
    
    # Run business metrics evaluation if table provided
    if validation_table:
        business_improvement = compare_model_business_impact(model_name, validation_table)
        validation_results["business_improvement"] = business_improvement
    
    # Promote if all validations pass
    if all(validation_results.values()):
        print(f"\n🎉 Promoting model {model_name} v{model_version} to Champion!")
        
        client.set_registered_model_alias(
            name=model_name,
            alias="Champion",
            version=model_version
        )
        
        # Tag promotion timestamp
        from datetime import datetime
        client.set_model_version_tag(
            name=model_name,
            version=str(model_version),
            key="promoted_to_champion_at",
            value=datetime.now().isoformat()
        )
        
        print("✅ Model successfully promoted to Champion!")
        return True
        
    else:
        failed_validations = [k for k, v in validation_results.items() if not v]
        raise Exception(f"Model promotion failed. Failed validations: {failed_validations}")

# Usage
try:
    promote_challenger_to_champion(
        model_name=f"{catalog}.{db}.mlops_churn",
        validation_table="mlops_churn_training",
        required_validations=["has_description", "metric_f1_passed", "predicts"]
    )
except Exception as e:
    print(f"Promotion failed: {e}")
```

---

## Advanced Validation with Feature Store

### Feature Store Integration for Model Validation

```python
from databricks.feature_engineering import FeatureEngineeringClient

def validate_model_with_feature_store(model_name, model_alias, label_table):
    """
    Advanced model validation using Feature Store integration
    """
    fe = FeatureEngineeringClient()
    client = MlflowClient()
    
    model_details = client.get_model_version_by_alias(model_name, model_alias)
    model_version = int(model_details.version)
    
    try:
        # Read labels and perform feature lookup + scoring
        labels_df = spark.table(label_table)
        
        # Batch score with feature store integration
        features_w_preds = fe.score_batch(
            df=labels_df,
            model_uri=f"models:/{model_name}@{model_alias}",
            result_type=labels_df.schema["churn"].dataType  # Adjust label column
        )
        
        print(f"✅ Feature Store prediction successful")
        print(f"Predictions shape: {features_w_preds.count()} rows")
        
        # Tag successful prediction
        client.set_model_version_tag(
            name=model_name,
            version=str(model_version),
            key="feature_store_compatible",
            value="True"
        )
        
        return features_w_preds
        
    except Exception as e:
        print(f"❌ Feature Store prediction failed: {e}")
        
        client.set_model_version_tag(
            name=model_name,
            version=str(model_version),
            key="feature_store_compatible",
            value="False"
        )
        
        return None

# Usage
predictions = validate_model_with_feature_store(
    f"{catalog}.{db}.advanced_mlops_churn",
    "Challenger",
    "advanced_churn_label_table"
)
```

---

## Best Practices

### 1. Model Governance and Permissions

```python
def setup_model_governance(model_name, team_permissions=None):
    """Setup model governance and permissions"""
    
    if team_permissions is None:
        team_permissions = {
            "data_scientists": ["READ", "EDIT"],
            "ml_engineers": ["READ", "EDIT", "EXECUTE"],
            "business_users": ["READ"]
        }
    
    # Example governance setup (implementation depends on your access control)
    print(f"Setting up governance for {model_name}")
    for role, permissions in team_permissions.items():
        print(f"  {role}: {', '.join(permissions)}")
```

### 2. Model Lineage Tracking

```python
def track_model_lineage(model_name, training_data_tables, feature_functions=None):
    """Document model lineage information"""
    
    client = MlflowClient()
    
    # Add lineage information as model tags
    lineage_info = {
        "training_tables": ",".join(training_data_tables),
        "lineage_tracked": "True"
    }
    
    if feature_functions:
        lineage_info["feature_functions"] = ",".join(feature_functions)
    
    # This would be applied during model registration
    print("Model lineage information:")
    for key, value in lineage_info.items():
        print(f"  {key}: {value}")
```

### 3. Environment Management

```python
def validate_model_requirements(model_name, model_alias):
    """Download and validate model requirements"""
    from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
    
    # Download requirements
    requirements_path = ModelsArtifactRepository(
        f"models:/{model_name}@{model_alias}"
    ).download_artifacts(artifact_path="requirements.txt")
    
    # Install requirements for validation
    # %pip install --quiet -r $requirements_path
    
    print(f"Model requirements downloaded: {requirements_path}")
```

### 4. Cross-Environment Promotion

```python
def promote_across_environments(model_name, source_alias, target_catalog, target_alias):
    """
    Promote model across environments (Dev -> QA -> Prod)
    """
    client = MlflowClient()
    
    # Get source model
    source_model = client.get_model_version_by_alias(model_name, source_alias)
    
    # Create target model name
    target_model_name = model_name.replace(
        model_name.split(".")[0], 
        target_catalog
    )
    
    # Register to target environment
    target_model = mlflow.register_model(
        f"runs:/{source_model.run_id}/model",
        target_model_name
    )
    
    # Set target alias
    client.set_registered_model_alias(
        name=target_model_name,
        alias=target_alias,
        version=target_model.version
    )
    
    print(f"Model promoted from {model_name}@{source_alias} to {target_model_name}@{target_alias}")

# Usage
# promote_across_environments(
#     "dev_catalog.schema.model",
#     "Champion", 
#     "prod_catalog",
#     "Champion"
# )
```

---

## Usage Examples

### Complete Model Training to Production Pipeline

```python
# 1. Find best model from experiments
best_model = find_best_model_run(f"{xp_path}/experiment_name")
run_id = best_model.iloc[0]['run_id']

# 2. Register model
model_details = mlflow.register_model(f"runs:/{run_id}/model", model_name)

# 3. Add documentation and metadata
add_model_metadata(model_name, model_details.version, {"f1_score": 0.85})

# 4. Set as Challenger
set_model_alias(model_name, "Challenger", model_details.version)

# 5. Run validation suite
validator = ModelValidator(model_name, "Challenger")
validation_passed = validator.run_all_validations("test_data_table")

# 6. Promote if validation passes
if validation_passed:
    promote_challenger_to_champion(model_name, "validation_table")
```

This provides a complete framework for model training, validation, and lifecycle management in MLOps workflows using Databricks and Unity Catalog.