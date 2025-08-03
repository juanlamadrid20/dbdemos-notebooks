# Model Training and Validation Best Practices for MLOps on Databricks

This guide captures model training and validation best practices from the Databricks MLOps end-to-end demo, focusing on production-ready approaches for customer churn prediction using LightGBM and Unity Catalog.

## Overview

The project demonstrates two complementary approaches:
- **Quickstart**: Simple training with direct table data and MLflow tracking
- **Advanced**: Feature Store integration with on-demand features and enhanced lineage

Both approaches leverage AutoML-generated notebooks, comprehensive validation frameworks, and Unity Catalog for model governance.

## Key Dependencies

```python
# Core training packages
%pip install mlflow==2.22.0
%pip install databricks-automl-runtime==0.2.21
%pip install databricks-feature-engineering==0.12.1  # Advanced only
%pip install lightgbm==4.5.0
%pip install shap==0.46.0
%pip install hyperopt
```

## Model Training Patterns

### 1. Data Lineage and Input Tracking

**Best Practice**: Capture upstream data lineage to enable root cause analysis when model issues occur.

#### Quickstart Approach - Direct Table Lineage
```python
# Load dataset with version tracking
latest_table_version = max(
    spark.sql(f"describe history {catalog}.{db}.mlops_churn_training").toPandas()["version"]
)

src_dataset = mlflow.data.load_delta(
    table_name=f"{catalog}.{db}.mlops_churn_training", 
    version=str(latest_table_version)
)

# Log lineage during training
with mlflow.start_run() as mlflow_run:
    # ... model training code ...
    mlflow.log_input(src_dataset, context="training-input")
```

#### Advanced Approach - Feature Store Lineage
```python
from databricks.feature_engineering import FeatureEngineeringClient
from databricks.feature_store import FeatureLookup, FeatureFunction

# Define feature specifications
features = [
    FeatureLookup(
        table_name=f"{catalog}.{db}.advanced_churn_feature_table",
        lookup_key=["customer_id"],
        timestamp_lookup_key="transaction_ts"
    ),
    FeatureFunction(
        udf_name=f"{catalog}.{db}.avg_price_increase",
        input_bindings={
            "monthly_charges_in": "monthly_charges",
            "tenure_in": "tenure", 
            "total_charges_in": "total_charges"
        },
        output_name="avg_price_increase"
    )
]

fe = FeatureEngineeringClient()
training_set_specs = fe.create_training_set(
    df=labels_df,
    label="churn",
    feature_lookups=features,
    exclude_columns=["customer_id", "transaction_ts", 'split']
)
```

### 2. Column Selection and Preprocessing Pipeline

**Best Practice**: Use modular preprocessing pipelines that support different data types and handle missing values systematically.

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from databricks.automl_runtime.sklearn.column_selector import ColumnSelector

# Column selection for supported features
supported_cols = [
    "online_backup", "internet_service", "payment_method", "multiple_lines",
    "paperless_billing", "partner", "tech_support", "tenure", "contract",
    "phone_service", "streaming_movies", "dependents", "senior_citizen",
    "num_optional_services", "device_protection", "monthly_charges",
    "total_charges", "streaming_tv", "gender", "online_security"
]

col_selector = ColumnSelector(supported_cols)

# Boolean column preprocessing
bool_pipeline = Pipeline(steps=[
    ("cast_type", FunctionTransformer(lambda df: df.astype(object))),
    ("imputers", ColumnTransformer([], remainder="passthrough")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
])

# Numerical column preprocessing
numerical_pipeline = Pipeline(steps=[
    ("converter", FunctionTransformer(lambda df: df.apply(pd.to_numeric, errors='coerce'))),
    ("imputers", ColumnTransformer([
        ("impute_mean", SimpleImputer(), ["monthly_charges", "num_optional_services", "tenure", "total_charges"])
    ])),
    ("standardizer", StandardScaler()),
])

# Categorical column preprocessing
categorical_pipeline = Pipeline(steps=[
    ("imputers", ColumnTransformer([], remainder="passthrough")),
    ("one_hot_encoder", OneHotEncoder(handle_unknown="indicator")),
])

# Combine all transformers
transformers = [
    ("boolean", bool_pipeline, ["gender", "phone_service", "dependents", "senior_citizen", "paperless_billing", "partner"]),
    ("numerical", numerical_pipeline, ["monthly_charges", "total_charges", "tenure", "num_optional_services"]),
    ("onehot", categorical_pipeline, ["contract", "device_protection", "internet_service", "multiple_lines", "online_backup", "online_security", "payment_method", "streaming_movies", "streaming_tv", "tech_support"])
]

preprocessor = ColumnTransformer(transformers, remainder="passthrough", sparse_threshold=1)
```

### 3. Train-Validation-Test Split Strategies

#### Quickstart: Using Pre-defined Splits
```python
# Use existing split column from feature engineering
split_cols = [c for c in df_loaded.columns if c.startswith('_automl_split_col') or c == 'split']
split_col = split_cols[0]

# Separate data by splits
split_train_df = df_loaded.loc[df_loaded[split_col] == "train"]
split_val_df = df_loaded.loc[df_loaded[split_col] == "validate"] 
split_test_df = df_loaded.loc[df_loaded[split_col] == "test"]

# Extract features and target
X_train = split_train_df.drop([target_col] + split_cols, errors='ignore', axis=1)
y_train = split_train_df[target_col]
```

#### Advanced: Stratified Splitting
```python
from sklearn.model_selection import train_test_split

# Stratified split to maintain class balance
X_train, X_eval, y_train, y_eval = train_test_split(
    df_loaded.drop(label_col, axis=1), 
    df_loaded[label_col], 
    test_size=0.4, 
    stratify=df_loaded[label_col], 
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_eval, y_eval, 
    test_size=0.5, 
    stratify=y_eval, 
    random_state=42
)
```

### 4. Hyperparameter Optimization with Hyperopt

**Best Practice**: Use structured hyperparameter search with MLflow tracking.

```python
from hyperopt import hp, tpe, fmin, STATUS_OK, Trials
from mlflow.models import infer_signature

def objective(params):
    with mlflow.start_run(run_name="mlops_best_run") as mlflow_run:
        # Initialize model with hyperparameters
        lgbmc_classifier = LGBMClassifier(**params)
        
        # Create complete pipeline
        model = Pipeline([
            ("column_selector", col_selector),
            ("preprocessor", preprocessor),
            ("classifier", lgbmc_classifier),
        ])
        
        # Enable MLflow autologging
        mlflow.sklearn.autolog(log_models=False, silent=True)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Infer and log model signature
        signature = infer_signature(X_train, y_train)
        mlflow.sklearn.log_model(
            model, "sklearn_model", 
            input_example=X_train.iloc[0].to_dict(), 
            signature=signature
        )
        
        # Comprehensive evaluation across all splits
        training_eval_result = mlflow.evaluate(
            model=pyfunc_model,
            data=X_train.assign(**{str(target_col): y_train}),
            targets=target_col,
            model_type="classifier",
            evaluator_config={
                "log_model_explainability": False,
                "metric_prefix": "training_",
                "pos_label": "Yes"
            }
        )
        
        val_eval_result = mlflow.evaluate(
            model=pyfunc_model,
            data=X_val.assign(**{str(target_col): y_val}),
            targets=target_col,
            model_type="classifier",
            evaluator_config={
                "log_model_explainability": False,
                "metric_prefix": "val_",
                "pos_label": "Yes"
            }
        )
        
        test_eval_result = mlflow.evaluate(
            model=pyfunc_model,
            data=X_test.assign(**{str(target_col): y_test}),
            targets=target_col,
            model_type="classifier",
            evaluator_config={
                "log_model_explainability": False,
                "metric_prefix": "test_",
                "pos_label": "Yes"
            }
        )
        
        # Return optimization target
        loss = -val_eval_result.metrics["val_f1_score"]
        
        return {
            "loss": loss,
            "status": STATUS_OK,
            "val_metrics": val_eval_result.metrics,
            "test_metrics": test_eval_result.metrics,
            "model": model,
            "run": mlflow_run,
        }

# Configure hyperparameter space
space = {
    "colsample_bytree": 0.4120544919020157,
    "lambda_l1": 2.6616074270114995,
    "lambda_l2": 514.9224373768443,
    "learning_rate": 0.0678497372371143,
    "max_bin": 229,
    "max_depth": 8,
    "min_child_samples": 66,
    "n_estimators": 250,
    "num_leaves": 100,
    "path_smooth": 61.06596877554017,
    "subsample": 0.6965257092078714,
    "random_state": 42,
}

# Run optimization
trials = Trials()
fmin(objective, space=space, algo=tpe.suggest, max_evals=1, trials=trials)
```

### 5. Advanced Feature Store Integration

**Best Practice**: Use Feature Engineering client for model logging to capture feature specifications.

```python
# Advanced training with feature store logging
def objective(params):
    with mlflow.start_run(run_name="mlops_best_run") as mlflow_run:
        lgbmc_classifier = LGBMClassifier(**params)
        
        model = Pipeline([
            ("column_selector", col_selector),
            ("preprocessor", preprocessor),
            ("classifier", lgbmc_classifier),
        ])
        
        # Early stopping with validation set
        model.fit(
            X_train, y_train,
            classifier__callbacks=[
                lightgbm.early_stopping(5),
                lightgbm.log_evaluation(0)
            ],
            classifier__eval_set=[(X_val_processed, y_val)]
        )
        
        # Use Feature Engineering client for enhanced logging
        fe.log_model(
            model=model,
            artifact_path="model",
            flavor=mlflow.sklearn,
            training_set=training_set_specs,  # Captures feature lineage
            output_schema=output_schema,
            extra_pip_requirements=["databricks-feature-lookup>=1.5.0"]
        )
```

## Model Registration and Unity Catalog Integration

### 1. Best Run Selection and Registration

**Best Practice**: Programmatically select the best model based on evaluation metrics.

```python
# Find best model from experiments
experiment_id = mlflow.search_experiments(
    filter_string=f"name LIKE '{xp_path}/dbdemos_automl%'", 
    order_by=["last_update_time DESC"]
)[0].experiment_id

best_model = mlflow.search_runs(
    experiment_ids=experiment_id,
    order_by=["metrics.test_f1_score DESC"],
    max_results=1,
    filter_string="status = 'FINISHED' and run_name='mlops_best_run'"
)

# Register to Unity Catalog
run_id = best_model.iloc[0]['run_id']
model_details = mlflow.register_model(
    f"runs:/{run_id}/sklearn_model",  # or /model for advanced
    model_name
)
```

### 2. Model Documentation and Metadata

**Best Practice**: Add comprehensive documentation and tags for governance.

```python
from mlflow import MlflowClient

client = MlflowClient()

# Overall model description
client.update_registered_model(
    name=model_details.name,
    description="This model predicts customer churn using features from the training table. Used to power the Telco Churn Dashboard in DB SQL."
)

# Version-specific details
best_score = best_model['metrics.test_f1_score'].values[0]
version_desc = f"F1 validation metric: {round(best_score,4)*100}%. See training run for details."

client.update_model_version(
    name=model_details.name,
    version=model_details.version,
    description=version_desc
)

# Add performance tags
client.set_model_version_tag(
    name=model_details.name,
    version=model_details.version,
    key="f1_score",
    value=f"{round(best_score,4)}"
)
```

### 3. Model Lifecycle Management with Aliases

**Best Practice**: Use aliases to manage model lifecycle stages.

```python
# Set as Challenger for validation
client.set_registered_model_alias(
    name=model_name,
    alias="Challenger",
    version=model_details.version
)
```

## Model Validation Framework

### 1. Comprehensive Validation Checks

**Best Practice**: Implement multiple validation dimensions for production readiness.

```python
def validate_model(model_name, model_alias="Challenger"):
    """Comprehensive model validation framework"""
    
    client = MlflowClient()
    model_details = client.get_model_version_by_alias(model_name, model_alias)
    model_version = int(model_details.version)
    
    validation_results = {}
    
    # 1. Description Check
    if not model_details.description or len(model_details.description) <= 20:
        has_description = False
        print("Please add detailed model description (40+ chars)")
    else:
        has_description = True
    
    client.set_model_version_tag(
        name=model_name, 
        version=str(model_version), 
        key="has_description", 
        value=has_description
    )
    
    # 2. Performance Metric Validation
    model_run_id = model_details.run_id
    f1_score = mlflow.get_run(model_run_id).data.metrics['test_f1_score']
    
    try:
        # Compare against existing Champion
        champion_model = client.get_model_version_by_alias(model_name, "Champion")
        champion_f1 = mlflow.get_run(champion_model.run_id).data.metrics['test_f1_score']
        metric_f1_passed = f1_score >= champion_f1
        print(f'Champion F1: {champion_f1}, Challenger F1: {f1_score}')
    except:
        print("No Champion found. Accepting as first model.")
        metric_f1_passed = True
    
    client.set_model_version_tag(
        name=model_name, 
        version=str(model_version), 
        key="metric_f1_passed", 
        value=metric_f1_passed
    )
    
    return {
        "has_description": has_description,
        "metric_f1_passed": metric_f1_passed
    }
```

### 2. Production Inference Testing

**Best Practice**: Test model prediction capabilities on production-like data.

#### Quickstart Approach
```python
def predict_churn(validation_df, model_alias):
    """Test model prediction using Spark UDF"""
    model_uri = f"models:/{catalog}.{db}.mlops_churn@{model_alias}"
    model = mlflow.pyfunc.spark_udf(spark, model_uri=model_uri)
    
    return validation_df.withColumn(
        'predictions', 
        model(*model.metadata.get_input_schema().input_names())
    )

# Test on validation data
validation_df = spark.table('mlops_churn_training').filter("split='validate'")
predictions = predict_churn(validation_df, "Challenger")
```

#### Advanced Approach with Feature Store
```python
def test_prediction_capability(model_name, model_alias):
    """Test prediction using Feature Engineering client"""
    
    fe = FeatureEngineeringClient()
    model_uri = f"models:/{model_name}@{model_alias}"
    
    try:
        # Read labels and IDs
        labelsDF = spark.read.table("advanced_churn_label_table")
        
        # Batch score with feature lookup
        features_w_preds = fe.score_batch(
            df=labelsDF, 
            model_uri=model_uri, 
            result_type=labelsDF.schema[label_col].dataType
        )
        
        client.set_model_version_tag(
            name=model_name, 
            version=str(model_version), 
            key="predicts", 
            value=True
        )
        return True
        
    except Exception as e:
        print(f"Prediction failed: {e}")
        client.set_model_version_tag(
            name=model_name, 
            version=str(model_version), 
            key="predicts", 
            value=False
        )
        return False
```

### 3. Business Metrics Evaluation

**Best Practice**: Validate models against business KPIs, not just technical metrics.

```python
def evaluate_business_impact(validation_df, model_alias):
    """Evaluate model business value using cost-benefit analysis"""
    
    # Business parameters
    cost_of_customer_churn = 2000  # dollars
    cost_of_discount = 500         # dollars
    
    # Cost matrix
    cost_true_negative = 0                                    # correct non-churn prediction
    cost_false_negative = cost_of_customer_churn             # missed churn
    cost_true_positive = cost_of_customer_churn - cost_of_discount  # prevented churn
    cost_false_positive = -cost_of_discount                  # unnecessary discount
    
    # Get predictions
    model_predictions = predict_churn(validation_df, model_alias).toPandas()
    
    # Calculate confusion matrix
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(
        model_predictions['churn'], 
        model_predictions['predictions']
    ).ravel()
    
    # Calculate total business value
    total_value = (tn * cost_true_negative + 
                   fp * cost_false_positive + 
                   fn * cost_false_negative + 
                   tp * cost_true_positive)
    
    return total_value

# Compare business value
try:
    champion_value = evaluate_business_impact(validation_df, "Champion")
    challenger_value = evaluate_business_impact(validation_df, "Challenger")
    
    print(f"Champion business value: ${champion_value}")
    print(f"Challenger business value: ${challenger_value}")
    print(f"Improvement: ${challenger_value - champion_value}")
    
except:
    print("No Champion found for comparison")
```

### 4. Model Promotion Decision Framework

**Best Practice**: Implement systematic promotion criteria.

```python
def promote_model_if_ready(model_name, model_version):
    """Promote model to Champion if all validations pass"""
    
    results = client.get_model_version(model_name, model_version)
    
    # Check all validation criteria
    validations_passed = (
        results.tags.get("has_description") == "True" and
        results.tags.get("metric_f1_passed") == "True" and
        results.tags.get("predicts", "True") == "True"  # Default True for quickstart
    )
    
    if validations_passed:
        print(f'Promoting model {model_name} Version {model_version} to Champion!')
        client.set_registered_model_alias(
            name=model_name,
            alias="Champion",
            version=model_version
        )
        return True
    else:
        print("Model validation failed:")
        for key, value in results.tags.items():
            if value == "False":
                print(f"  - {key}: {value}")
        raise Exception("Model not ready for promotion")
```

## Model Explainability and Artifacts

### 1. SHAP Integration

**Best Practice**: Include model explainability for interpretable ML.

```python
def generate_shap_explanations(model, X_train, X_val, shap_enabled=True):
    """Generate SHAP explanations for model interpretability"""
    
    if shap_enabled:
        from shap import KernelExplainer, summary_plot
        
        # Handle missing values for SHAP
        mode = X_train.mode().iloc[0]
        train_sample = X_train.sample(n=min(100, X_train.shape[0]), random_state=42).fillna(mode)
        example = X_val.sample(n=min(100, X_val.shape[0]), random_state=42).fillna(mode)
        
        # Create SHAP explainer
        predict = lambda x: model.predict_proba(pd.DataFrame(x, columns=X_train.columns))
        explainer = KernelExplainer(predict, train_sample, link="logit")
        shap_values = explainer.shap_values(example, l1_reg=False, nsamples=100)
        
        # Generate summary plot
        summary_plot(shap_values, example, class_names=model.classes_)
        
        return shap_values
```

### 2. Model Artifacts and Visualizations

**Best Practice**: Log comprehensive evaluation artifacts.

```python
def validate_model_artifacts(model_name, model_version):
    """Check for required model artifacts"""
    
    run_info = client.get_run(run_id=model_details.run_id)
    
    # Download artifacts for inspection
    local_dir = "/tmp/model_artifacts"
    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_info.info.run_id, 
        dst_path=local_dir
    )
    
    if not os.listdir(local_path):
        has_artifacts = False
        print("Missing artifacts: Add visualizations or data profiling")
    else:
        has_artifacts = True
        print(f"Artifacts found: {os.listdir(local_path)}")
    
    client.set_model_version_tag(
        name=model_name, 
        version=str(model_version), 
        key="has_artifacts", 
        value=has_artifacts
    )
```

## Summary of Best Practices

### Training Phase
1. **Capture data lineage**: Use MLflow data logging and Feature Store specifications
2. **Structured preprocessing**: Modular pipelines for different data types
3. **Proper data splitting**: Stratified splits with deterministic seeds
4. **Hyperparameter optimization**: Systematic search with MLflow tracking
5. **Comprehensive evaluation**: Multi-dataset evaluation with business metrics

### Registration Phase
6. **Programmatic selection**: Automated best model identification
7. **Rich documentation**: Detailed descriptions and performance tags
8. **Lifecycle management**: Alias-based stage management
9. **Metadata tracking**: Tags for validation status and metrics

### Validation Phase
10. **Multi-dimensional validation**: Technical, business, and operational checks
11. **Production testing**: Real inference capability validation
12. **Business impact assessment**: Cost-benefit analysis beyond accuracy
13. **Systematic promotion**: Criteria-based advancement decisions
14. **Explainability**: SHAP integration for model interpretability
15. **Artifact management**: Comprehensive evaluation materials

### Model Governance
16. **Unity Catalog integration**: Centralized model registry with permissions
17. **Lineage tracking**: Feature-to-model traceability
18. **Version control**: Systematic versioning and aliasing
19. **Automated workflows**: MLOps pipeline automation capability
20. **Monitoring readiness**: Structured for downstream monitoring integration

This framework provides a comprehensive foundation for production-ready model training and validation on Databricks, ensuring models meet both technical and business requirements before deployment.