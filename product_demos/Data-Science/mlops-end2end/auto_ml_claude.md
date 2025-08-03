# AutoML Best Practices for MLOps on Databricks

This guide captures AutoML best practices from the Databricks MLOps end-to-end demo, focusing on leveraging AutoML-generated code for production-ready ML pipelines, Feature Store integration, and MLOps automation.

## Overview

Databricks AutoML is a glass-box solution that accelerates ML development while maintaining full transparency and control. This project demonstrates two complementary approaches:
- **Quickstart**: Direct table AutoML with generated notebook customization
- **Advanced**: Feature Store integration with on-demand features and enhanced lineage

Both approaches leverage AutoML-generated patterns for preprocessing, hyperparameter optimization, and comprehensive model evaluation.

## Key Dependencies

```python
# Core AutoML packages
%pip install databricks-automl-runtime==0.2.21
%pip install lightgbm==4.5.0
%pip install shap==0.46.0
%pip install hyperopt
%pip install databricks-feature-engineering==0.12.1  # Advanced only
```

## Data Preparation for AutoML

### 1. Data Quality and Metadata Enhancement

**Best Practice**: Enhance data with semantic metadata to improve AutoML feature understanding.

```python
def enhance_data_for_automl(df, target_col, categorical_cols=None, numerical_cols=None):
    """Add semantic metadata to improve AutoML performance"""
    
    enhanced_df = df
    
    # Add target metadata
    enhanced_df = enhanced_df.withMetadata(
        target_col, 
        {"spark.contentAnnotation.semanticType": "categorical"}
    )
    
    # Add categorical metadata
    if categorical_cols:
        for col in categorical_cols:
            if col in df.columns:
                enhanced_df = enhanced_df.withMetadata(
                    col, 
                    {"spark.contentAnnotation.semanticType": "categorical"}
                )
    
    # Add numerical metadata  
    if numerical_cols:
        for col in numerical_cols:
            if col in df.columns:
                enhanced_df = enhanced_df.withMetadata(
                    col, 
                    {"spark.contentAnnotation.semanticType": "numeric"}
                )
    
    return enhanced_df

# Usage
enhanced_df = enhance_data_for_automl(
    df=churn_df,
    target_col="churn",
    categorical_cols=["gender", "contract", "payment_method"],
    numerical_cols=["monthly_charges", "total_charges", "tenure"]
)
```

### 2. Train-Validation-Test Split Preparation

**Best Practice**: Use consistent splitting strategies that AutoML can leverage.

```python
# Quickstart approach - use existing split column
def prepare_automl_data_with_splits(table_name, target_col, split_col="split"):
    """Prepare data with existing splits for AutoML"""
    
    dataset = spark.table(table_name)
    
    # Validate split distribution
    split_counts = dataset.groupBy(split_col).count().collect()
    print("Split distribution:")
    for row in split_counts:
        print(f"  {row[split_col]}: {row['count']}")
    
    return dataset

# Advanced approach - create stratified splits
def create_stratified_splits(df, target_col, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """Create stratified splits for AutoML"""
    
    splits = df.randomSplit([train_ratio, val_ratio, test_ratio], seed=42)
    
    train_df = splits[0].withColumn("split", F.lit("train"))
    val_df = splits[1].withColumn("split", F.lit("validate")) 
    test_df = splits[2].withColumn("split", F.lit("test"))
    
    return train_df.union(val_df).union(test_df)
```

## AutoML Generated Code Patterns

### 1. Column Selection and Preprocessing

**Best Practice**: Leverage AutoML-generated preprocessing pipelines for production consistency.

```python
from databricks.automl_runtime.sklearn.column_selector import ColumnSelector
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from databricks.automl_runtime.sklearn import OneHotEncoder as DBOneHotEncoder

class AutoMLPreprocessingPatterns:
    """Reusable preprocessing patterns from AutoML generated notebooks"""
    
    def __init__(self, supported_cols):
        self.supported_cols = supported_cols
        self.col_selector = ColumnSelector(supported_cols)
    
    def create_comprehensive_preprocessor(self):
        """Create preprocessing pipeline following AutoML patterns"""
        
        # Boolean columns preprocessing
        bool_pipeline = Pipeline(steps=[
            ("cast_type", FunctionTransformer(lambda df: df.astype(object))),
            ("imputers", ColumnTransformer([], remainder="passthrough")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
        ])
        
        # Numerical columns preprocessing
        numerical_pipeline = Pipeline(steps=[
            ("converter", FunctionTransformer(lambda df: df.apply(pd.to_numeric, errors='coerce'))),
            ("imputers", ColumnTransformer([
                ("impute_mean", SimpleImputer(), ["monthly_charges", "total_charges", "tenure", "num_optional_services"])
            ])),
            ("standardizer", StandardScaler()),
        ])
        
        # Categorical columns preprocessing
        categorical_pipeline = Pipeline(steps=[
            ("imputers", ColumnTransformer([], remainder="passthrough")),
            ("one_hot_encoder", DBOneHotEncoder(handle_unknown="indicator")),
        ])
        
        # Combine all transformers
        transformers = [
            ("boolean", bool_pipeline, ["gender", "phone_service", "dependents", "senior_citizen", "paperless_billing", "partner"]),
            ("numerical", numerical_pipeline, ["monthly_charges", "total_charges", "tenure", "num_optional_services"]),
            ("categorical", categorical_pipeline, ["contract", "device_protection", "internet_service", "multiple_lines", "online_backup", "online_security", "payment_method", "streaming_movies", "streaming_tv", "tech_support"])
        ]
        
        preprocessor = ColumnTransformer(transformers, remainder="passthrough", sparse_threshold=1)
        return preprocessor

# Usage
supported_cols = ["online_backup", "internet_service", "payment_method", "multiple_lines", 
                  "paperless_billing", "partner", "tech_support", "tenure", "contract", 
                  "phone_service", "streaming_movies", "dependents", "senior_citizen", 
                  "num_optional_services", "device_protection", "monthly_charges", 
                  "total_charges", "streaming_tv", "gender", "online_security"]

automl_preprocessor = AutoMLPreprocessingPatterns(supported_cols)
preprocessor = automl_preprocessor.create_comprehensive_preprocessor()
```

### 2. Data Split Handling

**Best Practice**: Handle AutoML-generated split columns systematically.

```python
def handle_automl_data_splits(df_loaded, target_col):
    """Handle data splits as generated by AutoML notebooks"""
    
    # Find split columns (AutoML generated or existing)
    split_cols = [c for c in df_loaded.columns if c.startswith('_automl_split_col') or c == 'split']
    
    if not split_cols:
        # Create manual stratified split if no split column exists
        from sklearn.model_selection import train_test_split
        
        X = df_loaded.drop([target_col], axis=1)
        y = df_loaded[target_col]
        
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.4, stratify=y, random_state=42
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    # Use existing split column
    split_col = split_cols[0]
    
    # Split data by split column values
    split_train_df = df_loaded.loc[df_loaded[split_col] == "train"]
    split_val_df = df_loaded.loc[df_loaded[split_col] == "validate"]
    split_test_df = df_loaded.loc[df_loaded[split_col] == "test"]
    
    # Extract features and targets, dropping all split-related columns
    X_train = split_train_df.drop([target_col] + split_cols, errors='ignore', axis=1)
    y_train = split_train_df[target_col]
    
    X_val = split_val_df.drop([target_col] + split_cols, errors='ignore', axis=1)
    y_val = split_val_df[target_col]
    
    X_test = split_test_df.drop([target_col] + split_cols, errors='ignore', axis=1)
    y_test = split_test_df[target_col]
    
    # Handle edge case where validation set is empty (demo compatibility)
    if len(X_val) == 0:
        X_val, y_val = X_test, y_test
    
    return X_train, X_val, X_test, y_train, y_val, y_test
```

## Hyperparameter Optimization Patterns

### 1. AutoML-Style Objective Function

**Best Practice**: Use comprehensive evaluation across all data splits with proper MLflow integration.

```python
from hyperopt import hp, tpe, fmin, STATUS_OK, Trials
from mlflow.models.signature import infer_signature
from mlflow.models import Model
from mlflow.pyfunc import PyFuncModel
from mlflow import pyfunc

def create_automl_objective_function(X_train, X_val, X_test, y_train, y_val, y_test, 
                                   col_selector, preprocessor, target_col="churn"):
    """Create comprehensive objective function following AutoML patterns"""
    
    def objective(params):
        with mlflow.start_run(run_name=params.get("run_name", "automl_best_run")) as mlflow_run:
            # Initialize model with hyperparameters
            lgbmc_classifier = LGBMClassifier(**{k: v for k, v in params.items() if k != "run_name"})
            
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
            
            # Create PyFunc model for evaluation
            mlflow_model = Model()
            pyfunc.add_to_model(mlflow_model, loader_module="mlflow.sklearn")
            pyfunc_model = PyFuncModel(model_meta=mlflow_model, model_impl=model)
            
            # Comprehensive evaluation across all splits
            for split_name, (X_split, y_split) in [
                ("training", (X_train, y_train)),
                ("val", (X_val, y_val)),
                ("test", (X_test, y_test))
            ]:
                eval_result = mlflow.evaluate(
                    model=pyfunc_model,
                    data=X_split.assign(**{str(target_col): y_split}),
                    targets=target_col,
                    model_type="classifier",
                    evaluator_config={
                        "log_model_explainability": False,
                        "metric_prefix": f"{split_name}_",
                        "pos_label": "Yes"
                    }
                )
            
            # Return optimization target (negative F1 for minimization)
            val_metrics = eval_result.metrics
            loss = -val_metrics.get("val_f1_score", 0)
            
            return {
                "loss": loss,
                "status": STATUS_OK,
                "val_metrics": {k.replace("val_", ""): v for k, v in val_metrics.items() if k.startswith("val_")},
                "model": model,
                "run": mlflow_run,
            }
    
    return objective

# Usage with AutoML-optimized hyperparameters
space = {
    "run_name": "mlops_best_run",
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

# Create and run objective function
objective = create_automl_objective_function(
    X_train, X_val, X_test, y_train, y_val, y_test, 
    col_selector, preprocessor
)

trials = Trials()
best_result = fmin(objective, space=space, algo=tpe.suggest, max_evals=1, trials=trials)
```

### 2. Early Stopping Integration

**Best Practice**: Use early stopping for efficient training, especially with validation sets.

```python
# Advanced pattern with early stopping (from advanced track)
def create_advanced_objective_function(X_train, X_val, X_test, y_train, y_val, y_test,
                                     col_selector, preprocessor, training_set_specs=None):
    """Advanced objective function with early stopping and Feature Store integration"""
    
    # Create separate pipeline for validation preprocessing (early stopping requirement)
    mlflow.sklearn.autolog(disable=True)
    pipeline_val = Pipeline([
        ("column_selector", col_selector),
        ("preprocessor", preprocessor),
    ])
    pipeline_val.fit(X_train, y_train)
    X_val_processed = pipeline_val.transform(X_val)
    
    def objective(params):
        with mlflow.start_run(run_name="mlops_best_run") as mlflow_run:
            lgbmc_classifier = LGBMClassifier(**params)
            
            model = Pipeline([
                ("column_selector", col_selector),
                ("preprocessor", preprocessor),
                ("classifier", lgbmc_classifier),
            ])
            
            # Enable autologging
            mlflow.sklearn.autolog(
                log_input_examples=True,
                log_models=False,
                silent=True
            )
            
            # Train with early stopping
            model.fit(
                X_train, y_train,
                classifier__callbacks=[
                    lightgbm.early_stopping(5),
                    lightgbm.log_evaluation(0)
                ],
                classifier__eval_set=[(X_val_processed, y_val)]
            )
            
            # Advanced: Use Feature Engineering client for enhanced logging
            if training_set_specs is not None:
                from databricks.feature_engineering import FeatureEngineeringClient
                fe = FeatureEngineeringClient()
                
                # Infer output schema
                try:
                    from mlflow.types.utils import _infer_schema
                    output_schema = _infer_schema(y_train)
                except Exception:
                    output_schema = None
                
                # Log model with feature specifications
                fe.log_model(
                    model=model,
                    artifact_path="model",
                    flavor=mlflow.sklearn,
                    training_set=training_set_specs,
                    output_schema=output_schema,
                    extra_pip_requirements=["databricks-feature-lookup>=1.5.0"]
                )
            else:
                # Standard sklearn logging
                signature = infer_signature(X_train, y_train)
                mlflow.sklearn.log_model(
                    model, "sklearn_model",
                    input_example=X_train.iloc[0].to_dict(),
                    signature=signature
                )
            
            # Comprehensive evaluation (same as before)
            # ... evaluation code ...
            
            return {
                "loss": loss,
                "status": STATUS_OK,
                "model": model,
                "run": mlflow_run,
            }
    
    return objective
```

## Feature Store Integration

### 1. AutoML with Feature Lookups

**Best Practice**: Use Feature Store for complex feature engineering and lineage tracking.

```python
from databricks.feature_engineering import FeatureEngineeringClient
from databricks.feature_store import FeatureLookup, FeatureFunction

def setup_automl_with_feature_store(catalog, db, label_table_name, target_col="churn"):
    """Setup AutoML training with Feature Store integration"""
    
    fe = FeatureEngineeringClient()
    
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
    
    # Load labels
    labels_df = spark.read.table(f"{catalog}.{db}.{label_table_name}")
    
    # Create training set specifications
    training_set_specs = fe.create_training_set(
        df=labels_df,
        label=target_col,
        feature_lookups=features,
        exclude_columns=["customer_id", "transaction_ts", "split"]
    )
    
    # Load training data
    df_loaded = training_set_specs.load_df().toPandas()
    
    print(f"Training set shape: {df_loaded.shape}")
    print(f"Features: {[col for col in df_loaded.columns if col != target_col]}")
    
    return df_loaded, training_set_specs

# Usage
df_loaded, training_set_specs = setup_automl_with_feature_store(
    catalog, db, "advanced_churn_label_table"
)

# Use in training with enhanced lineage
X_train, X_val, X_test, y_train, y_val, y_test = train_test_split(
    df_loaded.drop(target_col, axis=1), 
    df_loaded[target_col], 
    test_size=0.4, 
    stratify=df_loaded[target_col], 
    random_state=42
)
```

### 2. On-Demand Feature Functions

**Best Practice**: Create reusable feature functions for real-time and batch inference.

```python
# Create feature function in Unity Catalog
spark.sql(f"""
CREATE OR REPLACE FUNCTION {catalog}.{db}.avg_price_increase(
    monthly_charges_in DOUBLE, 
    tenure_in DOUBLE, 
    total_charges_in DOUBLE
)
RETURNS FLOAT
LANGUAGE PYTHON
COMMENT "[Feature Function] Calculate potential average price increase"
AS $$
if tenure_in > 0:
    return monthly_charges_in - total_charges_in/tenure_in
else:
    return 0
$$
""")

# Verify function works
display(spark.sql(f"DESCRIBE FUNCTION {catalog}.{db}.avg_price_increase"))
```

## Model Explainability Integration

### 1. SHAP Integration Following AutoML Patterns

**Best Practice**: Generate SHAP explanations with proper error handling and artifact logging.

```python
def generate_automl_shap_explanations(model, X_train, X_val, shap_enabled=True, sample_size=100):
    """Generate SHAP explanations following AutoML patterns"""
    
    if not shap_enabled:
        print("SHAP explanations disabled")
        return None
    
    try:
        from shap import KernelExplainer, summary_plot
        import warnings
        warnings.filterwarnings('ignore')
        
        # Handle missing values (SHAP requirement)
        mode = X_train.mode().iloc[0]
        train_sample = X_train.sample(n=min(sample_size, X_train.shape[0]), random_state=790671489).fillna(mode)
        val_sample = X_val.sample(n=min(sample_size, X_val.shape[0]), random_state=790671489).fillna(mode)
        
        # Create prediction function
        predict = lambda x: model.predict_proba(pd.DataFrame(x, columns=X_train.columns))
        
        # Create SHAP explainer
        explainer = KernelExplainer(predict, train_sample, link="logit")
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(val_sample, l1_reg=False, nsamples=100)
        
        # Generate and save summary plot
        summary_plot(shap_values, val_sample, class_names=model.classes_)
        
        print("✅ SHAP explanations generated successfully")
        return shap_values
        
    except Exception as e:
        print(f"❌ SHAP explanation failed: {e}")
        return None

# Usage
shap_values = generate_automl_shap_explanations(
    model=best_model,
    X_train=X_train,
    X_val=X_val,
    shap_enabled=True,
    sample_size=100
)
```

### 2. Model Artifacts and Visualizations

**Best Practice**: Save comprehensive evaluation artifacts following AutoML patterns.

```python
def create_automl_evaluation_artifacts(mlflow_run, eval_temp_dir="/tmp/automl_eval"):
    """Download and display AutoML evaluation artifacts"""
    
    import uuid
    from IPython.display import Image
    import os
    
    # Create temp directory
    eval_temp_dir = os.path.join(os.environ["SPARK_LOCAL_DIRS"], "tmp", str(uuid.uuid4())[:8])
    os.makedirs(eval_temp_dir, exist_ok=True)
    
    # Download artifacts
    eval_path = mlflow.artifacts.download_artifacts(
        run_id=mlflow_run.info.run_id, 
        dst_path=eval_temp_dir
    )
    
    # Display key evaluation plots
    plots = {
        "Confusion Matrix": "val_confusion_matrix.png",
        "ROC Curve": "val_roc_curve_plot.png", 
        "Precision-Recall Curve": "val_precision_recall_curve_plot.png"
    }
    
    for plot_name, filename in plots.items():
        plot_path = os.path.join(eval_path, filename)
        if os.path.exists(plot_path):
            print(f"\n### {plot_name}")
            display(Image(filename=plot_path))
        else:
            print(f"⚠️ {plot_name} not found at {plot_path}")
    
    return eval_path

# Usage after training
eval_path = create_automl_evaluation_artifacts(mlflow_run)
```

## Data Lineage and Tracking

### 1. Comprehensive Data Lineage

**Best Practice**: Capture upstream data lineage for root cause analysis.

```python
def setup_comprehensive_data_lineage(catalog, db, table_name, run_context="training-input"):
    """Setup comprehensive data lineage tracking"""
    
    # Get latest table version for lineage
    latest_table_version = max(
        spark.sql(f"describe history {catalog}.{db}.{table_name}").toPandas()["version"]
    )
    
    # Load dataset object from Unity Catalog
    src_dataset = mlflow.data.load_delta(
        table_name=f"{catalog}.{db}.{table_name}", 
        version=str(latest_table_version)
    )
    
    # Log lineage during training
    mlflow.log_input(src_dataset, context=run_context)
    
    print(f"✅ Data lineage captured for {table_name} v{latest_table_version}")
    return src_dataset

# Usage in training loop
src_dataset = setup_comprehensive_data_lineage(catalog, db, "mlops_churn_training")

# Log in training run
with mlflow.start_run() as run:
    # ... training code ...
    mlflow.log_input(src_dataset, context="training-input")
```

### 2. Model Performance Tracking

**Best Practice**: Track comprehensive model performance across all data splits.

```python
def log_comprehensive_model_metrics(model, X_train, X_val, X_test, y_train, y_val, y_test, target_col):
    """Log comprehensive metrics across all data splits"""
    
    # Create PyFunc model for evaluation
    mlflow_model = Model()
    pyfunc.add_to_model(mlflow_model, loader_module="mlflow.sklearn")
    pyfunc_model = PyFuncModel(model_meta=mlflow_model, model_impl=model)
    
    # Evaluate on all splits
    metrics_summary = {}
    
    for split_name, (X_split, y_split) in [
        ("training", (X_train, y_train)),
        ("validation", (X_val, y_val)),
        ("test", (X_test, y_test))
    ]:
        eval_result = mlflow.evaluate(
            model=pyfunc_model,
            data=X_split.assign(**{str(target_col): y_split}),
            targets=target_col,
            model_type="classifier",
            evaluator_config={
                "log_model_explainability": False,
                "metric_prefix": f"{split_name}_",
                "pos_label": "Yes"
            }
        )
        
        # Store metrics for summary
        split_metrics = {k.replace(f"{split_name}_", ""): v 
                        for k, v in eval_result.metrics.items() 
                        if k.startswith(f"{split_name}_")}
        metrics_summary[split_name] = split_metrics
    
    # Create metrics comparison DataFrame
    metrics_df = pd.DataFrame(metrics_summary).round(4)
    
    # Log metrics table as artifact
    metrics_df.to_csv("/tmp/metrics_summary.csv", index=True)
    mlflow.log_artifact("/tmp/metrics_summary.csv", "evaluation")
    
    return metrics_df

# Usage
metrics_summary = log_comprehensive_model_metrics(
    model, X_train, X_val, X_test, y_train, y_val, y_test, target_col
)
display(metrics_summary)
```

## Production Integration Patterns

### 1. Model Registration with AutoML Metadata

**Best Practice**: Register models with comprehensive metadata from AutoML runs.

```python
def register_automl_model(run_id, model_name, description=None, tags=None):
    """Register AutoML model with comprehensive metadata"""
    
    from mlflow import MlflowClient
    
    client = MlflowClient()
    
    # Register model
    model_details = mlflow.register_model(
        f"runs:/{run_id}/sklearn_model", 
        model_name
    )
    
    # Get run metrics for documentation
    run = mlflow.get_run(run_id)
    metrics = run.data.metrics
    
    # Create comprehensive description
    if description is None:
        description = f"""AutoML generated churn prediction model.
        
Performance Metrics:
- F1 Score: {metrics.get('test_f1_score', 'N/A'):.4f}
- Precision: {metrics.get('test_precision', 'N/A'):.4f}
- Recall: {metrics.get('test_recall', 'N/A'):.4f}
- ROC AUC: {metrics.get('test_roc_auc', 'N/A'):.4f}

Generated using Databricks AutoML with LightGBM classifier.
Includes comprehensive preprocessing pipeline and data lineage tracking."""
    
    # Update model description
    client.update_registered_model(
        name=model_details.name,
        description=description
    )
    
    # Add performance tags
    default_tags = {
        "f1_score": f"{metrics.get('test_f1_score', 0):.4f}",
        "model_type": "automl_generated",
        "algorithm": "lightgbm",
        "preprocessing": "comprehensive"
    }
    
    if tags:
        default_tags.update(tags)
    
    for key, value in default_tags.items():
        client.set_model_version_tag(
            name=model_details.name,
            version=model_details.version,
            key=key,
            value=str(value)
        )
    
    # Set initial alias
    client.set_registered_model_alias(
        name=model_details.name,
        alias="Challenger",
        version=model_details.version
    )
    
    print(f"✅ Model registered: {model_details.name} v{model_details.version}")
    print(f"📊 F1 Score: {metrics.get('test_f1_score', 'N/A'):.4f}")
    
    return model_details

# Usage
model_details = register_automl_model(
    run_id=mlflow_run.info.run_id,
    model_name=f"{catalog}.{db}.automl_churn_model",
    tags={"business_unit": "customer_retention", "data_source": "crm_system"}
)
```

### 2. Automated Model Validation

**Best Practice**: Implement systematic validation before model promotion.

```python
def validate_automl_model(model_name, model_alias="Challenger", validation_thresholds=None):
    """Comprehensive validation for AutoML models"""
    
    default_thresholds = {
        "f1_score": 0.70,
        "precision": 0.65,
        "recall": 0.65,
        "roc_auc": 0.75
    }
    
    thresholds = validation_thresholds or default_thresholds
    
    client = MlflowClient()
    model_details = client.get_model_version_by_alias(model_name, model_alias)
    
    # Get run metrics
    run = mlflow.get_run(model_details.run_id)
    metrics = run.data.metrics
    
    validation_results = {
        "passed": True,
        "checks": {},
        "model_version": model_details.version
    }
    
    # Performance validation
    for metric, threshold in thresholds.items():
        actual_value = metrics.get(f"test_{metric}", 0)
        passed = actual_value >= threshold
        
        validation_results["checks"][metric] = {
            "actual": actual_value,
            "threshold": threshold,
            "passed": passed
        }
        
        if not passed:
            validation_results["passed"] = False
    
    # Add validation tags
    client.set_model_version_tag(
        name=model_name,
        version=str(model_details.version),
        key="validation_passed",
        value=str(validation_results["passed"])
    )
    
    # Log validation summary
    validation_score = sum(1 for check in validation_results["checks"].values() if check["passed"])
    total_checks = len(validation_results["checks"])
    
    print(f"📋 Validation Results: {validation_score}/{total_checks} checks passed")
    
    if validation_results["passed"]:
        print("✅ Model ready for promotion to Champion")
    else:
        print("❌ Model validation failed - review performance before promotion")
        for metric, check in validation_results["checks"].items():
            if not check["passed"]:
                print(f"  - {metric}: {check['actual']:.4f} < {check['threshold']:.4f}")
    
    return validation_results

# Usage
validation_results = validate_automl_model(
    model_name=f"{catalog}.{db}.automl_churn_model",
    validation_thresholds={"f1_score": 0.75, "precision": 0.70}
)
```

## Summary of Best Practices

### Data Preparation
1. **Semantic metadata**: Add proper type annotations for better AutoML feature understanding
2. **Split strategies**: Use consistent train/validation/test splits with proper stratification
3. **Data quality**: Validate data quality and handle missing values systematically
4. **Feature selection**: Remove high-cardinality and high-missing features before AutoML

### Code Generation and Reuse
5. **Preprocessing pipelines**: Leverage AutoML-generated preprocessing for production consistency
6. **Column selection**: Use AutoML-generated column selectors for supported features
7. **Data split handling**: Systematically handle AutoML-generated split columns
8. **Pipeline patterns**: Reuse complete sklearn pipeline patterns from AutoML

### Training and Optimization
9. **Hyperparameter spaces**: Start with AutoML-optimized hyperparameters for baseline
10. **Early stopping**: Use validation sets for efficient training with early stopping
11. **Comprehensive evaluation**: Evaluate across all data splits with consistent metrics
12. **MLflow integration**: Use proper MLflow logging with signatures and artifacts

### Feature Store Integration
13. **Feature lookups**: Use Feature Store for complex feature engineering and lineage
14. **On-demand features**: Create reusable Unity Catalog functions for real-time features
15. **Training specifications**: Capture feature specifications for inference-time lookup
16. **Enhanced logging**: Use Feature Engineering client for model logging with lineage

### Model Explainability
17. **SHAP integration**: Generate explanations following AutoML patterns with error handling
18. **Artifact management**: Save and organize evaluation plots and explanations
19. **Sample handling**: Properly handle missing values and sampling for SHAP
20. **Visualization**: Create comprehensive evaluation visualizations

### Production Integration
21. **Data lineage**: Capture comprehensive upstream data lineage for troubleshooting
22. **Model registration**: Register with rich metadata including performance metrics
23. **Validation frameworks**: Implement systematic validation before model promotion
24. **Alias management**: Use proper alias progression (Challenger → Champion → Production)

This AutoML framework provides production-ready patterns for leveraging Databricks AutoML in MLOps pipelines, ensuring models meet both technical and business requirements while maintaining full transparency and control over the generated solutions.
