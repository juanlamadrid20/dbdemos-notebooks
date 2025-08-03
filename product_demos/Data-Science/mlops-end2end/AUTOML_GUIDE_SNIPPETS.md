# Databricks AutoML Complete Guide & Code Snippets

This comprehensive guide covers Databricks AutoML patterns for rapid ML model development, from UI workflows to advanced API integration with Feature Store and MLOps pipelines.

## Table of Contents

- [AutoML Overview & Benefits](#automl-overview--benefits)
- [AutoML UI Workflows](#automl-ui-workflows)
- [AutoML API Patterns](#automl-api-patterns)
- [Feature Store Integration](#feature-store-integration)
- [Generated Notebook Patterns](#generated-notebook-patterns)
- [Advanced Configuration](#advanced-configuration)
- [Model Evaluation & Explainability](#model-evaluation--explainability)
- [MLOps Pipeline Integration](#mlops-pipeline-integration)
- [Best Practices & Optimization](#best-practices--optimization)

---

## AutoML Overview & Benefits

### What is Databricks AutoML?

Databricks AutoML is a glass-box solution that automatically generates state-of-the-art models for classification, regression, and forecasting while providing full transparency and control over the generated code.

### Key Benefits

- **Accelerated Development**: Weeks of effort reduced to hours
- **Best Practices Built-in**: Automatic feature engineering, model selection, and hyperparameter tuning
- **Full Transparency**: Generated notebooks can be customized and reused
- **MLOps Integration**: Seamless integration with MLflow and Unity Catalog
- **Feature Store Support**: Automatic feature lookup and engineering

### Supported Problem Types

```python
# Classification
automl.classify(dataset=df, target_col="churn", timeout_minutes=60)

# Regression  
automl.regressor(dataset=df, target_col="price", timeout_minutes=60)

# Forecasting
automl.forecast(dataset=df, target_col="sales", time_col="date", timeout_minutes=60)
```

---

## AutoML UI Workflows

### Basic UI Workflow

1. **Navigate to AutoML**
   - Go to "Machine Learning" → "AutoML" in Databricks workspace
   - Click "Create AutoML Experiment"

2. **Configure Experiment**
   ```
   Experiment Name: Customer_Churn_AutoML
   ML Problem Type: Classification
   Input Training Dataset: catalog.schema.churn_training_table
   Prediction Target: churn
   ```

3. **Advanced Settings**
   ```
   Data Split: Use existing split column "split"
   Timeout: 60 minutes
   Evaluation Metric: F1 Score
   Exclude Columns: customer_id, timestamp
   ```

4. **Feature Store Integration**
   ```
   Primary Keys: customer_id
   Features Table: catalog.schema.customer_features
   Label Table: catalog.schema.churn_labels
   ```

### UI Best Practices

```markdown
✅ **Do's:**
- Use descriptive experiment names with timestamps
- Set appropriate timeout based on data size (start with 30-60 min)
- Exclude ID columns and irrelevant features
- Use existing train/val/test splits when available
- Enable feature store integration for production scenarios

❌ **Don'ts:**
- Don't include target leakage columns
- Don't set unrealistic short timeouts for complex datasets
- Don't ignore AutoML alerts about data quality issues
- Don't skip reviewing generated notebooks before production use
```

---

## AutoML API Patterns

### Basic AutoML API Usage

```python
from databricks import automl
from datetime import datetime
import mlflow

def run_automl_experiment(dataset, target_col, problem_type="classification", **kwargs):
    """
    Run AutoML experiment with comprehensive configuration
    
    Args:
        dataset: Spark DataFrame or table name
        target_col: Target column name
        problem_type: "classification", "regression", or "forecasting"
        **kwargs: Additional AutoML parameters
    """
    
    # Experiment configuration
    experiment_name = f"automl_{target_col}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    experiment_dir = f"/Users/{dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')}/automl_experiments"
    
    # Default parameters
    default_params = {
        "experiment_name": experiment_name,
        "experiment_dir": experiment_dir,
        "dataset": dataset,
        "target_col": target_col,
        "timeout_minutes": 60,
        "max_trials": 10,
        "primary_metric": "f1" if problem_type == "classification" else "r2"
    }
    
    # Merge with provided parameters
    params = {**default_params, **kwargs}
    
    try:
        if problem_type == "classification":
            automl_run = automl.classify(**params)
        elif problem_type == "regression":
            automl_run = automl.regressor(**params)
        elif problem_type == "forecasting":
            automl_run = automl.forecast(**params)
        else:
            raise ValueError(f"Unsupported problem type: {problem_type}")
        
        print(f"✅ AutoML experiment '{experiment_name}' completed successfully")
        print(f"📊 Experiment URL: {automl_run.experiment.experiment_url}")
        print(f"🏆 Best trial: {automl_run.best_trial}")
        
        return automl_run
        
    except Exception as e:
        print(f"❌ AutoML experiment failed: {e}")
        raise e

# Usage examples
# Basic classification
automl_run = run_automl_experiment(
    dataset=churn_df,
    target_col="churn",
    problem_type="classification",
    timeout_minutes=30
)

# Advanced configuration
automl_run = run_automl_experiment(
    dataset=sales_df,
    target_col="revenue",
    problem_type="regression",
    timeout_minutes=120,
    exclude_cols=["customer_id", "order_id"],
    primary_metric="rmse",
    max_trials=20
)
```

### AutoML with Pre-Split Data

```python
def run_automl_with_split(table_name, target_col, split_col="split", **kwargs):
    """
    Run AutoML using existing train/validation/test splits
    
    Args:
        table_name: Full table name (catalog.schema.table)
        target_col: Target column name
        split_col: Column containing split information
    """
    
    # Load data with existing splits
    dataset = spark.table(table_name)
    
    # Add metadata for better AutoML performance
    dataset = dataset.withMetadata(target_col, {"spark.contentAnnotation.semanticType": "categorical"})
    
    # Configure AutoML with split column
    automl_params = {
        "dataset": dataset,
        "target_col": target_col,
        "split_col": split_col,  # Requires DBRML 15.3+
        "experiment_name": f"automl_{table_name.split('.')[-1]}_{datetime.now().strftime('%Y%m%d_%H%M')}",
        **kwargs
    }
    
    automl_run = automl.classify(**automl_params)
    return automl_run

# Usage
automl_run = run_automl_with_split(
    table_name=f"{catalog}.{db}.churn_training",
    target_col="churn",
    split_col="split",
    timeout_minutes=45,
    exclude_cols=["customer_id", "transaction_ts"]
)
```

### AutoML with Metadata Enhancement

```python
def enhance_data_for_automl(df, target_col, categorical_cols=None, numerical_cols=None):
    """
    Add semantic metadata to improve AutoML performance
    
    Args:
        df: Spark DataFrame
        target_col: Target column name
        categorical_cols: List of categorical column names
        numerical_cols: List of numerical column names
    """
    
    enhanced_df = df
    
    # Add target metadata
    if target_col in df.columns:
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

automl_run = automl.classify(
    dataset=enhanced_df,
    target_col="churn",
    timeout_minutes=60
)
```

---

## Feature Store Integration

### AutoML with Feature Store Lookup

```python
from databricks.feature_engineering import FeatureEngineeringClient
from databricks.feature_store import FeatureLookup, FeatureFunction

def run_automl_with_feature_store(label_table, feature_lookups, target_col, **kwargs):
    """
    Run AutoML with Feature Store integration
    
    Args:
        label_table: Table containing labels and lookup keys
        feature_lookups: List of FeatureLookup objects
        target_col: Target column name
    """
    
    fe = FeatureEngineeringClient()
    
    # Create training set with feature store integration
    training_set = fe.create_training_set(
        df=spark.table(label_table),
        label=target_col,
        feature_lookups=feature_lookups,
        exclude_columns=["customer_id", "transaction_ts"]  # Exclude keys and timestamps
    )
    
    # Load as pandas for AutoML
    training_df = training_set.load_df().toPandas()
    
    # Run AutoML
    experiment_name = f"automl_with_fs_{datetime.now().strftime('%Y%m%d_%H%M')}"
    
    automl_run = automl.classify(
        experiment_name=experiment_name,
        dataset=training_df,
        target_col=target_col,
        **kwargs
    )
    
    return automl_run, training_set

# Define feature lookups
feature_lookups = [
    FeatureLookup(
        table_name=f"{catalog}.{db}.customer_features",
        lookup_key=["customer_id"],
        timestamp_lookup_key="transaction_ts"
    ),
    FeatureLookup(
        table_name=f"{catalog}.{db}.usage_features", 
        lookup_key=["customer_id"],
        timestamp_lookup_key="transaction_ts"
    )
]

# Run AutoML with Feature Store
automl_run, training_set = run_automl_with_feature_store(
    label_table=f"{catalog}.{db}.churn_labels",
    feature_lookups=feature_lookups,
    target_col="churn",
    timeout_minutes=90
)
```

### AutoML with On-Demand Features

```python
def run_automl_with_feature_functions(label_table, feature_lookups, feature_functions, target_col, **kwargs):
    """
    Run AutoML with Feature Store and on-demand feature functions
    
    Args:
        label_table: Table containing labels
        feature_lookups: List of FeatureLookup objects
        feature_functions: List of FeatureFunction objects
        target_col: Target column name
    """
    
    fe = FeatureEngineeringClient()
    
    # Combine feature lookups and functions
    all_features = feature_lookups + feature_functions
    
    # Create training set
    training_set = fe.create_training_set(
        df=spark.table(label_table),
        label=target_col,
        feature_lookups=all_features,
        exclude_columns=["customer_id", "transaction_ts", "split"]
    )
    
    # Load training data
    training_df = training_set.load_df().toPandas()
    
    print(f"Training set shape: {training_df.shape}")
    print(f"Features: {list(training_df.columns)}")
    
    # Run AutoML
    automl_run = automl.classify(
        dataset=training_df,
        target_col=target_col,
        experiment_name=f"automl_with_functions_{datetime.now().strftime('%Y%m%d_%H%M')}",
        **kwargs
    )
    
    return automl_run, training_set

# Define feature functions (SQL UDFs)
feature_functions = [
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

# Run AutoML with feature functions
automl_run, training_set = run_automl_with_feature_functions(
    label_table=f"{catalog}.{db}.churn_labels",
    feature_lookups=feature_lookups,
    feature_functions=feature_functions,
    target_col="churn",
    timeout_minutes=120
)
```

---

## Generated Notebook Patterns

### Leveraging AutoML Generated Code

```python
# AutoML generates complete notebooks with production-ready patterns
# Here are the key patterns to reuse:

class AutoMLGeneratedPatterns:
    """Reusable patterns from AutoML generated notebooks"""
    
    def __init__(self, target_col="churn"):
        self.target_col = target_col
        self.supported_cols = []  # Set by AutoML based on data analysis
        
    def setup_column_selector(self, supported_cols):
        """Setup column selector as generated by AutoML"""
        from databricks.automl_runtime.sklearn.column_selector import ColumnSelector
        
        self.supported_cols = supported_cols
        return ColumnSelector(supported_cols)
    
    def create_preprocessing_pipeline(self):
        """Create comprehensive preprocessing pipeline"""
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
        from databricks.automl_runtime.sklearn import OneHotEncoder as DBOneHotEncoder
        
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
                ("impute_mean", SimpleImputer(), ["monthly_charges", "total_charges", "tenure"])
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
            ("boolean", bool_pipeline, ["gender", "phone_service", "dependents"]),
            ("numerical", numerical_pipeline, ["monthly_charges", "total_charges", "tenure"]),
            ("categorical", categorical_pipeline, ["contract", "internet_service", "payment_method"])
        ]
        
        preprocessor = ColumnTransformer(transformers, remainder="passthrough", sparse_threshold=1)
        return preprocessor
    
    def create_model_pipeline(self, model_class, preprocessor, col_selector, **model_params):
        """Create complete model pipeline as generated by AutoML"""
        from sklearn.pipeline import Pipeline
        
        model = model_class(**model_params)
        
        pipeline = Pipeline([
            ("column_selector", col_selector),
            ("preprocessor", preprocessor),
            ("classifier", model),
        ])
        
        return pipeline
    
    def objective_function_template(self, params, X_train, y_train, X_val, y_val, X_test, y_test):
        """
        Template for hyperparameter optimization objective function
        Based on AutoML generated patterns
        """
        from lightgbm import LGBMClassifier
        from mlflow.models import Model, infer_signature
        from mlflow.pyfunc import PyFuncModel
        from mlflow import pyfunc
        from hyperopt import STATUS_OK
        
        with mlflow.start_run(run_name="automl_best_run") as mlflow_run:
            # Create model with hyperparameters
            classifier = LGBMClassifier(**params)
            
            # Create pipeline
            model = self.create_model_pipeline(
                model_class=LGBMClassifier,
                preprocessor=self.preprocessor,
                col_selector=self.col_selector,
                **params
            )
            
            # Enable MLflow autologging
            mlflow.sklearn.autolog(log_models=False, silent=True)
            
            # Train model
            model.fit(X_train, y_train)
            
            # Log model with signature
            signature = infer_signature(X_train, y_train)
            mlflow.sklearn.log_model(
                model, "sklearn_model",
                input_example=X_train.iloc[0].to_dict(),
                signature=signature
            )
            
            # Comprehensive evaluation
            mlflow_model = Model()
            pyfunc.add_to_model(mlflow_model, loader_module="mlflow.sklearn")
            pyfunc_model = PyFuncModel(model_meta=mlflow_model, model_impl=model)
            
            # Evaluate on all splits
            for split_name, (X_split, y_split) in [
                ("training", (X_train, y_train)),
                ("val", (X_val, y_val)),
                ("test", (X_test, y_test))
            ]:
                eval_result = mlflow.evaluate(
                    model=pyfunc_model,
                    data=X_split.assign(**{str(self.target_col): y_split}),
                    targets=self.target_col,
                    model_type="classifier",
                    evaluator_config={
                        "log_model_explainability": False,
                        "metric_prefix": f"{split_name}_",
                        "pos_label": "Yes"
                    }
                )
            
            # Return optimization objective
            val_metrics = eval_result.metrics
            loss = -val_metrics[f"val_f1_score"]
            
            return {
                "loss": loss,
                "status": STATUS_OK,
                "model": model,
                "run": mlflow_run,
            }

# Usage of generated patterns
automl_patterns = AutoMLGeneratedPatterns(target_col="churn")

# Setup components
col_selector = automl_patterns.setup_column_selector(supported_cols)
preprocessor = automl_patterns.create_preprocessing_pipeline()

# Store for use in objective function
automl_patterns.col_selector = col_selector
automl_patterns.preprocessor = preprocessor
```

### Data Split Handling from AutoML

```python
def handle_automl_data_splits(df_loaded, target_col):
    """
    Handle data splits as generated by AutoML notebooks
    AutoML creates _automl_split_col_xxxx or uses existing 'split' column
    """
    
    # Find split columns (AutoML generated or existing)
    split_cols = [c for c in df_loaded.columns if c.startswith('_automl_split_col') or c == 'split']
    
    if not split_cols:
        # Create manual split if no split column exists
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
    
    # Split data
    split_train_df = df_loaded.loc[df_loaded[split_col] == "train"]
    split_val_df = df_loaded.loc[df_loaded[split_col] == "validate"]
    split_test_df = df_loaded.loc[df_loaded[split_col] == "test"]
    
    # Extract features and targets
    X_train = split_train_df.drop([target_col] + split_cols, errors='ignore', axis=1)
    y_train = split_train_df[target_col]
    
    X_val = split_val_df.drop([target_col] + split_cols, errors='ignore', axis=1)
    y_val = split_val_df[target_col]
    
    X_test = split_test_df.drop([target_col] + split_cols, errors='ignore', axis=1)
    y_test = split_test_df[target_col]
    
    # Handle edge case where validation set is empty
    if len(X_val) == 0:
        X_val, y_val = X_test, y_test
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# Usage
X_train, X_val, X_test, y_train, y_val, y_test = handle_automl_data_splits(
    df_loaded, target_col="churn"
)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
```

---

## Advanced Configuration

### Custom AutoML Configuration

```python
class AdvancedAutoMLConfig:
    """Advanced AutoML configuration patterns"""
    
    @staticmethod
    def create_classification_config(dataset, target_col, **overrides):
        """Create comprehensive classification configuration"""
        
        base_config = {
            "dataset": dataset,
            "target_col": target_col,
            "timeout_minutes": 120,
            "max_trials": 15,
            "primary_metric": "f1",
            "exclude_cols": ["customer_id", "timestamp", "row_id"],
            "exclude_frameworks": ["sklearn"],  # Focus on tree-based models
            "experiment_name": f"advanced_classification_{datetime.now().strftime('%Y%m%d_%H%M')}",
            "experiment_dir": "/Users/{user}/automl_experiments",
            "data_dir": "/tmp/automl_data"
        }
        
        # Apply overrides
        config = {**base_config, **overrides}
        return config
    
    @staticmethod
    def create_regression_config(dataset, target_col, **overrides):
        """Create comprehensive regression configuration"""
        
        base_config = {
            "dataset": dataset,
            "target_col": target_col,
            "timeout_minutes": 90,
            "max_trials": 12,
            "primary_metric": "rmse",
            "exclude_cols": ["id", "timestamp"],
            "experiment_name": f"advanced_regression_{datetime.now().strftime('%Y%m%d_%H%M')}",
            "experiment_dir": "/Users/{user}/automl_experiments"
        }
        
        config = {**base_config, **overrides}
        return config
    
    @staticmethod
    def create_forecasting_config(dataset, target_col, time_col, **overrides):
        """Create comprehensive forecasting configuration"""
        
        base_config = {
            "dataset": dataset,
            "target_col": target_col,
            "time_col": time_col,
            "timeout_minutes": 180,
            "horizon": 30,  # Forecast 30 periods ahead
            "frequency": "D",  # Daily frequency
            "primary_metric": "smape",
            "experiment_name": f"advanced_forecasting_{datetime.now().strftime('%Y%m%d_%H%M')}",
            "experiment_dir": "/Users/{user}/automl_experiments"
        }
        
        config = {**base_config, **overrides}
        return config

# Usage examples
config = AdvancedAutoMLConfig.create_classification_config(
    dataset=churn_df,
    target_col="churn",
    timeout_minutes=180,
    primary_metric="roc_auc",
    exclude_frameworks=["sklearn", "xgboost"]  # Only LightGBM
)

automl_run = automl.classify(**config)
```

### Custom Metric and Evaluation

```python
def setup_custom_automl_evaluation(automl_run, validation_data, business_metrics=True):
    """
    Setup custom evaluation for AutoML results
    
    Args:
        automl_run: AutoML run object
        validation_data: Validation dataset for business metrics
        business_metrics: Whether to calculate business impact metrics
    """
    
    # Get best model
    best_trial = automl_run.best_trial
    best_model_uri = f"runs:/{best_trial.mlflow_run_id}/model"
    
    # Load model for evaluation
    model = mlflow.pyfunc.load_model(best_model_uri)
    
    # Make predictions
    predictions = model.predict(validation_data.drop(["churn"], axis=1))
    
    # Standard metrics
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    
    standard_metrics = {
        "accuracy": accuracy_score(validation_data["churn"], predictions),
        "roc_auc": roc_auc_score(validation_data["churn"], predictions),
        "classification_report": classification_report(validation_data["churn"], predictions)
    }
    
    # Business metrics (if enabled)
    business_metrics_results = {}
    if business_metrics:
        # Calculate business impact
        tn, fp, fn, tp = confusion_matrix(validation_data["churn"], predictions).ravel()
        
        # Business costs
        cost_per_churn = 2000  # Lost customer value
        cost_per_intervention = 500  # Retention campaign cost
        
        business_metrics_results = {
            "total_cost_avoided": tp * cost_per_churn - tp * cost_per_intervention,
            "false_positive_cost": fp * cost_per_intervention,
            "missed_churn_cost": fn * cost_per_churn,
            "net_business_value": (tp * (cost_per_churn - cost_per_intervention)) - (fp * cost_per_intervention)
        }
    
    evaluation_results = {
        "model_uri": best_model_uri,
        "standard_metrics": standard_metrics,
        "business_metrics": business_metrics_results,
        "confusion_matrix": confusion_matrix(validation_data["churn"], predictions).tolist()
    }
    
    return evaluation_results

# Usage
evaluation = setup_custom_automl_evaluation(
    automl_run=automl_run,
    validation_data=validation_df,
    business_metrics=True
)

print(f"Net Business Value: ${evaluation['business_metrics']['net_business_value']:,.2f}")
```

---

## Model Evaluation & Explainability

### SHAP Integration from AutoML

```python
def setup_automl_explainability(model, X_train, X_val, shap_enabled=True, sample_size=100):
    """
    Setup model explainability following AutoML patterns
    
    Args:
        model: Trained sklearn pipeline
        X_train: Training features
        X_val: Validation features
        shap_enabled: Whether to generate SHAP explanations
        sample_size: Sample size for SHAP calculations
    """
    
    if not shap_enabled:
        print("SHAP explanations disabled")
        return None
    
    try:
        from shap import KernelExplainer, summary_plot
        import warnings
        warnings.filterwarnings('ignore')
        
        # Handle missing values (SHAP requirement)
        mode = X_train.mode().iloc[0]
        train_sample = X_train.sample(n=min(sample_size, X_train.shape[0]), random_state=42).fillna(mode)
        val_sample = X_val.sample(n=min(sample_size, X_val.shape[0]), random_state=42).fillna(mode)
        
        # Create SHAP explainer
        predict_fn = lambda x: model.predict_proba(pd.DataFrame(x, columns=X_train.columns))
        explainer = KernelExplainer(predict_fn, train_sample, link="logit")
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(val_sample, l1_reg=False, nsamples=100)
        
        # Generate summary plot
        summary_plot(shap_values, val_sample, class_names=model.classes_, show=False)
        
        # Log SHAP artifacts to MLflow
        import matplotlib.pyplot as plt
        plt.savefig("/tmp/shap_summary.png", bbox_inches='tight', dpi=150)
        mlflow.log_artifact("/tmp/shap_summary.png", "explainability")
        
        print("✅ SHAP explanations generated successfully")
        
        return {
            "explainer": explainer,
            "shap_values": shap_values,
            "explained_samples": val_sample
        }
        
    except Exception as e:
        print(f"❌ SHAP explanation failed: {e}")
        return None

# Usage
explainability_results = setup_automl_explainability(
    model=best_model,
    X_train=X_train,
    X_val=X_val,
    shap_enabled=True,
    sample_size=50
)
```

### Model Performance Visualization

```python
def create_automl_evaluation_plots(y_true, y_pred, y_proba=None, save_path="/tmp/automl_plots"):
    """
    Create comprehensive evaluation plots following AutoML patterns
    
    Args:
        y_true: True labels
        y_pred: Predicted labels  
        y_proba: Prediction probabilities
        save_path: Path to save plots
    """
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
    import os
    
    os.makedirs(save_path, exist_ok=True)
    
    # 1. Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f"{save_path}/confusion_matrix.png", bbox_inches='tight')
    plt.close()
    
    if y_proba is not None:
        # 2. ROC Curve
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1] if y_proba.ndim > 1 else y_proba)
        auc_score = roc_auc_score(y_true, y_proba[:, 1] if y_proba.ndim > 1 else y_proba)
        
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.savefig(f"{save_path}/roc_curve.png", bbox_inches='tight')
        plt.close()
        
        # 3. Precision-Recall Curve
        plt.figure(figsize=(8, 6))
        precision, recall, _ = precision_recall_curve(y_true, y_proba[:, 1] if y_proba.ndim > 1 else y_proba)
        
        plt.plot(recall, precision, label='Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.savefig(f"{save_path}/precision_recall_curve.png", bbox_inches='tight')
        plt.close()
    
    # Log all plots to MLflow
    for plot_file in os.listdir(save_path):
        if plot_file.endswith('.png'):
            mlflow.log_artifact(os.path.join(save_path, plot_file), "evaluation_plots")
    
    print(f"✅ Evaluation plots saved to {save_path} and logged to MLflow")

# Usage
create_automl_evaluation_plots(
    y_true=y_test,
    y_pred=test_predictions,
    y_proba=test_probabilities,
    save_path="/tmp/automl_evaluation"
)
```

---

## MLOps Pipeline Integration

### AutoML in MLOps Workflows

```python
class AutoMLMLOpsIntegration:
    """Integration patterns for AutoML in MLOps pipelines"""
    
    def __init__(self, catalog, schema):
        self.catalog = catalog
        self.schema = schema
        self.fe = FeatureEngineeringClient()
        
    def automated_automl_pipeline(self, config):
        """
        Complete automated AutoML pipeline for MLOps
        
        Args:
            config: Pipeline configuration dict
        """
        
        pipeline_results = {}
        
        try:
            # 1. Data preparation and validation
            print("🔄 Step 1: Data preparation...")
            dataset = self._prepare_automl_data(config["data_source"])
            pipeline_results["data_prep"] = "success"
            
            # 2. Run AutoML experiment
            print("🔄 Step 2: Running AutoML...")
            automl_run = self._run_automl_experiment(dataset, config["automl_config"])
            pipeline_results["automl_run"] = automl_run
            
            # 3. Model validation and testing
            print("🔄 Step 3: Model validation...")
            validation_results = self._validate_automl_model(automl_run, config["validation_config"])
            pipeline_results["validation"] = validation_results
            
            # 4. Model registration (if validation passes)
            if validation_results["passed"]:
                print("🔄 Step 4: Model registration...")
                registered_model = self._register_automl_model(automl_run, config["model_config"])
                pipeline_results["registered_model"] = registered_model
                
                # 5. Model deployment (optional)
                if config.get("auto_deploy", False):
                    print("🔄 Step 5: Model deployment...")
                    deployment_results = self._deploy_automl_model(registered_model)
                    pipeline_results["deployment"] = deployment_results
            
            print("✅ AutoML MLOps pipeline completed successfully")
            return pipeline_results
            
        except Exception as e:
            print(f"❌ AutoML MLOps pipeline failed: {e}")
            pipeline_results["error"] = str(e)
            return pipeline_results
    
    def _prepare_automl_data(self, data_config):
        """Prepare data for AutoML with validation"""
        
        # Load data
        if data_config["type"] == "table":
            dataset = spark.table(data_config["source"])
        elif data_config["type"] == "feature_store":
            dataset = self._load_from_feature_store(data_config)
        else:
            raise ValueError(f"Unsupported data type: {data_config['type']}")
        
        # Data quality validation
        self._validate_data_quality(dataset, data_config["target_col"])
        
        # Add metadata for AutoML
        dataset = self._add_semantic_metadata(dataset, data_config)
        
        return dataset
    
    def _run_automl_experiment(self, dataset, automl_config):
        """Run AutoML experiment with error handling"""
        
        try:
            automl_run = automl.classify(**automl_config)
            
            # Wait for completion
            while automl_run.state.name != "FINISHED":
                time.sleep(30)
                print(f"AutoML status: {automl_run.state.name}")
            
            return automl_run
            
        except Exception as e:
            print(f"AutoML experiment failed: {e}")
            raise e
    
    def _validate_automl_model(self, automl_run, validation_config):
        """Validate AutoML model against business requirements"""
        
        best_trial = automl_run.best_trial
        metrics = best_trial.metrics
        
        validation_results = {
            "passed": True,
            "checks": {},
            "metrics": metrics
        }
        
        # Performance thresholds
        thresholds = validation_config.get("thresholds", {})
        
        for metric, threshold in thresholds.items():
            actual_value = metrics.get(metric, 0)
            passed = actual_value >= threshold
            
            validation_results["checks"][metric] = {
                "actual": actual_value,
                "threshold": threshold,
                "passed": passed
            }
            
            if not passed:
                validation_results["passed"] = False
        
        return validation_results
    
    def _register_automl_model(self, automl_run, model_config):
        """Register AutoML model to Unity Catalog"""
        
        best_trial = automl_run.best_trial
        model_uri = f"runs:/{best_trial.mlflow_run_id}/model"
        model_name = model_config["name"]
        
        # Register model
        registered_model = mlflow.register_model(model_uri, model_name)
        
        # Add model description and tags
        client = MlflowClient()
        client.update_registered_model(
            name=model_name,
            description=model_config.get("description", "AutoML generated model")
        )
        
        # Set initial alias
        client.set_registered_model_alias(
            name=model_name,
            alias="Challenger",
            version=registered_model.version
        )
        
        return registered_model

# Usage
automl_mlops = AutoMLMLOpsIntegration(catalog, schema)

pipeline_config = {
    "data_source": {
        "type": "table",
        "source": f"{catalog}.{schema}.churn_training",
        "target_col": "churn"
    },
    "automl_config": {
        "dataset": None,  # Will be set by pipeline
        "target_col": "churn",
        "timeout_minutes": 90,
        "primary_metric": "f1"
    },
    "validation_config": {
        "thresholds": {
            "f1_score": 0.75,
            "precision": 0.70,
            "recall": 0.70
        }
    },
    "model_config": {
        "name": f"{catalog}.{schema}.automl_churn_model",
        "description": "AutoML generated churn prediction model"
    },
    "auto_deploy": False
}

results = automl_mlops.automated_automl_pipeline(pipeline_config)
```

### AutoML Model Monitoring Setup

```python
def setup_automl_model_monitoring(registered_model_name, inference_table):
    """
    Setup monitoring for AutoML-generated models
    
    Args:
        registered_model_name: Name of registered model in Unity Catalog
        inference_table: Table containing model predictions
    """
    
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.catalog import MonitorInferenceLog, MonitorInferenceLogProblemType
    
    w = WorkspaceClient()
    
    try:
        # Create monitor for inference table
        monitor_info = w.quality_monitors.create(
            table_name=inference_table,
            inference_log=MonitorInferenceLog(
                problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_CLASSIFICATION,
                prediction_col="prediction",
                timestamp_col="inference_timestamp",
                granularities=["1 day"],
                model_id_col="model_version",
                label_col="churn"  # If available
            ),
            assets_dir="/tmp/automl_monitoring",
            output_schema_name=f"{catalog}.{schema}",
            baseline_table_name=f"{catalog}.{schema}.churn_baseline"
        )
        
        print(f"✅ Monitoring setup complete for {registered_model_name}")
        print(f"📊 Monitor ID: {monitor_info.monitor_name}")
        
        return monitor_info
        
    except Exception as e:
        print(f"❌ Monitoring setup failed: {e}")
        return None

# Usage
monitor_info = setup_automl_model_monitoring(
    registered_model_name=f"{catalog}.{schema}.automl_churn_model",
    inference_table=f"{catalog}.{schema}.churn_inference_table"
)
```

---

## Best Practices & Optimization

### AutoML Performance Optimization

```python
class AutoMLOptimization:
    """Best practices for AutoML performance optimization"""
    
    @staticmethod
    def optimize_dataset_for_automl(df, target_col, max_rows=1000000, max_features=1000):
        """
        Optimize dataset for AutoML performance
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            max_rows: Maximum number of rows for AutoML
            max_features: Maximum number of features
        """
        
        optimized_df = df.copy()
        
        # 1. Sample data if too large
        if len(optimized_df) > max_rows:
            print(f"⚠️ Dataset has {len(optimized_df)} rows, sampling to {max_rows}")
            optimized_df = optimized_df.sample(n=max_rows, random_state=42)
        
        # 2. Remove high-cardinality categorical features
        categorical_cols = optimized_df.select_dtypes(include=['object']).columns
        high_cardinality_cols = []
        
        for col in categorical_cols:
            if col != target_col and optimized_df[col].nunique() > 100:
                high_cardinality_cols.append(col)
        
        if high_cardinality_cols:
            print(f"⚠️ Removing high-cardinality columns: {high_cardinality_cols}")
            optimized_df = optimized_df.drop(columns=high_cardinality_cols)
        
        # 3. Remove columns with too many missing values
        missing_threshold = 0.8
        high_missing_cols = []
        
        for col in optimized_df.columns:
            if col != target_col and optimized_df[col].isna().mean() > missing_threshold:
                high_missing_cols.append(col)
        
        if high_missing_cols:
            print(f"⚠️ Removing high-missing columns: {high_missing_cols}")
            optimized_df = optimized_df.drop(columns=high_missing_cols)
        
        # 4. Feature selection if too many features
        if len(optimized_df.columns) > max_features:
            print(f"⚠️ Too many features ({len(optimized_df.columns)}), applying feature selection")
            optimized_df = AutoMLOptimization._apply_feature_selection(
                optimized_df, target_col, max_features
            )
        
        print(f"✅ Dataset optimized: {optimized_df.shape}")
        return optimized_df
    
    @staticmethod
    def _apply_feature_selection(df, target_col, max_features):
        """Apply feature selection to reduce dimensionality"""
        
        from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
        from sklearn.preprocessing import LabelEncoder
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Encode categorical variables for feature selection
        X_encoded = X.copy()
        label_encoders = {}
        
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
        
        # Select top features
        selector = SelectKBest(score_func=mutual_info_classif, k=max_features-1)
        X_selected = selector.fit_transform(X_encoded, y)
        
        # Get selected feature names
        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask].tolist()
        
        # Return DataFrame with selected features + target
        result_df = df[selected_features + [target_col]].copy()
        
        print(f"✅ Selected {len(selected_features)} features using mutual information")
        return result_df
    
    @staticmethod
    def get_optimal_automl_config(dataset_size, problem_complexity="medium"):
        """
        Get optimal AutoML configuration based on dataset characteristics
        
        Args:
            dataset_size: Tuple of (rows, columns)
            problem_complexity: "simple", "medium", "complex"
        """
        
        rows, cols = dataset_size
        
        # Base configuration
        config = {
            "max_trials": 10,
            "timeout_minutes": 60,
            "exclude_frameworks": []
        }
        
        # Adjust based on dataset size
        if rows < 10000:
            config.update({
                "timeout_minutes": 30,
                "max_trials": 5
            })
        elif rows > 500000:
            config.update({
                "timeout_minutes": 180,
                "max_trials": 20,
                "exclude_frameworks": ["sklearn"]  # Use only tree-based models for large data
            })
        
        # Adjust based on problem complexity
        if problem_complexity == "simple":
            config.update({
                "timeout_minutes": config["timeout_minutes"] // 2,
                "max_trials": max(config["max_trials"] // 2, 3)
            })
        elif problem_complexity == "complex":
            config.update({
                "timeout_minutes": config["timeout_minutes"] * 2,
                "max_trials": config["max_trials"] * 2
            })
        
        print(f"📋 Optimal AutoML config for {rows:,} rows × {cols} cols ({problem_complexity}):")
        for key, value in config.items():
            print(f"   {key}: {value}")
        
        return config

# Usage
# Optimize dataset
optimized_df = AutoMLOptimization.optimize_dataset_for_automl(
    df=churn_df,
    target_col="churn",
    max_rows=500000,
    max_features=50
)

# Get optimal configuration
optimal_config = AutoMLOptimization.get_optimal_automl_config(
    dataset_size=optimized_df.shape,
    problem_complexity="medium"
)

# Run AutoML with optimized settings
automl_run = automl.classify(
    dataset=optimized_df,
    target_col="churn",
    **optimal_config
)
```

### AutoML Quality Assurance

```python
def automl_quality_checklist(automl_run, validation_data):
    """
    Comprehensive quality checklist for AutoML results
    
    Args:
        automl_run: Completed AutoML run
        validation_data: Validation dataset for additional checks
    """
    
    quality_report = {
        "overall_score": 0,
        "checks": {},
        "recommendations": []
    }
    
    # 1. Model Performance Check
    best_trial = automl_run.best_trial
    metrics = best_trial.metrics
    
    performance_score = 0
    if metrics.get("f1_score", 0) >= 0.7:
        performance_score += 25
    if metrics.get("precision", 0) >= 0.7:
        performance_score += 25
    if metrics.get("recall", 0) >= 0.7:
        performance_score += 25
    if metrics.get("roc_auc", 0) >= 0.8:
        performance_score += 25
    
    quality_report["checks"]["performance"] = {
        "score": performance_score,
        "metrics": metrics
    }
    
    # 2. Data Quality Check
    data_quality_score = 0
    
    # Check for data leakage indicators
    feature_importance = getattr(best_trial, "feature_importance", {})
    if feature_importance:
        # Flag suspiciously high importance features
        max_importance = max(feature_importance.values()) if feature_importance else 0
        if max_importance < 0.8:  # No single feature dominates
            data_quality_score += 50
        else:
            quality_report["recommendations"].append(
                "⚠️ Single feature has very high importance - check for data leakage"
            )
    
    quality_report["checks"]["data_quality"] = {"score": data_quality_score}
    
    # 3. Model Complexity Check
    complexity_score = 50  # Default score
    
    # Prefer simpler models for interpretability
    if "lightgbm" in str(best_trial.model_path).lower():
        complexity_score = 70
    elif "sklearn" in str(best_trial.model_path).lower():
        complexity_score = 90
    
    quality_report["checks"]["complexity"] = {"score": complexity_score}
    
    # 4. Generalization Check (if validation data provided)
    generalization_score = 0
    if validation_data is not None:
        try:
            # Load best model and test on validation data
            model_uri = f"runs:/{best_trial.mlflow_run_id}/model"
            model = mlflow.pyfunc.load_model(model_uri)
            
            val_predictions = model.predict(validation_data.drop(["churn"], axis=1))
            val_accuracy = accuracy_score(validation_data["churn"], val_predictions)
            
            # Compare with training accuracy
            train_accuracy = metrics.get("accuracy", 0)
            accuracy_gap = abs(train_accuracy - val_accuracy)
            
            if accuracy_gap < 0.05:  # Less than 5% gap
                generalization_score = 100
            elif accuracy_gap < 0.1:  # Less than 10% gap
                generalization_score = 75
            else:
                generalization_score = 25
                quality_report["recommendations"].append(
                    f"⚠️ Large accuracy gap ({accuracy_gap:.3f}) between train and validation"
                )
            
        except Exception as e:
            quality_report["recommendations"].append(f"❌ Could not validate generalization: {e}")
    
    quality_report["checks"]["generalization"] = {"score": generalization_score}
    
    # Calculate overall score
    total_possible = sum(100 for _ in quality_report["checks"])
    total_actual = sum(check["score"] for check in quality_report["checks"].values())
    quality_report["overall_score"] = (total_actual / total_possible) * 100
    
    # Generate recommendations based on score
    if quality_report["overall_score"] >= 80:
        quality_report["recommendations"].insert(0, "✅ High quality AutoML result - ready for production")
    elif quality_report["overall_score"] >= 60:
        quality_report["recommendations"].insert(0, "⚠️ Good AutoML result - consider improvements before production")
    else:
        quality_report["recommendations"].insert(0, "❌ AutoML result needs improvement - not ready for production")
    
    return quality_report

# Usage
quality_report = automl_quality_checklist(
    automl_run=automl_run,
    validation_data=validation_df
)

print(f"Overall Quality Score: {quality_report['overall_score']:.1f}/100")
for recommendation in quality_report["recommendations"]:
    print(recommendation)
```

---

## Complete AutoML Workflow Example

```python
def complete_automl_workflow():
    """
    Complete end-to-end AutoML workflow demonstrating all patterns
    """
    
    print("🚀 Starting Complete AutoML Workflow")
    
    # 1. Data Preparation
    print("\n📊 Step 1: Data Preparation")
    raw_data = spark.table(f"{catalog}.{schema}.churn_training")
    
    # Optimize dataset
    optimized_data = AutoMLOptimization.optimize_dataset_for_automl(
        df=raw_data.toPandas(),
        target_col="churn"
    )
    
    # Add metadata
    enhanced_data = enhance_data_for_automl(
        df=spark.createDataFrame(optimized_data),
        target_col="churn",
        categorical_cols=["gender", "contract", "payment_method"],
        numerical_cols=["monthly_charges", "total_charges", "tenure"]
    )
    
    # 2. AutoML Configuration
    print("\n⚙️ Step 2: AutoML Configuration")
    optimal_config = AutoMLOptimization.get_optimal_automl_config(
        dataset_size=optimized_data.shape,
        problem_complexity="medium"
    )
    
    automl_config = {
        "dataset": enhanced_data,
        "target_col": "churn",
        "experiment_name": f"complete_automl_{datetime.now().strftime('%Y%m%d_%H%M')}",
        "split_col": "split",
        "exclude_cols": ["customer_id"],
        **optimal_config
    }
    
    # 3. Run AutoML
    print("\n🤖 Step 3: Running AutoML")
    automl_run = automl.classify(**automl_config)
    
    # 4. Quality Validation
    print("\n✅ Step 4: Quality Validation")
    validation_data = optimized_data[optimized_data["split"] == "validate"]
    quality_report = automl_quality_checklist(automl_run, validation_data)
    
    print(f"Quality Score: {quality_report['overall_score']:.1f}/100")
    
    # 5. Model Registration (if quality passes)
    if quality_report["overall_score"] >= 70:
        print("\n📝 Step 5: Model Registration")
        
        best_trial = automl_run.best_trial
        model_uri = f"runs:/{best_trial.mlflow_run_id}/model"
        model_name = f"{catalog}.{schema}.automl_churn_complete"
        
        registered_model = mlflow.register_model(model_uri, model_name)
        
        # Add model documentation
        client = MlflowClient()
        client.update_registered_model(
            name=model_name,
            description=f"AutoML generated churn model. Quality Score: {quality_report['overall_score']:.1f}/100"
        )
        
        # Set alias
        client.set_registered_model_alias(
            name=model_name,
            alias="Challenger",
            version=registered_model.version
        )
        
        print(f"✅ Model registered: {model_name} v{registered_model.version}")
        
        # 6. Setup Monitoring
        print("\n📈 Step 6: Setup Monitoring")
        monitor_info = setup_automl_model_monitoring(
            registered_model_name=model_name,
            inference_table=f"{catalog}.{schema}.churn_inference_table"
        )
        
        workflow_results = {
            "automl_run": automl_run,
            "quality_report": quality_report,
            "registered_model": registered_model,
            "monitor_info": monitor_info,
            "status": "success"
        }
    else:
        print("\n❌ Quality validation failed - model not registered")
        workflow_results = {
            "automl_run": automl_run,
            "quality_report": quality_report,
            "status": "quality_failed"
        }
    
    print("\n🎉 Complete AutoML Workflow Finished")
    return workflow_results

# Execute complete workflow
workflow_results = complete_automl_workflow()

# Print final summary
print("\n" + "="*50)
print("WORKFLOW SUMMARY")
print("="*50)
print(f"Status: {workflow_results['status']}")
print(f"Quality Score: {workflow_results['quality_report']['overall_score']:.1f}/100")

if workflow_results['status'] == 'success':
    print(f"Registered Model: {workflow_results['registered_model'].name}")
    print("✅ Ready for production deployment")
else:
    print("❌ Review quality issues before proceeding")
```

This comprehensive AutoML guide provides production-ready patterns for integrating Databricks AutoML into MLOps workflows, from basic UI usage to advanced API integration with Feature Store and monitoring.