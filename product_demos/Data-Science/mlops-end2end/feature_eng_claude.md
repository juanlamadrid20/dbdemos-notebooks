# Feature Engineering Best Practices for MLOps on Databricks

This guide captures feature engineering best practices from the Databricks MLOps end-to-end demo, focusing on scalable, production-ready approaches for customer churn prediction.

## Overview

Feature engineering is the foundation of successful MLOps pipelines. This project demonstrates two approaches:
- **Quickstart**: Simple feature engineering with labeled training data
- **Advanced**: Feature Store integration with separate feature and label tables

## Key Dependencies

```python
# Core packages for feature engineering
%pip install mlflow==2.22.0 
%pip install databricks-automl-runtime==0.2.21
%pip install databricks-feature-engineering==0.12.1  # Advanced track only
```

## Data Exploration Patterns

### 1. Multi-Modal Data Analysis
```python
# SQL-based exploration
%sql
SELECT * FROM mlops_churn_bronze_customers

# Pandas API for visualization
telco_df = spark.read.table("mlops_churn_bronze_customers").pandas_api()
telco_df["internet_service"].value_counts().plot.pie()

# Spark DataFrame for processing
telcoDF = spark.read.table("mlops_churn_bronze_customers")
display(telcoDF)
```

**Best Practice**: Use multiple analysis approaches - SQL for quick exploration, Pandas API for familiar data science workflows, and Spark DataFrames for scalable processing.

## Feature Engineering Approaches

### Approach 1: Pandas on Spark API (Quickstart)

**When to use**: Data scientists familiar with pandas, simple transformations, smaller datasets.

```python
import pyspark.sql.functions as F
from pyspark.sql import DataFrame

def clean_churn_features(dataDF: DataFrame) -> DataFrame:
    """
    Simple cleaning function leveraging pandas API
    """
    # Convert to pandas on spark dataframe
    data_psdf = dataDF.pandas_api()
    
    # Type conversions
    data_psdf = data_psdf.astype({"senior_citizen": "string"})
    data_psdf["senior_citizen"] = data_psdf["senior_citizen"].map({"1": "Yes", "0": "No"})
    
    # Handle missing values and data cleaning
    data_psdf["total_charges"] = data_psdf["total_charges"].apply(
        lambda x: float(x) if x.strip() else 0
    )
    
    # Fill missing numerical values
    data_psdf = data_psdf.fillna({
        "tenure": 0.0,
        "monthly_charges": 0.0,
        "total_charges": 0.0
    })
    
    # Feature engineering: count optional services
    def sum_optional_services(df):
        cols = ["online_security", "online_backup", "device_protection", 
                "tech_support", "streaming_tv", "streaming_movies"]
        return sum(map(lambda c: (df[c] == "Yes"), cols))
    
    data_psdf["num_optional_services"] = sum_optional_services(data_psdf)
    
    # Return Spark DataFrame
    return data_psdf.to_spark()
```

**Key Benefits**:
- Familiar pandas syntax
- Automatic distribution via Spark
- Easy transition from pandas workflows

### Approach 2: PySpark with Pandas UDFs (Advanced)

**When to use**: Complex aggregations, better performance control, larger datasets.

```python
from pyspark.sql.functions import pandas_udf, col, when, lit
from pyspark.sql import DataFrame as SparkDataFrame

def compute_service_features(inputDF: SparkDataFrame) -> SparkDataFrame:
    # Create pandas UDF function
    @pandas_udf('double')
    def num_optional_services(*cols):
        return sum(map(lambda s: (s == "Yes").astype('double'), cols))
    
    return inputDF.withColumn("num_optional_services",
        num_optional_services("online_security", "online_backup", 
                             "device_protection", "tech_support", 
                             "streaming_tv", "streaming_movies"))

def clean_churn_features(dataDF: SparkDataFrame) -> SparkDataFrame:
    """
    Advanced cleaning using pandas API with metadata
    """
    data_psdf = dataDF.pandas_api()
    
    # Data cleaning (same as quickstart)
    data_psdf = data_psdf.astype({"senior_citizen": "string"})
    data_psdf["senior_citizen"] = data_psdf["senior_citizen"].map({"1": "Yes", "0": "No"})
    data_psdf["total_charges"] = data_psdf["total_charges"].apply(
        lambda x: float(x) if x.strip() else 0
    )
    data_psdf = data_psdf.fillna({
        "tenure": 0.0, "monthly_charges": 0.0, "total_charges": 0.0
    })
    
    # Convert back to Spark and add metadata for AutoML
    data_cleanDF = data_psdf.to_spark()
    data_cleanDF = data_cleanDF.withMetadata("customer_id", 
        {"spark.contentAnnotation.semanticType": "native"})
    data_cleanDF = data_cleanDF.withMetadata("num_optional_services", 
        {"spark.contentAnnotation.semanticType": "numeric"})
    
    return data_cleanDF
```

**Key Benefits**:
- Better performance for complex operations
- Metadata annotation for AutoML
- More control over Spark execution

## Train-Validation-Test Split Pattern

**Best Practice**: Always implement proper data splitting for model validation.

```python
import pyspark.sql.functions as F

# Define split ratios
train_ratio, val_ratio, test_ratio = 0.7, 0.2, 0.1

# Create deterministic splits
churn_features = (churn_features
    .withColumn("random", F.rand(seed=42))
    .withColumn("split",
        F.when(F.col("random") < train_ratio, "train")
        .when(F.col("random") < train_ratio + val_ratio, "validate")
        .otherwise("test"))
    .drop("random"))

# Save training table
(churn_features.write.mode("overwrite")
               .option("overwriteSchema", "true")
               .saveAsTable("mlops_churn_training"))
```

## Feature Store Integration (Advanced)

### 1. Separate Features from Labels

**Best Practice**: Prevent label leakage by storing features and labels in separate tables.

```python
from datetime import datetime
from pyspark.sql.functions import lit

# Add timestamp for feature versioning
this_time = (datetime.now()).timestamp()
churn_features = (clean_churn_features(compute_service_features(telcoDF))
                 .withColumn("transaction_ts", lit(this_time).cast("timestamp")))

# Separate label table
churn_features.select("customer_id", "transaction_ts", "churn") \
              .withColumn("random", F.rand(seed=42)) \
              .withColumn("split",
                          F.when(F.col("random") < train_ratio, "train")
                          .when(F.col("random") < train_ratio + val_ratio, "validate")
                          .otherwise("test")) \
              .drop("random") \
              .write.format("delta") \
              .mode("overwrite").option("overwriteSchema", "true") \
              .saveAsTable("advanced_churn_label_table")

# Feature table without labels
churn_featuresDF = churn_features.drop("churn")
```

### 2. Create Feature Store Tables

```python
from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()

# Create feature table with primary keys and time series support
churn_feature_table = fe.create_table(
    name="advanced_churn_feature_table",
    primary_keys=["customer_id", "transaction_ts"],
    schema=churn_featuresDF.schema,
    timeseries_columns="transaction_ts",
    description="Customer churn features with service counts and cleaned data"
)

# Write features to the feature store
fe.write_table(
    name=f"{catalog}.{db}.advanced_churn_feature_table",
    df=churn_featuresDF,
    mode='merge'  # Supports schema evolution
)
```

### 3. Add Primary Key Constraints

```sql
-- Essential for feature lookup operations
ALTER TABLE advanced_churn_label_table ALTER COLUMN customer_id SET NOT NULL;
ALTER TABLE advanced_churn_label_table ALTER COLUMN transaction_ts SET NOT NULL;
ALTER TABLE advanced_churn_label_table ADD CONSTRAINT advanced_churn_label_table_pk 
    PRIMARY KEY(customer_id, transaction_ts);
```

## On-Demand Feature Functions

**Best Practice**: Use Unity Catalog functions for features that need real-time calculation.

```sql
CREATE OR REPLACE FUNCTION avg_price_increase(
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
```

**Benefits**:
- Governance through Unity Catalog
- Reusable across batch and real-time serving
- Version controlled and documented

## AutoML Integration Patterns

### 1. Metadata for Feature Types

```python
# Help AutoML understand feature semantics
churn_features = churn_features.withMetadata("num_optional_services", 
    {"spark.contentAnnotation.semanticType": "numeric"})
```

### 2. AutoML with Feature Store

```python
from databricks import automl

# Method 1: Direct table training (Quickstart)
automl_run = automl.classify(
    experiment_name=xp_name,
    experiment_dir=xp_path,
    dataset=churn_features,
    target_col="churn",
    split_col="split",
    timeout_minutes=10,
    exclude_cols='customer_id'
)

# Method 2: Feature Store integration (Advanced)
# Select label table and join with feature store tables via UI or API
```

## Data Quality Patterns

### 1. Missing Value Handling

```python
# Explicit missing value strategy
data_psdf = data_psdf.fillna({
    "tenure": 0.0,           # Business logic: new customers
    "monthly_charges": 0.0,   # Zero for inactive services
    "total_charges": 0.0      # Consistent with monthly charges
})
```

### 2. Data Type Conversions

```python
# Explicit type conversion with mapping
data_psdf = data_psdf.astype({"senior_citizen": "string"})
data_psdf["senior_citizen"] = data_psdf["senior_citizen"].map({"1": "Yes", "0": "No"})

# Safe numeric conversion with error handling
data_psdf["total_charges"] = data_psdf["total_charges"].apply(
    lambda x: float(x) if x.strip() else 0
)
```

## Documentation and Lineage

### 1. Table Comments

```python
# Add business context to tables
spark.sql(f"""
COMMENT ON TABLE {catalog}.{db}.mlops_churn_training IS 
'Features derived from bronze customers table. 
Service features and cleaned names. No aggregations performed.'
""")
```

### 2. Feature Table Descriptions

```python
# Detailed feature table documentation
churn_feature_table = fe.create_table(
    name="advanced_churn_feature_table",
    primary_keys=["customer_id", "transaction_ts"],
    schema=churn_featuresDF.schema,
    timeseries_columns="transaction_ts",
    description=f"""Customer churn features derived from {bronze_table_name}. 
    Includes service counts and data cleaning. 
    Warning: Ground truth labels stored separately to prevent leakage."""
)
```

## Performance Optimization

### 1. Efficient Column Operations

```python
# Use vectorized operations instead of row-by-row processing
@pandas_udf('double')
def num_optional_services(*cols):
    return sum(map(lambda s: (s == "Yes").astype('double'), cols))
```

### 2. Schema Evolution Support

```python
# Enable schema evolution for feature tables
fe.write_table(
    name=f"{catalog}.{db}.advanced_churn_feature_table",
    df=churn_featuresDF,
    mode='merge'  # Supports adding new columns
)
```

## Summary of Best Practices

1. **Use appropriate APIs**: Pandas on Spark for familiarity, PySpark UDFs for performance
2. **Separate features from labels**: Prevent leakage in production systems
3. **Implement proper data splits**: Use deterministic splits with seeds
4. **Add metadata**: Help AutoML and downstream systems understand features
5. **Use Feature Store**: For governance, reusability, and lineage
6. **Create on-demand functions**: For real-time feature calculation
7. **Document everything**: Tables, functions, and business logic
8. **Handle missing data explicitly**: With business-justified strategies
9. **Enable schema evolution**: For maintenance and updates
10. **Add constraints**: Primary keys and NOT NULL for data quality

This feature engineering approach provides a solid foundation for scalable MLOps pipelines on Databricks, supporting both batch and real-time inference scenarios.