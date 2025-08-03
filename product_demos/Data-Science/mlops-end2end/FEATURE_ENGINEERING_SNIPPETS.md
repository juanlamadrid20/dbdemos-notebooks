# MLOps Feature Engineering Code Snippets

This document contains reusable code snippets for feature engineering in MLOps workflows using Databricks. The snippets are organized from basic to advanced patterns.

## Table of Contents

- [Basic Data Cleaning with Pandas API on Spark](#basic-data-cleaning-with-pandas-api-on-spark)
- [Advanced Feature Computation with PandasUDF](#advanced-feature-computation-with-pandasudf)
- [Feature Store Integration](#feature-store-integration)
- [On-Demand Feature Functions](#on-demand-feature-functions)
- [Train/Validation/Test Split](#trainvalidationtest-split)
- [AutoML Integration](#automl-integration)

---

## Basic Data Cleaning with Pandas API on Spark

Scale pandas operations across Spark clusters for data cleaning and feature engineering.

```python
import pyspark.sql.functions as F
from pyspark.sql import DataFrame

def clean_churn_features(dataDF: DataFrame) -> DataFrame:
    """
    Simple cleaning function leveraging pandas API
    """
    # Convert to pandas on spark dataframe
    data_psdf = dataDF.pandas_api()
    
    # Convert some columns
    data_psdf = data_psdf.astype({"senior_citizen": "string"})
    data_psdf["senior_citizen"] = data_psdf["senior_citizen"].map({"1": "Yes", "0": "No"})
    
    # Handle string-to-float conversion with error handling
    data_psdf["total_charges"] = data_psdf["total_charges"].apply(
        lambda x: float(x) if x.strip() else 0
    )
    
    # Fill missing numerical values with 0
    data_psdf = data_psdf.fillna({
        "tenure": 0.0,
        "monthly_charges": 0.0,
        "total_charges": 0.0
    })
    
    # Count optional services
    def sum_optional_services(df):
        """Count number of optional services enabled, like streaming TV"""
        cols = ["online_security", "online_backup", "device_protection", 
                "tech_support", "streaming_tv", "streaming_movies"]
        return sum(map(lambda c: (df[c] == "Yes"), cols))
    
    data_psdf["num_optional_services"] = sum_optional_services(data_psdf)
    
    # Return the cleaned Spark dataframe
    return data_psdf.to_spark()
```

---

## Advanced Feature Computation with PandasUDF

Use PandasUDF for efficient vectorized operations on Spark DataFrames.

```python
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql.functions import pandas_udf, col, when, lit

def compute_service_features(inputDF: SparkDataFrame) -> SparkDataFrame:
    """Count the number of optional services enabled using PandasUDF"""
    
    # Create pandas UDF function
    @pandas_udf('double')
    def num_optional_services(*cols):
        # Nested helper function to count optional services in pandas dataframe
        return sum(map(lambda s: (s == "Yes").astype('double'), cols))
    
    return inputDF.withColumn(
        "num_optional_services",
        num_optional_services(
            "online_security", "online_backup", "device_protection", 
            "tech_support", "streaming_tv", "streaming_movies"
        )
    )
```

---

## Feature Store Integration

Create and manage feature tables using Databricks Feature Store with Unity Catalog.

### Feature Table Creation

```python
from databricks.feature_engineering import FeatureEngineeringClient
from datetime import datetime
from pyspark.sql.functions import lit

# Initialize Feature Engineering Client
fe = FeatureEngineeringClient()

# Add timestamp for time-series features
this_time = (datetime.now()).timestamp()
churn_features_df = clean_churn_features(compute_service_features(telco_df)) \
                    .withColumn("transaction_ts", lit(this_time).cast("timestamp"))

# Create feature table
churn_feature_table = fe.create_table(
    name="advanced_churn_feature_table",
    primary_keys=["customer_id", "transaction_ts"],
    schema=churn_features_df.schema,
    timeseries_columns="transaction_ts",
    description="Features derived from bronze customers table with service features and data cleaning"
)

# Write features to the table
fe.write_table(
    name=f"{catalog}.{db}.advanced_churn_feature_table",
    df=churn_features_df,
    mode='merge'  # 'merge' supports schema evolution
)
```

### Label Table Separation (Best Practice)

```python
import pyspark.sql.functions as F

# Extract ground-truth labels separately to avoid label leakage
train_ratio, val_ratio, test_ratio = 0.7, 0.2, 0.1

churn_features_df.select("customer_id", "transaction_ts", "churn") \
                 .withColumn("random", F.rand(seed=42)) \
                 .withColumn("split",
                            F.when(F.col("random") < train_ratio, "train")
                            .when(F.col("random") < train_ratio + val_ratio, "validate")
                            .otherwise("test")) \
                 .drop("random") \
                 .write.format("delta") \
                 .mode("overwrite").option("overwriteSchema", "true") \
                 .saveAsTable("advanced_churn_label_table")

# Add primary key constraints
spark.sql("""
    ALTER TABLE advanced_churn_label_table DROP CONSTRAINT IF EXISTS advanced_churn_label_table_pk;
    ALTER TABLE advanced_churn_label_table ALTER COLUMN customer_id SET NOT NULL;
    ALTER TABLE advanced_churn_label_table ALTER COLUMN transaction_ts SET NOT NULL;
    ALTER TABLE advanced_churn_label_table ADD CONSTRAINT advanced_churn_label_table_pk 
    PRIMARY KEY(customer_id, transaction_ts);
""")
```

---

## On-Demand Feature Functions

Create SQL-based feature functions for real-time feature computation.

```sql
CREATE OR REPLACE FUNCTION avg_price_increase(
    monthly_charges_in DOUBLE, 
    tenure_in DOUBLE, 
    total_charges_in DOUBLE
)
RETURNS FLOAT
LANGUAGE PYTHON
COMMENT "[Feature Function] Calculate potential average price increase for tenured customers"
AS $$
if tenure_in > 0:
    return monthly_charges_in - total_charges_in/tenure_in
else:
    return 0
$$
```

**Usage in Python:**

```python
# Test the feature function
spark.sql("""
    SELECT customer_id, 
           monthly_charges, 
           tenure, 
           total_charges,
           avg_price_increase(monthly_charges, tenure, total_charges) as price_increase
    FROM advanced_churn_feature_table
    LIMIT 10
""").show()
```

---

## Train/Validation/Test Split

Create deterministic data splits with categorical labels for ML workflows.

```python
import pyspark.sql.functions as F

def create_data_splits(df, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
    """
    Create train/validation/test splits with categorical labels
    """
    return (df.withColumn("random", F.rand(seed=seed))
             .withColumn("split",
                        F.when(F.col("random") < train_ratio, "train")
                        .when(F.col("random") < train_ratio + val_ratio, "validate")
                        .otherwise("test"))
             .drop("random"))

# Apply splits
churn_features_with_splits = create_data_splits(churn_features)

# Write training table
(churn_features_with_splits.write.mode("overwrite")
                          .option("overwriteSchema", "true")
                          .saveAsTable("mlops_churn_training"))

# Add table comment for documentation
spark.sql(f"""
    COMMENT ON TABLE {catalog}.{db}.mlops_churn_training IS 
    'Features derived from bronze customers table with service features and data cleaning. 
     No aggregations were performed.'
""")
```

---

## AutoML Integration

### Basic AutoML with Feature Table

```python
from databricks import automl
from datetime import datetime

# Prepare experiment configuration
xp_path = f"/Users/{current_user}/dbdemos_mlops"
xp_name = f"dbdemos_automl_churn_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"

# Add semantic metadata for AutoML
churn_features = churn_features.withMetadata(
    "num_optional_services", 
    {"spark.contentAnnotation.semanticType": "numeric"}
)

# Run AutoML experiment
try:
    automl_run = automl.classify(
        experiment_name=xp_name,
        experiment_dir=xp_path,
        dataset=churn_features,
        target_col="churn",
        split_col="split",  # Requires DBRML 15.3+
        timeout_minutes=10,
        exclude_cols='customer_id'
    )
    
    # Set experiment permissions for sharing
    DBDemos.set_experiment_permission(f"{xp_path}/{xp_name}")
    
except Exception as e:
    if "cannot import name 'automl'" in str(e):
        print("AutoML not available in serverless compute")
    else:
        raise e
```

### Advanced AutoML with Feature Store

When using Feature Store, AutoML can automatically join features from multiple tables:

```python
# AutoML with Feature Store integration (UI-based)
# 1. Select the label table: advanced_churn_label_table
# 2. Join features from: advanced_churn_feature_table
# 3. Target column: churn
# 4. AutoML handles feature lookup automatically
```

---

## Best Practices

### 1. Metadata Management
```python
# Add semantic type metadata for better AutoML performance
df = df.withMetadata("customer_id", {"spark.contentAnnotation.semanticType": "native"})
df = df.withMetadata("num_optional_services", {"spark.contentAnnotation.semanticType": "numeric"})
```

### 2. Schema Evolution
```python
# Use merge mode for schema evolution in feature tables
fe.write_table(
    name=f"{catalog}.{db}.feature_table",
    df=features_df,
    mode='merge'  # Allows schema changes over time
)
```

### 3. Online Feature Table Management
```python
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# Clean up existing online tables before creating new ones
try:
    w.online_tables.delete(f"{catalog}.{db}.feature_table_online")
    print("Dropped existing online feature table")
except Exception as e:
    print(f"No existing online table found: {e}")
```

### 4. Data Quality Validation
```python
# Add NOT NULL constraints for data quality
spark.sql("""
    ALTER TABLE feature_table ALTER COLUMN customer_id SET NOT NULL;
    ALTER TABLE feature_table ALTER COLUMN transaction_ts SET NOT NULL;
""")
```

---

## Usage Examples

### Complete Feature Engineering Pipeline

```python
# 1. Read raw data
raw_df = spark.read.table("bronze_customers")

# 2. Apply feature engineering
features_df = clean_churn_features(compute_service_features(raw_df))

# 3. Add timestamp
features_df = features_df.withColumn("transaction_ts", lit(datetime.now()).cast("timestamp"))

# 4. Create feature table
fe.create_table(
    name="customer_features",
    primary_keys=["customer_id", "transaction_ts"],
    schema=features_df.schema,
    timeseries_columns="transaction_ts"
)

# 5. Write features
fe.write_table(name="customer_features", df=features_df, mode='merge')

# 6. Extract labels separately
labels_df = features_df.select("customer_id", "transaction_ts", "target")
labels_df.write.mode("overwrite").saveAsTable("customer_labels")
```

This completes the feature engineering patterns commonly used in MLOps workflows with Databricks.