# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Databricks MLOps end-to-end demo repository focused on customer churn prediction. It demonstrates a complete MLOps pipeline using Databricks Lakehouse, MLflow, Auto ML, and Unity Catalog model registry.

## Repository Structure

The repository contains two main demo tracks:

- **01-mlops-quickstart/**: Basic MLOps workflow covering fundamental concepts
- **02-mlops-advanced/**: Advanced MLOps with model serving, online tables, monitoring, and drift detection
- **_resources/**: Shared utilities and setup scripts

## Development Environment

This is a **Databricks notebook-based project** with Python notebooks (`.py` files) that contain Databricks-specific magic commands:
- `# MAGIC %md` for markdown cells
- `# MAGIC %sql` for SQL commands  
- `# MAGIC %pip install` for package installation
- `# MAGIC %run` for including other notebooks
- `# COMMAND ----------` for cell separation

## Key Dependencies

All notebooks use these core packages:
- `mlflow==2.22.0` - ML lifecycle management
- `databricks-automl-runtime==0.2.21` - AutoML functionality
- Databricks Feature Engineering Client
- Databricks SDK (`databricks.sdk`)

## Common Notebook Execution Patterns

1. **Setup Phase**: Each notebook runs `%run ../_resources/00-setup` to initialize environment
2. **Package Installation**: Uses `%pip install` followed by `dbutils.library.restartPython()`
3. **Data Access**: Tables accessed via `spark.read.table()` or `spark.table()`
4. **MLflow Integration**: Configured for Unity Catalog model registry with `mlflow.set_registry_uri("databricks-uc")`

## Workflow Architecture

### Quickstart Track (01-mlops-quickstart/)
1. **00_mlops_end2end_quickstart_presentation.py** - Overview and setup
2. **01_feature_engineering.py** - Data preparation and feature store
3. **02_automl_best_run.py** - Model training with AutoML
4. **03_from_notebook_to_models_in_uc.py** - Model registration in Unity Catalog
5. **04_challenger_validation.py** - Model validation and promotion
6. **05_batch_inference.py** - Batch scoring pipeline

### Advanced Track (02-mlops-advanced/)
Same workflow as quickstart plus:
6. **06_serve_features_and_model.py** - Real-time serving endpoints
7. **07_model_monitoring.py** - Lakehouse monitoring setup
8. **08_drift_detection.py** - Data drift detection and alerts

## Data Pipeline

- **Bronze Layer**: Raw customer data from IBM Telco dataset
- **Feature Store**: Engineered features for ML training
- **Model Registry**: Versioned models in Unity Catalog
- **Inference Tables**: Batch and real-time prediction results

## Configuration

Key configuration managed in `_resources/00-setup.py`:
- Catalog: `main__build`
- Database: `dbdemos_mlops`
- Experiment paths: `/Users/{current_user}/dbdemos_mlops`

## Testing and Validation

Model validation includes:
- Champion-challenger model comparison
- Model performance metrics validation
- Data quality checks
- Drift detection monitoring

## Jobs and Workflows

The demo includes pre-configured Databricks workflows:
- **Init Job**: Sets up demo environment and runs quickstart pipeline
- **Retraining Job**: Automated retraining triggered by drift detection

## MLOps Best Practices Guides

This repository includes comprehensive best practices guides covering all aspects of production MLOps on Databricks:

### 📊 [Feature Engineering Best Practices](./feature_eng_claude.md)
Comprehensive guide on feature engineering patterns for MLOps:
- **Data preparation**: Multi-modal analysis and feature store integration
- **Engineering approaches**: Pandas on Spark vs PySpark with UDFs
- **Feature Store patterns**: Separating features from labels, on-demand functions
- **AutoML integration**: Metadata annotation and feature semantics
- **10 key principles** for production-ready feature engineering

### 🤖 [Model Training and Validation](./model_train_validate_claude.md)
Complete framework for model training and validation:
- **Training patterns**: Data lineage tracking and preprocessing pipelines
- **Model registration**: Unity Catalog integration with lifecycle management
- **Validation frameworks**: Multi-dimensional validation and business metrics
- **Hyperparameter optimization**: Structured search with MLflow tracking
- **20 best practices** for production-ready model development

### 🚀 [Model Deployment and Serving](./model_deployment_serving_claude.md)
Production deployment strategies for batch and real-time serving:
- **Batch inference**: Spark UDFs and Feature Store integration
- **Real-time serving**: Online tables and serving endpoints
- **A/B testing**: Multi-version deployment with traffic management
- **Performance optimization**: Scaling and cost optimization strategies
- **16 best practices** for reliable production deployment

### 📈 [Model Monitoring and Alerting](./model_monitoring_alerting_claude.md)
Advanced monitoring and alerting for production ML systems:
- **Lakehouse monitoring**: Comprehensive data and model monitoring
- **Drift detection**: Multi-dimensional analysis with risk scoring
- **Intelligent alerting**: Multi-channel notifications with business context
- **Automated responses**: Workflow integration with conditional logic
- **24 best practices** for maintaining model health in production

### 🔬 [AutoML Best Practices](./auto_ml_claude.md)
Leveraging AutoML for rapid, production-ready ML development:
- **Data preparation**: Semantic metadata and quality optimization
- **Code generation**: Reusable preprocessing and training patterns
- **Feature Store integration**: Complex feature engineering with lineage
- **Production integration**: Model registration and validation frameworks
- **24 best practices** for production AutoML implementation

### Quick Reference: Key Patterns by Use Case

**For Data Scientists starting new projects:**
1. Begin with [AutoML guide](./auto_ml_claude.md) for rapid prototyping
2. Follow [Feature Engineering guide](./feature_eng_claude.md) for feature development
3. Use [Training & Validation guide](./model_train_validate_claude.md) for model development

**For ML Engineers deploying to production:**
1. Reference [Deployment guide](./model_deployment_serving_claude.md) for serving strategies
2. Implement [Monitoring guide](./model_monitoring_alerting_claude.md) for operational excellence
3. Follow validation frameworks in [Training guide](./model_train_validate_claude.md)

**For MLOps Engineers building platforms:**
- All guides contain infrastructure-as-code patterns
- Workflow automation examples throughout
- Unity Catalog integration patterns
- Comprehensive error handling and retry logic

## Important Notes

- This is a **demo environment** - not production-ready code
- All notebooks are designed to run in Databricks workspace
- Unity Catalog integration requires appropriate permissions
- Some notebooks generate synthetic data for demonstration purposes