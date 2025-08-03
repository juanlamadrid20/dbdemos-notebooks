# MLOps Monitoring & Alerting Code Snippets

This document contains reusable code snippets for comprehensive MLOps monitoring, alerting, and automated response systems using Databricks Lakehouse Monitoring. Focuses on production-grade monitoring strategies, business metrics, and intelligent alerting.

## Table of Contents

- [Lakehouse Monitoring Setup](#lakehouse-monitoring-setup)
- [Custom Business Metrics](#custom-business-metrics)
- [Advanced Drift Detection](#advanced-drift-detection)
- [Multi-Channel Alerting System](#multi-channel-alerting-system)
- [Dashboard & Visualization Integration](#dashboard--visualization-integration)
- [Automated Response Workflows](#automated-response-workflows)
- [Performance Monitoring](#performance-monitoring)
- [Compliance & Audit Monitoring](#compliance--audit-monitoring)

---

## Lakehouse Monitoring Setup

### Comprehensive Monitor Configuration

```python
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    MonitorInferenceLog, MonitorInferenceLogProblemType, 
    MonitorMetric, MonitorMetricType, MonitorInfoStatus
)
from pyspark.sql.types import DoubleType, StructField
import time

class LakehouseMonitorManager:
    """Comprehensive Lakehouse Monitoring management class"""
    
    def __init__(self, catalog, schema):
        self.w = WorkspaceClient()
        self.catalog = catalog
        self.schema = schema
        
    def create_comprehensive_monitor(self, table_name, baseline_table_name, 
                                   config=None, custom_metrics=None):
        """
        Create comprehensive monitoring setup with custom business metrics
        
        Args:
            table_name: Full name of inference table to monitor
            baseline_table_name: Training/baseline data table
            config: Monitor configuration dictionary
            custom_metrics: List of custom business metrics
        """
        
        # Default configuration
        default_config = {
            "prediction_col": "prediction",
            "label_col": "churn",
            "timestamp_col": "inference_timestamp", 
            "model_id_col": "model_version",
            "granularities": ["1 hour", "1 day", "1 week"],
            "slicing_exprs": [
                "senior_citizen='Yes'",
                "contract='Month-to-month'",
                "payment_method='Electronic check'",
                "internet_service='Fiber optic'"
            ],
            "assets_dir": "/tmp/monitoring"
        }
        
        config = {**default_config, **(config or {})}
        
        # Define comprehensive custom business metrics
        if custom_metrics is None:
            custom_metrics = self._get_default_business_metrics()
        
        try:
            monitor_info = self.w.quality_monitors.create(
                table_name=table_name,
                inference_log=MonitorInferenceLog(
                    problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_CLASSIFICATION,
                    prediction_col=config["prediction_col"],
                    timestamp_col=config["timestamp_col"],
                    granularities=config["granularities"],
                    model_id_col=config["model_id_col"],
                    label_col=config["label_col"]
                ),
                assets_dir=config["assets_dir"],
                output_schema_name=f"{self.catalog}.{self.schema}",
                baseline_table_name=baseline_table_name,
                slicing_exprs=config["slicing_exprs"],
                custom_metrics=custom_metrics
            )
            
            print(f"✅ Monitor created for {table_name}")
            
        except Exception as e:
            if "already exist" in str(e).lower():
                print(f"Monitor exists, retrieving info for {table_name}")
                monitor_info = self.w.quality_monitors.get(table_name=table_name)
            else:
                raise e
        
        # Wait for monitor to be active
        self._wait_for_monitor_active(table_name)
        
        return monitor_info
    
    def _get_default_business_metrics(self):
        """Define comprehensive business metrics for monitoring"""
        
        return [
            # Expected loss metric (business impact)
            MonitorMetric(
                type=MonitorMetricType.CUSTOM_METRIC_TYPE_AGGREGATE,
                name="expected_loss",
                input_columns=[":table"],
                definition="""avg(CASE
                WHEN {{prediction_col}} != {{label_col}} AND {{label_col}} = 'Yes' THEN -monthly_charges
                ELSE 0 END
                )""",
                output_data_type=StructField("output", DoubleType()).json()
            ),
            
            # Revenue at risk metric
            MonitorMetric(
                type=MonitorMetricType.CUSTOM_METRIC_TYPE_AGGREGATE,
                name="revenue_at_risk",
                input_columns=[":table"],
                definition="""sum(CASE
                WHEN {{prediction_col}} = 'Yes' THEN monthly_charges * 12
                ELSE 0 END
                )""",
                output_data_type=StructField("output", DoubleType()).json()
            ),
            
            # False positive cost (unnecessary retention offers)
            MonitorMetric(
                type=MonitorMetricType.CUSTOM_METRIC_TYPE_AGGREGATE,
                name="false_positive_cost",
                input_columns=[":table"],
                definition="""sum(CASE
                WHEN {{prediction_col}} = 'Yes' AND {{label_col}} = 'No' THEN 500
                ELSE 0 END
                )""",
                output_data_type=StructField("output", DoubleType()).json()
            ),
            
            # High value customer prediction accuracy
            MonitorMetric(
                type=MonitorMetricType.CUSTOM_METRIC_TYPE_AGGREGATE,
                name="high_value_accuracy",
                input_columns=[":table"],
                definition="""
                sum(CASE WHEN monthly_charges > 80 AND {{prediction_col}} = {{label_col}} THEN 1 ELSE 0 END) / 
                sum(CASE WHEN monthly_charges > 80 THEN 1 ELSE 0 END)
                """,
                output_data_type=StructField("output", DoubleType()).json()
            )
        ]
    
    def _wait_for_monitor_active(self, table_name, timeout_minutes=30):
        """Wait for monitor to become active"""
        
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60
        
        while time.time() - start_time < timeout_seconds:
            monitor_info = self.w.quality_monitors.get(table_name=table_name)
            
            if monitor_info.status == MonitorInfoStatus.MONITOR_STATUS_ACTIVE:
                print(f"✅ Monitor is active for {table_name}")
                return monitor_info
            
            print(f"⏳ Monitor status: {monitor_info.status}, waiting...")
            time.sleep(30)
        
        raise TimeoutError(f"Monitor failed to become active within {timeout_minutes} minutes")

# Usage
monitor_manager = LakehouseMonitorManager(catalog, db)

monitor_info = monitor_manager.create_comprehensive_monitor(
    table_name=f"{catalog}.{db}.churn_inference_table",
    baseline_table_name=f"{catalog}.{db}.churn_baseline",
    config={
        "granularities": ["1 hour", "1 day"],
        "slicing_exprs": ["contract", "senior_citizen='Yes'"]
    }
)
```

---

## Custom Business Metrics

### Advanced Business Impact Metrics

```python
def create_industry_specific_metrics(industry="telecom"):
    """
    Create industry-specific business metrics for monitoring
    
    Args:
        industry: Industry type (telecom, retail, finance, healthcare)
    """
    
    if industry == "telecom":
        return [
            # Customer Lifetime Value impact
            MonitorMetric(
                type=MonitorMetricType.CUSTOM_METRIC_TYPE_AGGREGATE,
                name="clv_impact",
                input_columns=[":table"],
                definition="""avg(CASE
                WHEN {{prediction_col}} = 'Yes' AND {{label_col}} = 'Yes' THEN monthly_charges * tenure * 0.8
                WHEN {{prediction_col}} = 'No' AND {{label_col}} = 'Yes' THEN monthly_charges * tenure * -1.0
                ELSE 0 END
                )""",
                output_data_type=StructField("output", DoubleType()).json()
            ),
            
            # Network service prediction accuracy
            MonitorMetric(
                type=MonitorMetricType.CUSTOM_METRIC_TYPE_AGGREGATE,
                name="fiber_customer_accuracy",
                input_columns=[":table"],
                definition="""
                sum(CASE WHEN internet_service = 'Fiber optic' AND {{prediction_col}} = {{label_col}} THEN 1 ELSE 0 END) / 
                sum(CASE WHEN internet_service = 'Fiber optic' THEN 1 ELSE 0 END)
                """,
                output_data_type=StructField("output", DoubleType()).json()
            )
        ]
    
    elif industry == "retail":
        return [
            # Purchase behavior impact
            MonitorMetric(
                type=MonitorMetricType.CUSTOM_METRIC_TYPE_AGGREGATE,
                name="purchase_impact",
                input_columns=[":table"],
                definition="""avg(CASE
                WHEN {{prediction_col}} = 'high_value' AND actual_purchase_amount > 1000 THEN 1.0
                WHEN {{prediction_col}} = 'low_value' AND actual_purchase_amount < 100 THEN 1.0
                ELSE 0.0 END
                )""",
                output_data_type=StructField("output", DoubleType()).json()
            )
        ]
    
    elif industry == "finance":
        return [
            # Risk assessment accuracy
            MonitorMetric(
                type=MonitorMetricType.CUSTOM_METRIC_TYPE_AGGREGATE,
                name="risk_assessment_accuracy",
                input_columns=[":table"],
                definition="""
                sum(CASE WHEN {{prediction_col}} = {{label_col}} THEN credit_limit ELSE 0 END) / 
                sum(credit_limit)
                """,
                output_data_type=StructField("output", DoubleType()).json()
            )
        ]
    
    return []

# Usage
telecom_metrics = create_industry_specific_metrics("telecom")
```

### Dynamic Threshold Metrics

```python
def create_adaptive_threshold_metrics():
    """Create metrics with adaptive thresholds based on historical performance"""
    
    return [
        # Prediction confidence metric
        MonitorMetric(
            type=MonitorMetricType.CUSTOM_METRIC_TYPE_AGGREGATE,
            name="prediction_confidence",
            input_columns=[":table"],
            definition="""avg(CASE
            WHEN prediction_probability > 0.8 OR prediction_probability < 0.2 THEN 1.0
            ELSE 0.0 END
            )""",
            output_data_type=StructField("output", DoubleType()).json()
        ),
        
        # Model uncertainty metric
        MonitorMetric(
            type=MonitorMetricType.CUSTOM_METRIC_TYPE_AGGREGATE,
            name="model_uncertainty",
            input_columns=[":table"],
            definition="""avg(CASE
            WHEN prediction_probability BETWEEN 0.4 AND 0.6 THEN 1.0
            ELSE 0.0 END
            )""",
            output_data_type=StructField("output", DoubleType()).json()
        )
    ]
```

---

## Advanced Drift Detection

### Multi-Dimensional Drift Analysis

```python
class AdvancedDriftDetector:
    """Advanced drift detection with multiple metrics and adaptive thresholds"""
    
    def __init__(self, monitor_table_name):
        self.monitor_table_name = monitor_table_name
        self.w = WorkspaceClient()
        
        # Get monitoring table names
        monitor_info = self.w.quality_monitors.get(table_name=monitor_table_name)
        self.drift_table_name = monitor_info.drift_metrics_table_name
        self.profile_table_name = monitor_info.profile_metrics_table_name
    
    def detect_comprehensive_drift(self, time_window_days=7, thresholds=None):
        """
        Comprehensive drift detection across multiple dimensions
        
        Args:
            time_window_days: Number of days to analyze
            thresholds: Custom threshold configuration
        """
        
        if thresholds is None:
            thresholds = {
                "feature_drift": {
                    "js_distance": 0.15,
                    "chi_squared_pvalue": 0.05,
                    "tv_distance": 0.20
                },
                "prediction_drift": {
                    "js_distance": 0.10,
                    "tv_distance": 0.15
                },
                "label_drift": {
                    "js_distance": 0.20,
                    "chi_squared_pvalue": 0.01
                },
                "performance": {
                    "f1_score": 0.65,
                    "precision": 0.60,
                    "recall": 0.60,
                    "expected_loss": 40,
                    "revenue_at_risk": 100000
                }
            }
        
        # Analyze different types of drift
        drift_results = {
            "feature_drift": self._analyze_feature_drift(time_window_days, thresholds["feature_drift"]),
            "prediction_drift": self._analyze_prediction_drift(time_window_days, thresholds["prediction_drift"]),
            "label_drift": self._analyze_label_drift(time_window_days, thresholds["label_drift"]),
            "performance_degradation": self._analyze_performance_degradation(time_window_days, thresholds["performance"])
        }
        
        # Calculate overall risk score
        drift_results["overall_risk_score"] = self._calculate_risk_score(drift_results)
        drift_results["severity_level"] = self._determine_severity_level(drift_results["overall_risk_score"])
        
        return drift_results
    
    def _analyze_feature_drift(self, time_window_days, thresholds):
        """Analyze feature-level drift"""
        
        query = f"""
        SELECT 
            column_name,
            js_distance,
            chi_squared_test.pvalue as chi_squared_pvalue,
            tv_distance,
            window.start as time_window
        FROM {self.drift_table_name}
        WHERE 
            window.start >= date_sub(current_date(), {time_window_days})
            AND column_name NOT IN ('prediction', 'churn')
            AND drift_type = 'CONSECUTIVE'
        ORDER BY js_distance DESC
        """
        
        drift_df = spark.sql(query)
        
        # Count violations per threshold type
        violations = {}
        for metric, threshold in thresholds.items():
            if metric == "chi_squared_pvalue":
                # For p-value, violation is when p < threshold (significant drift)
                violations[metric] = drift_df.filter(F.col(metric) < threshold).count()
            else:
                # For distance metrics, violation is when value > threshold
                violations[metric] = drift_df.filter(F.col(metric) > threshold).count()
        
        return {
            "violations": violations,
            "total_features_checked": drift_df.select("column_name").distinct().count(),
            "details": drift_df.collect()
        }
    
    def _analyze_prediction_drift(self, time_window_days, thresholds):
        """Analyze prediction distribution drift"""
        
        query = f"""
        SELECT 
            js_distance,
            tv_distance,
            window.start as time_window
        FROM {self.drift_table_name}
        WHERE 
            window.start >= date_sub(current_date(), {time_window_days})
            AND column_name = 'prediction'
            AND drift_type = 'CONSECUTIVE'
        """
        
        drift_df = spark.sql(query)
        
        violations = {
            metric: drift_df.filter(F.col(metric) > threshold).count()
            for metric, threshold in thresholds.items()
        }
        
        return {
            "violations": violations,
            "details": drift_df.collect()
        }
    
    def _analyze_label_drift(self, time_window_days, thresholds):
        """Analyze label distribution drift"""
        
        query = f"""
        SELECT 
            js_distance,
            chi_squared_test.pvalue as chi_squared_pvalue,
            window.start as time_window
        FROM {self.drift_table_name}
        WHERE 
            window.start >= date_sub(current_date(), {time_window_days})
            AND column_name = 'churn'
            AND drift_type = 'CONSECUTIVE'
        """
        
        drift_df = spark.sql(query)
        
        violations = {}
        for metric, threshold in thresholds.items():
            if metric == "chi_squared_pvalue":
                violations[metric] = drift_df.filter(F.col(metric) < threshold).count()
            else:
                violations[metric] = drift_df.filter(F.col(metric) > threshold).count()
        
        return {
            "violations": violations,
            "details": drift_df.collect()
        }
    
    def _analyze_performance_degradation(self, time_window_days, thresholds):
        """Analyze model performance degradation"""
        
        query = f"""
        SELECT 
            f1_score.macro as f1_score,
            precision.macro as precision,
            recall.macro as recall,
            expected_loss,
            revenue_at_risk,
            window.start as time_window
        FROM {self.profile_table_name}
        WHERE 
            window.start >= date_sub(current_date(), {time_window_days})
            AND log_type = 'INPUT'
            AND column_name = ':table'
            AND slice_key is null
        """
        
        perf_df = spark.sql(query)
        
        violations = {}
        for metric, threshold in thresholds.items():
            if metric in ["expected_loss", "revenue_at_risk"]:
                # For cost metrics, violation when absolute value exceeds threshold
                violations[metric] = perf_df.filter(F.abs(F.col(metric)) > threshold).count()
            else:
                # For accuracy metrics, violation when below threshold
                violations[metric] = perf_df.filter(F.col(metric) < threshold).count()
        
        return {
            "violations": violations,
            "details": perf_df.collect()
        }
    
    def _calculate_risk_score(self, drift_results):
        """Calculate overall risk score from 0-100"""
        
        weights = {
            "feature_drift": 0.25,
            "prediction_drift": 0.30,
            "label_drift": 0.20,
            "performance_degradation": 0.25
        }
        
        risk_score = 0
        
        for drift_type, weight in weights.items():
            violations = drift_results[drift_type]["violations"]
            total_violations = sum(violations.values())
            
            # Normalize by expected maximum violations
            max_expected = {
                "feature_drift": 20,  # Assuming ~20 features
                "prediction_drift": 5,
                "label_drift": 5,
                "performance_degradation": 10
            }
            
            normalized_violations = min(total_violations / max_expected[drift_type], 1.0)
            risk_score += normalized_violations * weight * 100
        
        return min(risk_score, 100.0)
    
    def _determine_severity_level(self, risk_score):
        """Determine severity level based on risk score"""
        
        if risk_score >= 80:
            return "CRITICAL"
        elif risk_score >= 60:
            return "HIGH"
        elif risk_score >= 40:
            return "MEDIUM"
        elif risk_score >= 20:
            return "LOW"
        else:
            return "MINIMAL"

# Usage
detector = AdvancedDriftDetector(f"{catalog}.{db}.churn_inference_table")

drift_analysis = detector.detect_comprehensive_drift(
    time_window_days=7,
    thresholds={
        "performance": {
            "f1_score": 0.70,
            "expected_loss": 35
        }
    }
)

print(f"🎯 Overall Risk Score: {drift_analysis['overall_risk_score']:.1f}")
print(f"🚨 Severity Level: {drift_analysis['severity_level']}")
```

---

## Multi-Channel Alerting System

### Comprehensive Notification Manager

```python
import requests
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import logging

class MLOpsNotificationManager:
    """Comprehensive multi-channel notification system for MLOps alerts"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.logger = self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging for notification tracking"""
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
    
    def send_comprehensive_alert(self, alert_data, channels=None, priority="medium"):
        """
        Send comprehensive alert across multiple channels
        
        Args:
            alert_data: Alert information dictionary
            channels: List of channels to use ['slack', 'email', 'teams', 'pagerduty']
            priority: Alert priority (low, medium, high, critical)
        """
        
        if channels is None:
            channels = self._get_default_channels_for_priority(priority)
        
        # Enrich alert data
        enriched_alert = self._enrich_alert_data(alert_data, priority)
        
        # Send to each channel
        notification_results = {}
        
        for channel in channels:
            try:
                if channel == "slack":
                    result = self._send_slack_notification(enriched_alert)
                elif channel == "email":
                    result = self._send_email_notification(enriched_alert)
                elif channel == "teams":
                    result = self._send_teams_notification(enriched_alert)
                elif channel == "pagerduty":
                    result = self._send_pagerduty_notification(enriched_alert)
                elif channel == "webhook":
                    result = self._send_webhook_notification(enriched_alert)
                else:
                    result = {"success": False, "error": f"Unknown channel: {channel}"}
                
                notification_results[channel] = result
                
            except Exception as e:
                self.logger.error(f"Failed to send {channel} notification: {e}")
                notification_results[channel] = {"success": False, "error": str(e)}
        
        return notification_results
    
    def _get_default_channels_for_priority(self, priority):
        """Get default notification channels based on priority"""
        
        channel_map = {
            "low": ["email"],
            "medium": ["slack", "email"],
            "high": ["slack", "email", "teams"],
            "critical": ["slack", "email", "teams", "pagerduty"]
        }
        
        return channel_map.get(priority, ["email"])
    
    def _enrich_alert_data(self, alert_data, priority):
        """Enrich alert data with additional context"""
        
        enriched = {
            **alert_data,
            "timestamp": datetime.now().isoformat(),
            "priority": priority,
            "emoji": self._get_priority_emoji(priority),
            "color": self._get_priority_color(priority),
            "tags": self._generate_alert_tags(alert_data),
            "runbook_url": self._get_runbook_url(alert_data.get("alert_type")),
            "dashboard_url": self._get_dashboard_url(alert_data.get("model_name"))
        }
        
        return enriched
    
    def _get_priority_emoji(self, priority):
        """Get emoji for priority level"""
        
        emoji_map = {
            "low": "🟡",
            "medium": "🟠", 
            "high": "🔴",
            "critical": "🚨"
        }
        return emoji_map.get(priority, "ℹ️")
    
    def _get_priority_color(self, priority):
        """Get color code for priority level"""
        
        color_map = {
            "low": "#ffeb3b",
            "medium": "#ff9800",
            "high": "#f44336",
            "critical": "#d32f2f"
        }
        return color_map.get(priority, "#2196f3")
    
    def _generate_alert_tags(self, alert_data):
        """Generate relevant tags for the alert"""
        
        tags = [alert_data.get("alert_type", "unknown")]
        
        if "model_name" in alert_data:
            tags.append(f"model:{alert_data['model_name'].split('.')[-1]}")
        
        if "severity_level" in alert_data:
            tags.append(f"severity:{alert_data['severity_level'].lower()}")
        
        return tags
    
    def _get_runbook_url(self, alert_type):
        """Get runbook URL for alert type"""
        
        runbook_map = {
            "drift_detection": "https://company.wiki/mlops/drift-runbook",
            "performance_degradation": "https://company.wiki/mlops/performance-runbook",
            "data_quality": "https://company.wiki/mlops/data-quality-runbook"
        }
        
        return runbook_map.get(alert_type, "https://company.wiki/mlops/general-runbook")
    
    def _get_dashboard_url(self, model_name):
        """Get monitoring dashboard URL for model"""
        
        if model_name:
            model_simple_name = model_name.split('.')[-1]
            return f"https://databricks.company.com/sql/dashboards/model-monitoring-{model_simple_name}"
        
        return "https://databricks.company.com/sql/dashboards/mlops-overview"
    
    def _send_slack_notification(self, alert_data):
        """Send Slack notification with rich formatting"""
        
        webhook_url = self.config.get("slack", {}).get("webhook_url") or \
                     dbutils.secrets.get("mlops", "slack_webhook")
        
        if not webhook_url:
            return {"success": False, "error": "Slack webhook URL not configured"}
        
        # Create rich Slack message
        message = {
            "text": f"{alert_data['emoji']} MLOps Alert: {alert_data.get('title', 'Model Issue Detected')}",
            "attachments": [
                {
                    "color": alert_data["color"],
                    "fields": [
                        {
                            "title": "Model",
                            "value": alert_data.get("model_name", "Unknown"),
                            "short": True
                        },
                        {
                            "title": "Priority",
                            "value": alert_data["priority"].upper(),
                            "short": True
                        },
                        {
                            "title": "Alert Type", 
                            "value": alert_data.get("alert_type", "Unknown"),
                            "short": True
                        },
                        {
                            "title": "Risk Score",
                            "value": f"{alert_data.get('risk_score', 0):.1f}/100",
                            "short": True
                        }
                    ],
                    "actions": [
                        {
                            "type": "button",
                            "text": "View Dashboard",
                            "url": alert_data["dashboard_url"]
                        },
                        {
                            "type": "button", 
                            "text": "View Runbook",
                            "url": alert_data["runbook_url"]
                        }
                    ],
                    "footer": "MLOps Monitoring System",
                    "ts": int(datetime.now().timestamp())
                }
            ]
        }
        
        # Add detailed information if available
        if alert_data.get("details"):
            message["attachments"][0]["text"] = alert_data["details"]
        
        response = requests.post(webhook_url, json=message)
        
        return {
            "success": response.status_code == 200,
            "status_code": response.status_code,
            "response": response.text
        }
    
    def _send_email_notification(self, alert_data):
        """Send detailed email notification"""
        
        email_config = self.config.get("email", {})
        
        if not email_config:
            return {"success": False, "error": "Email configuration not provided"}
        
        # Create email content
        subject = f"{alert_data['emoji']} MLOps Alert: {alert_data.get('title', 'Model Issue Detected')}"
        
        html_body = f"""
        <html>
        <body>
            <h2 style="color: {alert_data['color']};">MLOps Alert - {alert_data['priority'].upper()} Priority</h2>
            
            <h3>Alert Details</h3>
            <table border="1" style="border-collapse: collapse;">
                <tr><td><strong>Model</strong></td><td>{alert_data.get('model_name', 'Unknown')}</td></tr>
                <tr><td><strong>Alert Type</strong></td><td>{alert_data.get('alert_type', 'Unknown')}</td></tr>
                <tr><td><strong>Risk Score</strong></td><td>{alert_data.get('risk_score', 0):.1f}/100</td></tr>
                <tr><td><strong>Timestamp</strong></td><td>{alert_data['timestamp']}</td></tr>
            </table>
            
            <h3>Actions Required</h3>
            <ul>
                <li><a href="{alert_data['dashboard_url']}">View Monitoring Dashboard</a></li>
                <li><a href="{alert_data['runbook_url']}">Follow Runbook</a></li>
            </ul>
            
            <h3>Additional Details</h3>
            <p>{alert_data.get('details', 'No additional details available.')}</p>
            
            <hr>
            <p><em>This alert was generated by the MLOps Monitoring System</em></p>
        </body>
        </html>
        """
        
        # Send email (implementation depends on your email service)
        try:
            # Example implementation - replace with your email service
            msg = MIMEMultipart()
            msg['From'] = email_config.get('from_address')
            msg['To'] = ', '.join(email_config.get('recipients', []))
            msg['Subject'] = subject
            
            msg.attach(MIMEText(html_body, 'html'))
            
            # Note: Add your SMTP configuration here
            # server = smtplib.SMTP(email_config.get('smtp_server'))
            # server.send_message(msg)
            
            return {"success": True, "message": "Email sent successfully"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _send_teams_notification(self, alert_data):
        """Send Microsoft Teams notification"""
        
        webhook_url = self.config.get("teams", {}).get("webhook_url")
        
        if not webhook_url:
            return {"success": False, "error": "Teams webhook URL not configured"}
        
        message = {
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "themeColor": alert_data["color"],
            "summary": f"MLOps Alert: {alert_data.get('title', 'Model Issue')}",
            "sections": [
                {
                    "activityTitle": f"{alert_data['emoji']} MLOps Alert",
                    "activitySubtitle": alert_data.get('title', 'Model Issue Detected'),
                    "facts": [
                        {"name": "Model", "value": alert_data.get('model_name', 'Unknown')},
                        {"name": "Priority", "value": alert_data['priority'].upper()},
                        {"name": "Risk Score", "value": f"{alert_data.get('risk_score', 0):.1f}/100"}
                    ]
                }
            ],
            "potentialAction": [
                {
                    "@type": "OpenUri",
                    "name": "View Dashboard",
                    "targets": [{"os": "default", "uri": alert_data["dashboard_url"]}]
                }
            ]
        }
        
        response = requests.post(webhook_url, json=message)
        
        return {
            "success": response.status_code == 200,
            "status_code": response.status_code
        }
    
    def _send_pagerduty_notification(self, alert_data):
        """Send PagerDuty alert for critical issues"""
        
        pagerduty_config = self.config.get("pagerduty", {})
        
        if not pagerduty_config:
            return {"success": False, "error": "PagerDuty configuration not provided"}
        
        event = {
            "routing_key": pagerduty_config.get("integration_key"),
            "event_action": "trigger",
            "payload": {
                "summary": f"MLOps Alert: {alert_data.get('title', 'Model Issue')}",
                "source": alert_data.get('model_name', 'MLOps'),
                "severity": self._map_priority_to_pagerduty_severity(alert_data['priority']),
                "custom_details": {
                    "model_name": alert_data.get('model_name'),
                    "alert_type": alert_data.get('alert_type'),
                    "risk_score": alert_data.get('risk_score'),
                    "dashboard_url": alert_data['dashboard_url']
                }
            }
        }
        
        response = requests.post(
            "https://events.pagerduty.com/v2/enqueue",
            json=event,
            headers={"Content-Type": "application/json"}
        )
        
        return {
            "success": response.status_code == 202,
            "status_code": response.status_code
        }
    
    def _map_priority_to_pagerduty_severity(self, priority):
        """Map our priority to PagerDuty severity"""
        
        severity_map = {
            "low": "info",
            "medium": "warning", 
            "high": "error",
            "critical": "critical"
        }
        
        return severity_map.get(priority, "warning")
    
    def _send_webhook_notification(self, alert_data):
        """Send generic webhook notification"""
        
        webhook_config = self.config.get("webhook", {})
        
        if not webhook_config.get("url"):
            return {"success": False, "error": "Webhook URL not configured"}
        
        payload = {
            "alert_data": alert_data,
            "source": "mlops_monitoring",
            "version": "1.0"
        }
        
        response = requests.post(
            webhook_config["url"],
            json=payload,
            headers=webhook_config.get("headers", {}),
            timeout=30
        )
        
        return {
            "success": response.status_code in [200, 201, 202],
            "status_code": response.status_code
        }

# Usage
notification_manager = MLOpsNotificationManager({
    "slack": {"webhook_url": "https://hooks.slack.com/services/..."},
    "email": {
        "from_address": "mlops@company.com",
        "recipients": ["ml-team@company.com", "data-engineering@company.com"],
        "smtp_server": "smtp.company.com"
    },
    "teams": {"webhook_url": "https://outlook.office.com/webhook/..."},
    "pagerduty": {"integration_key": "your_pagerduty_key"}
})

# Send comprehensive alert
alert_data = {
    "title": "Model Performance Degradation Detected",
    "model_name": f"{catalog}.{db}.churn_model",
    "alert_type": "performance_degradation",
    "risk_score": 75.5,
    "details": "F1 score dropped below 0.65 threshold. Expected loss increased by 40%.",
    "affected_segments": ["senior_citizens", "fiber_optic_customers"]
}

results = notification_manager.send_comprehensive_alert(
    alert_data, 
    channels=["slack", "email"], 
    priority="high"
)

print("Notification Results:", results)
```

---

## Dashboard & Visualization Integration

### Automated Dashboard Management

```python
from databricks.sdk.service.sql import CreateDashboardRequest, Dashboard

class MLOpsDashboardManager:
    """Manage MLOps monitoring dashboards programmatically"""
    
    def __init__(self):
        self.w = WorkspaceClient()
    
    def create_comprehensive_monitoring_dashboard(self, model_name, monitor_info):
        """Create comprehensive monitoring dashboard for a model"""
        
        dashboard_config = {
            "name": f"MLOps Monitor - {model_name.split('.')[-1]}",
            "queries": self._generate_dashboard_queries(monitor_info),
            "layout": self._generate_dashboard_layout(),
            "tags": ["mlops", "monitoring", model_name.split('.')[-1]]
        }
        
        # Create dashboard
        dashboard = self._create_dashboard(dashboard_config)
        
        # Set up automated refresh
        self._setup_dashboard_refresh(dashboard.id)
        
        return dashboard
    
    def _generate_dashboard_queries(self, monitor_info):
        """Generate SQL queries for dashboard visualizations"""
        
        drift_table = monitor_info.drift_metrics_table_name
        profile_table = monitor_info.profile_metrics_table_name
        
        queries = {
            "model_performance_trend": f"""
            SELECT 
                window.start as date,
                f1_score.macro as f1_score,
                precision.macro as precision,
                recall.macro as recall,
                expected_loss,
                revenue_at_risk
            FROM {profile_table}
            WHERE 
                window.start >= date_sub(current_date(), 30)
                AND log_type = 'INPUT'
                AND column_name = ':table'
                AND slice_key is null
            ORDER BY window.start
            """,
            
            "drift_heatmap": f"""
            SELECT 
                window.start as date,
                column_name as feature,
                js_distance as drift_score
            FROM {drift_table}
            WHERE 
                window.start >= date_sub(current_date(), 30)
                AND column_name NOT IN ('prediction', 'churn')
                AND drift_type = 'CONSECUTIVE'
            """,
            
            "prediction_distribution": f"""
            SELECT 
                window.start as date,
                prediction_distribution.Yes as positive_predictions,
                prediction_distribution.No as negative_predictions
            FROM {profile_table}
            WHERE 
                window.start >= date_sub(current_date(), 7)
                AND log_type = 'INPUT'
                AND column_name = 'prediction'
            ORDER BY window.start
            """,
            
            "business_impact_summary": f"""
            SELECT 
                window.start as date,
                expected_loss,
                false_positive_cost,
                revenue_at_risk,
                high_value_accuracy
            FROM {profile_table}
            WHERE 
                window.start >= date_sub(current_date(), 30)
                AND log_type = 'INPUT'
                AND column_name = ':table'
                AND slice_key is null
            ORDER BY window.start
            """,
            
            "alert_summary": f"""
            SELECT 
                'Feature Drift' as alert_type,
                count(*) as violation_count,
                max(js_distance) as max_severity
            FROM {drift_table}
            WHERE 
                window.start >= date_sub(current_date(), 7)
                AND js_distance > 0.2
                AND column_name NOT IN ('prediction', 'churn')
            
            UNION ALL
            
            SELECT 
                'Performance Degradation' as alert_type,
                count(*) as violation_count,
                min(f1_score.macro) as max_severity
            FROM {profile_table}
            WHERE 
                window.start >= date_sub(current_date(), 7)
                AND f1_score.macro < 0.65
                AND column_name = ':table'
            """
        }
        
        return queries
    
    def _generate_dashboard_layout(self):
        """Generate dashboard layout configuration"""
        
        return {
            "sections": [
                {
                    "title": "Model Performance Overview",
                    "widgets": ["model_performance_trend", "business_impact_summary"]
                },
                {
                    "title": "Drift Detection",
                    "widgets": ["drift_heatmap", "prediction_distribution"]
                },
                {
                    "title": "Alert Summary",
                    "widgets": ["alert_summary"]
                }
            ]
        }
    
    def _create_dashboard(self, config):
        """Create dashboard using Databricks SQL API"""
        
        # Note: Actual implementation would use Databricks SQL API
        # This is a simplified example
        dashboard_request = CreateDashboardRequest(
            name=config["name"],
            tags=config["tags"]
        )
        
        # dashboard = self.w.dashboards.create(dashboard_request)
        
        # For now, return mock dashboard object
        return type('Dashboard', (), {
            'id': 'dashboard_123',
            'name': config["name"],
            'url': f"https://databricks.company.com/sql/dashboards/dashboard_123"
        })()
    
    def _setup_dashboard_refresh(self, dashboard_id, refresh_interval_hours=1):
        """Setup automated dashboard refresh"""
        
        # Note: Implementation would depend on your scheduling system
        print(f"Setting up {refresh_interval_hours}h refresh for dashboard {dashboard_id}")
    
    def create_executive_summary_dashboard(self, models_list):
        """Create executive summary dashboard for multiple models"""
        
        summary_queries = {
            "models_health_overview": f"""
            SELECT 
                model_name,
                avg(f1_score.macro) as avg_f1_score,
                avg(expected_loss) as avg_expected_loss,
                count(*) as monitoring_points
            FROM (
                {' UNION ALL '.join([
                    f"SELECT '{model}' as model_name, f1_score.macro, expected_loss FROM {model}_profile_metrics WHERE window.start >= date_sub(current_date(), 7)"
                    for model in models_list
                ])}
            )
            GROUP BY model_name
            """,
            
            "risk_distribution": f"""
            SELECT 
                CASE 
                    WHEN f1_score.macro >= 0.8 THEN 'Low Risk'
                    WHEN f1_score.macro >= 0.65 THEN 'Medium Risk'
                    ELSE 'High Risk'
                END as risk_level,
                count(*) as model_count
            FROM (
                {' UNION ALL '.join([
                    f"SELECT f1_score.macro FROM {model}_profile_metrics WHERE window.start >= date_sub(current_date(), 1)"
                    for model in models_list
                ])}
            )
            GROUP BY risk_level
            """
        }
        
        return summary_queries

# Usage
dashboard_manager = MLOpsDashboardManager()

# Create model-specific dashboard
dashboard = dashboard_manager.create_comprehensive_monitoring_dashboard(
    model_name=f"{catalog}.{db}.churn_model",
    monitor_info=monitor_info
)

print(f"Dashboard created: {dashboard.url}")
```

---

## Automated Response Workflows

### Intelligent Workflow Orchestration

```python
class MLOpsResponseOrchestrator:
    """Orchestrate automated responses to monitoring alerts"""
    
    def __init__(self, notification_manager):
        self.notification_manager = notification_manager
        self.w = WorkspaceClient()
    
    def create_intelligent_response_workflow(self, model_name, response_config=None):
        """
        Create intelligent response workflow based on alert types and severity
        
        Args:
            model_name: Model being monitored
            response_config: Custom response configuration
        """
        
        if response_config is None:
            response_config = self._get_default_response_config()
        
        workflow_config = {
            "name": f"MLOps Response - {model_name.split('.')[-1]}",
            "schedule": "0 */6 * * *",  # Every 6 hours
            "tasks": [
                self._create_monitoring_assessment_task(),
                self._create_risk_evaluation_task(),
                self._create_decision_matrix_task(),
                self._create_automated_remediation_task(),
                self._create_escalation_task(),
                self._create_documentation_task()
            ]
        }
        
        return workflow_config
    
    def _get_default_response_config(self):
        """Get default response configuration"""
        
        return {
            "thresholds": {
                "auto_remediation": {
                    "risk_score": 30,
                    "confidence": 0.8
                },
                "human_escalation": {
                    "risk_score": 60,
                    "business_impact": 50000
                },
                "emergency_response": {
                    "risk_score": 80,
                    "critical_feature_drift": 0.5
                }
            },
            "remediation_actions": {
                "feature_drift": ["retrain_model", "update_features", "adjust_thresholds"],
                "performance_degradation": ["increase_monitoring", "rollback_model", "retrain_model"],
                "data_quality": ["pause_predictions", "alert_data_team", "use_backup_model"]
            }
        }
    
    def _create_monitoring_assessment_task(self):
        """Create task to assess current monitoring state"""
        
        return {
            "task_key": "assess_monitoring_state",
            "notebook_task": {
                "notebook_path": "/mlops/monitoring/assess_state",
                "parameters": {
                    "time_window_hours": "6",
                    "include_business_metrics": "true"
                }
            },
            "depends_on": []
        }
    
    def _create_risk_evaluation_task(self):
        """Create task to evaluate overall risk"""
        
        return {
            "task_key": "evaluate_risk",
            "notebook_task": {
                "notebook_path": "/mlops/monitoring/risk_evaluation",
                "parameters": {
                    "risk_model_version": "v2.1",
                    "include_external_factors": "true"
                }
            },
            "depends_on": [{"task_key": "assess_monitoring_state"}]
        }
    
    def _create_decision_matrix_task(self):
        """Create decision matrix for automated responses"""
        
        return {
            "task_key": "decision_matrix",
            "condition_task": {
                "op": "GREATER_THAN",
                "left": "{{tasks.evaluate_risk.values.overall_risk_score}}",
                "right": "30"
            },
            "depends_on": [{"task_key": "evaluate_risk"}]
        }
    
    def _create_automated_remediation_task(self):
        """Create automated remediation task"""
        
        return {
            "task_key": "automated_remediation",
            "notebook_task": {
                "notebook_path": "/mlops/responses/automated_remediation",
                "parameters": {
                    "remediation_mode": "{{tasks.evaluate_risk.values.recommended_action}}",
                    "safety_checks": "true"
                }
            },
            "depends_on": [{"task_key": "decision_matrix"}]
        }
    
    def _create_escalation_task(self):
        """Create escalation task for human intervention"""
        
        return {
            "task_key": "escalate_to_humans",
            "condition_task": {
                "op": "GREATER_THAN",
                "left": "{{tasks.evaluate_risk.values.overall_risk_score}}",
                "right": "60"
            },
            "depends_on": [{"task_key": "automated_remediation"}]
        }
    
    def _create_documentation_task(self):
        """Create task to document actions taken"""
        
        return {
            "task_key": "document_actions",
            "notebook_task": {
                "notebook_path": "/mlops/monitoring/document_response",
                "parameters": {
                    "include_metrics": "true",
                    "update_runbook": "true"
                }
            },
            "depends_on": [{"task_key": "escalate_to_humans"}]
        }
    
    def execute_emergency_response(self, alert_data):
        """Execute emergency response for critical alerts"""
        
        emergency_actions = []
        
        # Immediate model safety measures
        if alert_data.get("risk_score", 0) >= 90:
            emergency_actions.append(self._pause_model_serving(alert_data["model_name"]))
            emergency_actions.append(self._activate_backup_model(alert_data["model_name"]))
        
        # Immediate notifications
        emergency_actions.append(
            self.notification_manager.send_comprehensive_alert(
                alert_data,
                channels=["pagerduty", "slack", "teams"],
                priority="critical"
            )
        )
        
        # Emergency escalation
        emergency_actions.append(self._trigger_emergency_escalation(alert_data))
        
        return emergency_actions
    
    def _pause_model_serving(self, model_name):
        """Pause model serving endpoint"""
        
        try:
            # Find serving endpoint for model
            endpoints = self.w.serving_endpoints.list()
            
            for endpoint in endpoints:
                if model_name.split('.')[-1] in endpoint.name:
                    # Pause endpoint by updating traffic to 0%
                    self.w.serving_endpoints.update_config(
                        name=endpoint.name,
                        traffic_config={"routes": []}
                    )
                    
                    return {"action": "pause_serving", "status": "success", "endpoint": endpoint.name}
            
            return {"action": "pause_serving", "status": "no_endpoint_found"}
            
        except Exception as e:
            return {"action": "pause_serving", "status": "error", "error": str(e)}
    
    def _activate_backup_model(self, model_name):
        """Activate backup model if available"""
        
        try:
            # Look for backup model (e.g., previous Champion)
            backup_model_name = f"{model_name}_backup"
            
            # Implementation would depend on your backup strategy
            return {"action": "activate_backup", "status": "activated", "backup_model": backup_model_name}
            
        except Exception as e:
            return {"action": "activate_backup", "status": "error", "error": str(e)}
    
    def _trigger_emergency_escalation(self, alert_data):
        """Trigger emergency escalation procedures"""
        
        escalation_data = {
            **alert_data,
            "escalation_level": "EMERGENCY",
            "required_response_time": "15 minutes",
            "escalation_contacts": [
                "ml-lead@company.com",
                "cto@company.com",
                "data-engineering-lead@company.com"
            ]
        }
        
        return {
            "action": "emergency_escalation",
            "status": "triggered",
            "escalation_id": f"ESC-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        }

# Usage
response_orchestrator = MLOpsResponseOrchestrator(notification_manager)

# Create intelligent response workflow
workflow_config = response_orchestrator.create_intelligent_response_workflow(
    model_name=f"{catalog}.{db}.churn_model"
)

print("Intelligent Response Workflow Created")
print(json.dumps(workflow_config, indent=2))

# Execute emergency response for critical alert
emergency_alert = {
    "title": "Critical Model Failure",
    "model_name": f"{catalog}.{db}.churn_model",
    "alert_type": "critical_failure",
    "risk_score": 95,
    "details": "Model serving error rate > 50%, immediate intervention required"
}

emergency_response = response_orchestrator.execute_emergency_response(emergency_alert)
print("Emergency Response Executed:", emergency_response)
```

---

## Performance Monitoring

### Advanced Performance Tracking

```python
class PerformanceMonitor:
    """Advanced performance monitoring with trend analysis and forecasting"""
    
    def __init__(self, monitor_table_name):
        self.monitor_table_name = monitor_table_name
        self.w = WorkspaceClient()
        
        monitor_info = self.w.quality_monitors.get(table_name=monitor_table_name)
        self.profile_table_name = monitor_info.profile_metrics_table_name
    
    def analyze_performance_trends(self, time_window_days=30, forecast_days=7):
        """
        Analyze performance trends and forecast future performance
        
        Args:
            time_window_days: Historical window for trend analysis
            forecast_days: Number of days to forecast
        """
        
        # Get historical performance data
        historical_data = self._get_historical_performance(time_window_days)
        
        # Analyze trends
        trend_analysis = self._analyze_trends(historical_data)
        
        # Generate forecasts
        forecasts = self._generate_performance_forecasts(historical_data, forecast_days)
        
        # Identify concerning patterns
        alerts = self._identify_concerning_patterns(trend_analysis, forecasts)
        
        return {
            "historical_data": historical_data,
            "trend_analysis": trend_analysis,
            "forecasts": forecasts,
            "alerts": alerts,
            "summary": self._generate_performance_summary(trend_analysis, forecasts)
        }
    
    def _get_historical_performance(self, time_window_days):
        """Get historical performance metrics"""
        
        query = f"""
        SELECT 
            window.start as date,
            f1_score.macro as f1_score,
            precision.macro as precision,
            recall.macro as recall,
            accuracy as accuracy,
            expected_loss,
            revenue_at_risk,
            false_positive_cost,
            high_value_accuracy
        FROM {self.profile_table_name}
        WHERE 
            window.start >= date_sub(current_date(), {time_window_days})
            AND log_type = 'INPUT'
            AND column_name = ':table'
            AND slice_key is null
        ORDER BY window.start
        """
        
        return spark.sql(query).toPandas()
    
    def _analyze_trends(self, data):
        """Analyze performance trends"""
        import numpy as np
        from scipy import stats
        
        trends = {}
        
        for metric in ['f1_score', 'precision', 'recall', 'accuracy']:
            if metric in data.columns:
                values = data[metric].dropna()
                
                if len(values) > 1:
                    # Calculate trend slope
                    x = np.arange(len(values))
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
                    
                    trends[metric] = {
                        "slope": slope,
                        "direction": "improving" if slope > 0 else "degrading",
                        "correlation": r_value,
                        "significance": p_value,
                        "current_value": values.iloc[-1],
                        "trend_strength": abs(r_value)
                    }
        
        return trends
    
    def _generate_performance_forecasts(self, data, forecast_days):
        """Generate performance forecasts using simple trend extrapolation"""
        import numpy as np
        
        forecasts = {}
        
        for metric in ['f1_score', 'precision', 'recall', 'accuracy']:
            if metric in data.columns:
                values = data[metric].dropna()
                
                if len(values) > 2:
                    # Simple linear trend extrapolation
                    x = np.arange(len(values))
                    trend_line = np.polyfit(x, values, 1)
                    
                    # Forecast future values
                    future_x = np.arange(len(values), len(values) + forecast_days)
                    forecast_values = np.polyval(trend_line, future_x)
                    
                    forecasts[metric] = {
                        "forecast_values": forecast_values.tolist(),
                        "confidence": "medium",  # Would implement proper confidence intervals
                        "trend_continuation": np.polyval(trend_line, len(values) + forecast_days - 1)
                    }
        
        return forecasts
    
    def _identify_concerning_patterns(self, trends, forecasts):
        """Identify concerning performance patterns"""
        
        alerts = []
        
        for metric, trend_info in trends.items():
            # Check for significant degradation
            if (trend_info["direction"] == "degrading" and 
                trend_info["trend_strength"] > 0.5 and 
                trend_info["significance"] < 0.05):
                
                alerts.append({
                    "type": "performance_degradation",
                    "metric": metric,
                    "severity": "high" if trend_info["trend_strength"] > 0.8 else "medium",
                    "description": f"{metric} showing significant degradation (slope: {trend_info['slope']:.4f})",
                    "current_value": trend_info["current_value"]
                })
            
            # Check forecast predictions
            if metric in forecasts:
                forecast_end = forecasts[metric]["trend_continuation"]
                
                # Define minimum acceptable thresholds
                thresholds = {
                    "f1_score": 0.65,
                    "precision": 0.60,
                    "recall": 0.60,
                    "accuracy": 0.70
                }
                
                if metric in thresholds and forecast_end < thresholds[metric]:
                    alerts.append({
                        "type": "forecast_alert",
                        "metric": metric,
                        "severity": "medium",
                        "description": f"{metric} forecast to drop below threshold ({thresholds[metric]:.2f})",
                        "forecast_value": forecast_end
                    })
        
        return alerts
    
    def _generate_performance_summary(self, trends, forecasts):
        """Generate performance summary"""
        
        total_metrics = len(trends)
        degrading_metrics = sum(1 for t in trends.values() if t["direction"] == "degrading")
        
        summary = {
            "overall_health": "good" if degrading_metrics / total_metrics < 0.3 else "concerning",
            "degrading_metrics_count": degrading_metrics,
            "total_metrics_count": total_metrics,
            "key_concerns": [
                metric for metric, trend in trends.items() 
                if trend["direction"] == "degrading" and trend["trend_strength"] > 0.6
            ]
        }
        
        return summary

# Usage
perf_monitor = PerformanceMonitor(f"{catalog}.{db}.churn_inference_table")

performance_analysis = perf_monitor.analyze_performance_trends(
    time_window_days=30,
    forecast_days=7
)

print("Performance Analysis Summary:")
print(f"Overall Health: {performance_analysis['summary']['overall_health']}")
print(f"Key Concerns: {performance_analysis['summary']['key_concerns']}")

for alert in performance_analysis['alerts']:
    print(f"⚠️ {alert['type']}: {alert['description']}")
```

---

## Compliance & Audit Monitoring

### Regulatory Compliance Tracking

```python
class ComplianceMonitor:
    """Monitor model compliance with regulatory requirements"""
    
    def __init__(self, model_name, compliance_requirements=None):
        self.model_name = model_name
        self.compliance_requirements = compliance_requirements or self._get_default_requirements()
        self.w = WorkspaceClient()
    
    def _get_default_requirements(self):
        """Get default compliance requirements"""
        
        return {
            "model_documentation": {
                "required": True,
                "check_frequency": "monthly"
            },
            "bias_monitoring": {
                "required": True,
                "protected_attributes": ["senior_citizen", "gender"],
                "fairness_threshold": 0.8
            },
            "explainability": {
                "required": True,
                "methods": ["shap", "lime"],
                "coverage_threshold": 0.95
            },
            "data_lineage": {
                "required": True,
                "documentation_current": True
            },
            "audit_logging": {
                "required": True,
                "retention_days": 365
            }
        }
    
    def run_compliance_check(self):
        """Run comprehensive compliance check"""
        
        compliance_results = {}
        
        for requirement, config in self.compliance_requirements.items():
            if config.get("required", False):
                check_method = getattr(self, f"_check_{requirement}", None)
                
                if check_method:
                    compliance_results[requirement] = check_method(config)
                else:
                    compliance_results[requirement] = {
                        "status": "not_implemented",
                        "message": f"Check for {requirement} not implemented"
                    }
        
        # Generate compliance report
        compliance_report = self._generate_compliance_report(compliance_results)
        
        return compliance_report
    
    def _check_model_documentation(self, config):
        """Check model documentation compliance"""
        
        try:
            # Check if model has proper documentation
            client = MlflowClient()
            model_details = client.get_registered_model(self.model_name)
            
            checks = {
                "has_description": bool(model_details.description),
                "description_length": len(model_details.description or "") > 100,
                "has_tags": len(model_details.tags) > 0,
                "version_descriptions": True  # Would check all versions
            }
            
            compliance_score = sum(checks.values()) / len(checks)
            
            return {
                "status": "compliant" if compliance_score >= 0.8 else "non_compliant",
                "score": compliance_score,
                "details": checks,
                "recommendations": self._get_documentation_recommendations(checks)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
    
    def _check_bias_monitoring(self, config):
        """Check bias and fairness monitoring"""
        
        protected_attributes = config.get("protected_attributes", [])
        fairness_threshold = config.get("fairness_threshold", 0.8)
        
        bias_results = {}
        
        for attribute in protected_attributes:
            # Check fairness metrics for protected attribute
            fairness_query = f"""
            SELECT 
                {attribute},
                count(*) as total_predictions,
                avg(CASE WHEN prediction = 'Yes' THEN 1.0 ELSE 0.0 END) as positive_rate,
                avg(CASE WHEN prediction = label THEN 1.0 ELSE 0.0 END) as accuracy
            FROM {self.model_name.replace('.', '_')}_inference_table
            WHERE window.start >= date_sub(current_date(), 30)
            GROUP BY {attribute}
            """
            
            try:
                fairness_data = spark.sql(fairness_query).toPandas()
                
                # Calculate demographic parity
                if len(fairness_data) >= 2:
                    parity_ratio = fairness_data['positive_rate'].min() / fairness_data['positive_rate'].max()
                    
                    bias_results[attribute] = {
                        "demographic_parity": parity_ratio,
                        "passes_threshold": parity_ratio >= fairness_threshold,
                        "groups_data": fairness_data.to_dict('records')
                    }
                
            except Exception as e:
                bias_results[attribute] = {
                    "status": "error",
                    "message": str(e)
                }
        
        overall_compliance = all(
            result.get("passes_threshold", False) 
            for result in bias_results.values() 
            if "passes_threshold" in result
        )
        
        return {
            "status": "compliant" if overall_compliance else "non_compliant",
            "details": bias_results,
            "recommendations": self._get_bias_recommendations(bias_results)
        }
    
    def _check_explainability(self, config):
        """Check model explainability compliance"""
        
        # Check if explainability artifacts exist
        try:
            # Look for SHAP/LIME artifacts in model registry
            client = MlflowClient()
            latest_version = client.get_latest_versions(self.model_name)[0]
            
            run = client.get_run(latest_version.run_id)
            artifacts = client.list_artifacts(latest_version.run_id)
            
            explainability_artifacts = [
                artifact for artifact in artifacts 
                if any(method in artifact.path.lower() for method in config.get("methods", []))
            ]
            
            return {
                "status": "compliant" if len(explainability_artifacts) > 0 else "non_compliant",
                "artifacts_found": len(explainability_artifacts),
                "artifact_paths": [a.path for a in explainability_artifacts],
                "recommendations": [] if len(explainability_artifacts) > 0 else [
                    "Generate SHAP explanations for model predictions",
                    "Add LIME explanations for complex cases",
                    "Document explanation methodology"
                ]
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
    
    def _check_data_lineage(self, config):
        """Check data lineage documentation"""
        
        # Check Unity Catalog lineage
        try:
            # Would check actual lineage information
            # This is a simplified check
            
            lineage_checks = {
                "source_tables_documented": True,  # Would check actual lineage
                "feature_transformations_documented": True,
                "model_dependencies_tracked": True
            }
            
            compliance_score = sum(lineage_checks.values()) / len(lineage_checks)
            
            return {
                "status": "compliant" if compliance_score == 1.0 else "non_compliant",
                "score": compliance_score,
                "details": lineage_checks
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
    
    def _check_audit_logging(self, config):
        """Check audit logging compliance"""
        
        retention_days = config.get("retention_days", 365)
        
        # Check if inference logging is enabled and configured properly
        try:
            # Check for inference tables and their retention
            inference_tables = [
                f"{self.model_name.replace('.', '_')}_inference_table",
                f"{self.model_name.replace('.', '_')}_served_requests"
            ]
            
            audit_results = {}
            
            for table in inference_tables:
                try:
                    table_info = spark.sql(f"DESCRIBE DETAIL {table}").collect()[0]
                    
                    audit_results[table] = {
                        "exists": True,
                        "location": table_info["location"],
                        "retention_configured": True  # Would check actual retention settings
                    }
                    
                except Exception:
                    audit_results[table] = {"exists": False}
            
            has_logging = any(result.get("exists", False) for result in audit_results.values())
            
            return {
                "status": "compliant" if has_logging else "non_compliant",
                "details": audit_results,
                "recommendations": [] if has_logging else [
                    "Enable inference logging on serving endpoints",
                    "Configure appropriate data retention policies",
                    "Set up audit log monitoring"
                ]
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
    
    def _generate_compliance_report(self, compliance_results):
        """Generate comprehensive compliance report"""
        
        total_checks = len(compliance_results)
        compliant_checks = sum(
            1 for result in compliance_results.values() 
            if result.get("status") == "compliant"
        )
        
        compliance_score = compliant_checks / total_checks if total_checks > 0 else 0
        
        # Determine overall compliance level
        if compliance_score >= 0.9:
            compliance_level = "FULLY_COMPLIANT"
        elif compliance_score >= 0.7:
            compliance_level = "MOSTLY_COMPLIANT"
        elif compliance_score >= 0.5:
            compliance_level = "PARTIALLY_COMPLIANT"
        else:
            compliance_level = "NON_COMPLIANT"
        
        # Collect all recommendations
        all_recommendations = []
        for result in compliance_results.values():
            if "recommendations" in result:
                all_recommendations.extend(result["recommendations"])
        
        return {
            "model_name": self.model_name,
            "compliance_level": compliance_level,
            "compliance_score": compliance_score,
            "total_checks": total_checks,
            "compliant_checks": compliant_checks,
            "detailed_results": compliance_results,
            "recommendations": all_recommendations,
            "report_timestamp": datetime.now().isoformat(),
            "next_review_date": (datetime.now() + pd.Timedelta(days=30)).isoformat()
        }
    
    def _get_documentation_recommendations(self, checks):
        """Get documentation improvement recommendations"""
        
        recommendations = []
        
        if not checks.get("has_description"):
            recommendations.append("Add comprehensive model description")
        
        if not checks.get("description_length"):
            recommendations.append("Expand model description to include use case, assumptions, and limitations")
        
        if not checks.get("has_tags"):
            recommendations.append("Add relevant tags for model categorization and discovery")
        
        return recommendations
    
    def _get_bias_recommendations(self, bias_results):
        """Get bias mitigation recommendations"""
        
        recommendations = []
        
        for attribute, result in bias_results.items():
            if not result.get("passes_threshold", True):
                recommendations.extend([
                    f"Investigate bias in {attribute} predictions",
                    f"Consider rebalancing training data for {attribute}",
                    f"Implement bias mitigation techniques for {attribute}"
                ])
        
        return recommendations

# Usage
compliance_monitor = ComplianceMonitor(
    model_name=f"{catalog}.{db}.churn_model",
    compliance_requirements={
        "model_documentation": {"required": True},
        "bias_monitoring": {
            "required": True,
            "protected_attributes": ["senior_citizen"],
            "fairness_threshold": 0.8
        },
        "audit_logging": {"required": True, "retention_days": 365}
    }
)

compliance_report = compliance_monitor.run_compliance_check()

print(f"Compliance Level: {compliance_report['compliance_level']}")
print(f"Compliance Score: {compliance_report['compliance_score']:.2f}")
print(f"Recommendations: {len(compliance_report['recommendations'])}")

for rec in compliance_report['recommendations'][:3]:
    print(f"  • {rec}")
```

---

## Usage Examples

### Complete Monitoring Setup

```python
# 1. Setup comprehensive monitoring
monitor_manager = LakehouseMonitorManager(catalog, db)

monitor_info = monitor_manager.create_comprehensive_monitor(
    table_name=f"{catalog}.{db}.churn_inference_table",
    baseline_table_name=f"{catalog}.{db}.churn_baseline"
)

# 2. Setup drift detection
detector = AdvancedDriftDetector(f"{catalog}.{db}.churn_inference_table")

# 3. Setup notification system
notification_manager = MLOpsNotificationManager({
    "slack": {"webhook_url": "https://hooks.slack.com/services/..."},
    "email": {"recipients": ["ml-team@company.com"]}
})

# 4. Setup automated response
response_orchestrator = MLOpsResponseOrchestrator(notification_manager)

# 5. Run comprehensive analysis
drift_analysis = detector.detect_comprehensive_drift(time_window_days=7)

if drift_analysis["overall_risk_score"] > 50:
    alert_data = {
        "title": "Model Drift Detected",
        "model_name": f"{catalog}.{db}.churn_model",
        "alert_type": "drift_detection",
        "risk_score": drift_analysis["overall_risk_score"],
        "severity_level": drift_analysis["severity_level"],
        "details": f"Risk score: {drift_analysis['overall_risk_score']:.1f}/100"
    }
    
    # Send notifications
    notification_results = notification_manager.send_comprehensive_alert(
        alert_data, 
        priority="high"
    )
    
    # Execute automated response if critical
    if drift_analysis["overall_risk_score"] > 80:
        emergency_response = response_orchestrator.execute_emergency_response(alert_data)

# 6. Run compliance check
compliance_monitor = ComplianceMonitor(f"{catalog}.{db}.churn_model")
compliance_report = compliance_monitor.run_compliance_check()

print("✅ Complete MLOps monitoring setup deployed!")
```

This provides a comprehensive framework for production-grade MLOps monitoring, alerting, and automated response systems using Databricks.