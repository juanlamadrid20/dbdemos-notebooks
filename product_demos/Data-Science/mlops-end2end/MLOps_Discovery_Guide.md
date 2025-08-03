# MLOps Discovery Guide: Questions for Data Science & ML Engineering Teams

This guide provides structured discovery questions to understand your current MLOps practices, challenges, and opportunities for improvement. Use these questions to facilitate productive conversations with data science and ML engineering teams regardless of their current platform.

## 🎯 How to Use This Guide

**For Sales/Solutions Engineers:**
- Use sections relevant to the customer's maturity level
- Focus on pain points and desired outcomes
- Reference our best practices guides for specific solutions

**For Customer Teams:**
- Self-assess your current state honestly
- Identify gaps and improvement opportunities
- Prioritize areas with highest business impact

**Format:** Each section includes Current State questions, Pain Points, and Desired Future State to create a comprehensive picture.

---

## 📊 Feature Engineering & Data Preparation

### Current State Assessment

**Data Sources & Integration:**
- What data sources do you currently use for ML features? (databases, APIs, streaming, files)
- How do you handle data from multiple systems with different schemas or formats?
- What's your typical data volume and velocity? (GB/TB/PB, batch/streaming/real-time)
- How do you ensure data quality and validate incoming data?

**Feature Development Process:**
- How do data scientists currently create and test new features?
- What tools do you use for feature engineering? (pandas, Spark, dbt, custom code)
- How long does it typically take to go from raw data to ML-ready features?
- Do you have standardized feature engineering patterns or libraries?

**Feature Reusability & Governance:**
- How do you share features across different ML projects and teams?
- Do you have a centralized feature store or feature catalog?
- How do you handle feature versioning and lineage tracking?
- How do you prevent feature leakage between training and inference?

### Pain Points & Challenges

**🔍 Discovery Questions:**
- What percentage of your data science time is spent on data preparation vs. model development?
- How often do you discover data quality issues late in the ML pipeline?
- Do different teams recreate similar features because they can't find existing ones?
- How do you handle feature drift between training and production environments?
- What happens when a critical data source schema changes unexpectedly?

**⚠️ Common Pain Indicators:**
- "Our data scientists spend 80% of their time on data wrangling"
- "We keep rebuilding the same features for different projects"
- "Production models fail because features aren't available at inference time"
- "We don't know which features are actually being used in production"

### Desired Future State

**🎯 Vision Questions:**
- How would you like data scientists to discover and reuse existing features?
- What would an ideal feature development workflow look like?
- How important is real-time feature computation for your use cases?
- What level of feature governance and compliance do you need?

**📈 Success Metrics:**
- Reduce feature development time from weeks to days/hours
- Increase feature reusability across projects by X%
- Achieve <X% feature drift between training and production
- Enable real-time feature serving with <Xms latency

---

## 🤖 Model Training & Experimentation

### Current State Assessment

**Experimentation Platform:**
- What tools do you use for ML experimentation? (MLflow, Weights & Biases, custom solutions)
- How do you track model parameters, metrics, and artifacts?
- How do you compare experiments across different data scientists?
- Do you have standardized evaluation metrics and practices?

**Training Infrastructure:**
- Where do you train your models? (local, cloud VMs, managed services, on-premise)
- How do you handle compute scaling for large datasets or hyperparameter tuning?
- What ML frameworks and libraries do you primarily use?
- How do you manage training environments and dependencies?

**AutoML & Acceleration:**
- Do you use any AutoML tools or automated feature engineering?
- How do you handle hyperparameter optimization? (manual, grid search, Bayesian optimization)
- What's your approach to model selection and ensemble methods?
- How do you accelerate model development for new use cases?

### Pain Points & Challenges

**🔍 Discovery Questions:**
- How long does it take to go from business problem to first model prototype?
- How do you ensure reproducibility of training experiments?
- What happens when different data scientists use different evaluation approaches?
- How do you prevent overfitting and ensure model generalization?
- How do you handle training on sensitive or regulated data?

**⚠️ Common Pain Indicators:**
- "We can't reproduce results from 6 months ago"
- "Each data scientist has their own way of evaluating models"
- "Hyperparameter tuning takes weeks and burns through our compute budget"
- "We waste time rebuilding models that someone else already created"

### Desired Future State

**🎯 Vision Questions:**
- How would you like to standardize model development across your team?
- What level of automation do you want in your training pipeline?
- How important is it to have one platform for all ML frameworks?
- What would success look like for accelerating time-to-first-model?

**📈 Success Metrics:**
- Reduce time from business problem to prototype from X weeks to Y days
- Achieve 100% experiment reproducibility
- Increase model development velocity by X%
- Standardize evaluation metrics across 100% of projects

---

## ✅ Model Validation & Testing

### Current State Assessment

**Validation Frameworks:**
- How do you validate model performance before production deployment?
- What metrics do you use beyond accuracy? (fairness, explainability, business metrics)
- How do you test model behavior on edge cases and adversarial inputs?
- Do you have automated model validation pipelines?

**Business Validation:**
- How do you measure the business impact of your models?
- Who decides when a model is "good enough" for production?
- How do you validate models against regulatory or compliance requirements?
- What's your process for A/B testing models in production?

**Model Governance:**
- How do you ensure model explainability and interpretability?
- What documentation do you maintain for model decisions?
- How do you handle model bias detection and mitigation?
- Who is accountable for model performance in production?

### Pain Points & Challenges

**🔍 Discovery Questions:**
- How many models fail validation and need to be rebuilt?
- How long does your model validation process take?
- How do you catch issues that only appear in production?
- What happens when stakeholders disagree on model readiness?
- How do you validate models for regulated industries or sensitive decisions?

**⚠️ Common Pain Indicators:**
- "Models that look good in development fail in production"
- "We don't have consistent validation criteria across projects"
- "Stakeholders don't trust our models because they can't explain decisions"
- "We discovered bias in production that we didn't catch during development"

### Desired Future State

**🎯 Vision Questions:**
- What would comprehensive automated model validation look like?
- How would you like to demonstrate model fairness and compliance?
- What level of model explainability do different stakeholders need?
- How should business metrics be integrated into technical validation?

**📈 Success Metrics:**
- Achieve <X% of models failing production validation
- Reduce model validation time from X weeks to Y days
- Implement 100% automated bias detection
- Achieve regulatory compliance for 100% of applicable models

---

## 🚀 Model Deployment & Serving

### Current State Assessment

**Deployment Infrastructure:**
- How do you currently deploy models to production? (APIs, batch jobs, embedded)
- What platforms do you use for model serving? (cloud services, Kubernetes, serverless)
- How do you handle different inference patterns? (real-time, batch, streaming)
- What's your approach to model versioning and rollback strategies?

**Serving Architecture:**
- How do you handle model dependencies and environment management?
- What's your strategy for scaling model serving based on demand?
- How do you serve models that require real-time feature computation?
- How do you handle multi-model serving and A/B testing?

**Operational Excellence:**
- How do you monitor model serving performance and availability?
- What's your approach to cost optimization for model serving?
- How do you handle model updates and blue-green deployments?
- What SLAs do you maintain for model inference?

### Pain Points & Challenges

**🔍 Discovery Questions:**
- How long does it take to deploy a model from development to production?
- How often do model deployments fail or require rollbacks?
- What percentage of your infrastructure costs are related to model serving?
- How do you handle traffic spikes or unexpected load increases?
- What happens when you need to update a model that's being used by other systems?

**⚠️ Common Pain Indicators:**
- "It takes weeks to deploy a model that took days to develop"
- "Our model serving costs are unpredictable and often excessive"
- "We have different deployment processes for different types of models"
- "Models work in development but fail due to production environment differences"

### Desired Future State

**🎯 Vision Questions:**
- What would seamless model deployment from development to production look like?
- How would you like to handle traffic management and A/B testing?
- What level of automation do you want in your deployment pipeline?
- How important is multi-cloud or hybrid deployment capability?

**📈 Success Metrics:**
- Reduce model deployment time from X weeks to Y hours
- Achieve 99.9%+ model serving uptime
- Reduce model serving costs by X%
- Enable automated rollback within X minutes of issues

---

## 📈 Model Monitoring & Observability

### Current State Assessment

**Performance Monitoring:**
- How do you monitor model accuracy and performance in production?
- What tools do you use for model observability? (custom dashboards, APM tools)
- How do you detect when model performance degrades?
- How do you track business metrics and model ROI?

**Data & Drift Monitoring:**
- How do you monitor for data drift and distribution changes?
- How do you detect when input features change unexpectedly?
- What's your approach to concept drift and model staleness?
- How do you monitor for data quality issues in production?

**Alerting & Response:**
- What alerts do you have in place for model issues?
- Who gets notified when models underperform or fail?
- What's your incident response process for model problems?
- How do you decide when to retrain or replace a model?

### Pain Points & Challenges

**🔍 Discovery Questions:**
- How often do you discover model issues after they've impacted the business?
- How long does it take to identify the root cause of model performance problems?
- How do you distinguish between model issues and data pipeline issues?
- What percentage of model issues are caught proactively vs. reactively?
- How do you handle false positive alerts that lead to alert fatigue?

**⚠️ Common Pain Indicators:**
- "We only find out about model problems when customers complain"
- "We have too many alerts and can't tell which ones are important"
- "It takes days to figure out why a model suddenly started performing poorly"
- "We don't know the actual business impact of our model issues"

### Desired Future State

**🎯 Vision Questions:**
- What would comprehensive model observability look like for your use cases?
- How would you like to predict and prevent model issues before they occur?
- What level of automation do you want in model issue detection and response?
- How should model monitoring integrate with your broader observability stack?

**📈 Success Metrics:**
- Detect X% of model issues before they impact business metrics
- Reduce mean time to detection (MTTD) from X hours to Y minutes
- Reduce false positive alerts by X%
- Achieve automated resolution for X% of common model issues

---

## 🔄 MLOps Automation & Workflows

### Current State Assessment

**Pipeline Automation:**
- How much of your ML workflow is automated vs. manual?
- What tools do you use for ML pipeline orchestration? (Airflow, Kubeflow, custom)
- How do you handle dependencies between different ML pipeline stages?
- How do you manage pipeline scheduling and triggers?

**CI/CD for ML:**
- Do you have CI/CD practices for your ML code and models?
- How do you test ML pipelines and validate changes?
- What's your approach to environment promotion (dev → staging → prod)?
- How do you handle rollbacks and emergency fixes?

**Workflow Integration:**
- How do your ML workflows integrate with broader data engineering pipelines?
- What's your approach to handling failed pipeline runs and retries?
- How do you manage resource allocation and cost optimization for workflows?
- How do you handle sensitive data and security in automated pipelines?

### Pain Points & Challenges

**🔍 Discovery Questions:**
- How much manual intervention is required to run your ML workflows end-to-end?
- How often do pipeline failures block model updates or deployments?
- How do you handle dependencies between ML workflows and data engineering pipelines?
- What percentage of your team's time is spent on pipeline maintenance vs. ML development?
- How do you ensure consistency between development and production environments?

**⚠️ Common Pain Indicators:**
- "Our ML workflows require constant manual babysitting"
- "Pipeline failures cascade and take down multiple dependent processes"
- "We spend more time debugging pipelines than developing models"
- "Different environments behave differently, causing unexpected failures"

### Desired Future State

**🎯 Vision Questions:**
- What would fully automated, self-healing ML workflows look like?
- How would you like to handle workflow orchestration at scale?
- What level of integration do you need with your existing DevOps practices?
- How important is multi-cloud or hybrid workflow capability?

**📈 Success Metrics:**
- Achieve X% automation of ML workflow stages
- Reduce pipeline failure rate to <X%
- Reduce time spent on pipeline maintenance by X%
- Achieve 100% environment consistency between dev/staging/prod

---

## 🏢 Team Collaboration & Governance

### Current State Assessment

**Team Structure & Roles:**
- How are your data science and ML engineering teams organized?
- What are the handoff processes between data scientists, ML engineers, and DevOps?
- How do you handle knowledge sharing and collaboration across teams?
- What tools do you use for project management and communication?

**Platform & Tool Standardization:**
- How many different ML platforms and tools does your organization use?
- What's your approach to tool selection and standardization?
- How do you handle different preferences and requirements across teams?
- What's your strategy for managing technical debt in ML systems?

**Governance & Compliance:**
- How do you ensure compliance with data privacy regulations (GDPR, CCPA, etc.)?
- What's your approach to model governance and risk management?
- How do you handle audit trails and model documentation requirements?
- Who is responsible for ML platform decisions and standards?

### Pain Points & Challenges

**🔍 Discovery Questions:**
- How much time do teams spend on integration and compatibility issues?
- How do you handle conflicts between team preferences and organizational standards?
- What happens when key team members leave and take institutional knowledge?
- How do you onboard new team members to your ML platform and practices?
- How do you balance innovation with standardization and governance?

**⚠️ Common Pain Indicators:**
- "Each team uses different tools and we can't share work effectively"
- "We waste time rebuilding capabilities that exist elsewhere in the organization"
- "Compliance and governance slow down our innovation velocity"
- "New team members take months to become productive"

### Desired Future State

**🎯 Vision Questions:**
- What would seamless collaboration between data science and ML engineering teams look like?
- How would you like to balance standardization with team autonomy?
- What level of self-service capability do you want for data scientists?
- How should governance be embedded in the development workflow?

**📈 Success Metrics:**
- Reduce time-to-productivity for new team members from X weeks to Y days
- Increase cross-team feature/model reuse by X%
- Achieve 100% compliance with governance requirements
- Reduce tool sprawl and consolidate on X core platforms

---

## 💰 Business Impact & ROI

### Current State Assessment

**Value Measurement:**
- How do you measure the business impact of your ML initiatives?
- What's your approach to calculating ROI for ML projects?
- How do you track the performance of models in terms of business metrics?
- What percentage of your ML projects make it to production and drive value?

**Cost Management:**
- What are your current ML infrastructure and tooling costs?
- How do you allocate costs across different teams and projects?
- What's your biggest cost driver in your ML operations?
- How do you optimize for cost vs. performance trade-offs?

**Strategic Alignment:**
- How do ML initiatives align with broader business objectives?
- What's your process for prioritizing ML use cases and investments?
- How do you communicate ML value to executive stakeholders?
- What's your vision for scaling ML across the organization?

### Pain Points & Challenges

**🔍 Discovery Questions:**
- How many ML projects fail to deliver expected business value?
- How long does it typically take to see ROI from ML investments?
- What's your biggest challenge in demonstrating ML value to stakeholders?
- How do you handle the uncertainty and experimentation inherent in ML projects?
- What prevents you from scaling successful ML use cases?

**⚠️ Common Pain Indicators:**
- "We can't demonstrate clear business value from our ML investments"
- "ML projects take too long and cost too much relative to their impact"
- "We struggle to scale successful pilots to enterprise-wide solutions"
- "Executives are skeptical about continued ML investment"

### Desired Future State

**🎯 Vision Questions:**
- What would comprehensive business value tracking for ML look like?
- How would you like to accelerate time-to-value for new ML initiatives?
- What's your vision for ML becoming a competitive advantage?
- How should ML investment decisions be made and justified?

**📈 Success Metrics:**
- Increase percentage of ML projects delivering measurable business value to X%
- Reduce time-to-value for ML projects from X months to Y weeks
- Achieve X% ROI across ML portfolio
- Scale successful ML use cases to serve X% of relevant business processes

---

## 🎯 Platform Evaluation & Migration

### Current State Assessment

**Platform Landscape:**
- What ML platforms and tools are you currently using? (cloud providers, tools, versions)
- How satisfied are you with your current platform capabilities?
- What are the biggest limitations of your current setup?
- What integrations are critical for your ML workflows?

**Migration Considerations:**
- What would motivate you to consider a platform change or addition?
- What are your requirements for platform migration or expansion?
- How important is compatibility with your existing investments?
- What timeline would you consider for platform changes?

### Desired Platform Capabilities

**🔍 Evaluation Questions:**
- What capabilities are missing from your current platform?
- How important is having an integrated, end-to-end ML platform vs. best-of-breed tools?
- What are your requirements for multi-cloud or hybrid deployments?
- How important is vendor support and professional services?
- What compliance or security requirements must any platform meet?

**🎯 Success Criteria:**
- Platform should reduce total cost of ownership by X%
- Should accelerate ML development velocity by X%
- Must support all current use cases plus enable new capabilities
- Should reduce operational overhead and maintenance burden

---

## 📋 Discovery Session Templates

### Executive Stakeholder Session (30-45 minutes)
**Focus:** Business value, strategic alignment, ROI
- Business Impact & ROI questions
- High-level Team Collaboration & Governance
- Platform evaluation criteria and timeline

### Data Science Team Session (60-90 minutes)
**Focus:** Development workflow, experimentation, collaboration
- Feature Engineering & Data Preparation
- Model Training & Experimentation  
- Model Validation & Testing
- Team Collaboration workflow questions

### ML Engineering Team Session (60-90 minutes)
**Focus:** Infrastructure, deployment, operations
- Model Deployment & Serving
- Model Monitoring & Observability
- MLOps Automation & Workflows
- Platform technical requirements

### Platform/Architecture Team Session (45-60 minutes)
**Focus:** Infrastructure, integration, governance
- Platform Evaluation & Migration
- Integration requirements
- Security and compliance requirements
- Cost optimization strategies

---

## 🔄 Follow-up Framework

### After Discovery Sessions

**1. Gap Analysis (Internal)**
- Identify gaps between current state and desired outcomes
- Prioritize pain points by business impact and technical complexity
- Map our best practices guides to their specific challenges

**2. Solution Mapping**
- Reference specific sections from our best practices guides
- Identify quick wins and longer-term strategic initiatives
- Develop proof-of-concept or pilot project proposals

**3. Success Metrics Definition**
- Establish baseline measurements for key metrics
- Define success criteria for potential initiatives
- Create measurement plan for value demonstration

**4. Next Steps & Timeline**
- Propose pilot projects or proof-of-concepts
- Outline evaluation criteria and timeline
- Plan technical deep-dive sessions for specific areas

---

## 📚 Reference: Best Practices Guide Mapping

Use our comprehensive guides to address specific challenges identified during discovery:

- **Feature Engineering issues** → [Feature Engineering Best Practices](./feature_eng_claude.md)
- **Training/Validation challenges** → [Model Training & Validation Guide](./model_train_validate_claude.md)  
- **Deployment/Serving problems** → [Model Deployment & Serving Guide](./model_deployment_serving_claude.md)
- **Monitoring/Alerting gaps** → [Model Monitoring & Alerting Guide](./model_monitoring_alerting_claude.md)
- **Development velocity concerns** → [AutoML Best Practices Guide](./auto_ml_claude.md)

Each guide contains 10-24 specific best practices with implementation examples and production-ready code patterns.