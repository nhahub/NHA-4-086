<p align="center">
  <img src="https://github.com/OmarAhmedWahby/SAIA/blob/main/docs/Screenshots/SAIA.png">
</p>


<h2 align="center">
SAIA  is an AI-powered financial intelligence platform for stock analysis and prediction
It combines traditional market data with financial news sentiment to deliver smarter investment insights
</h2>

<p align="center">
  <a href="https://drive.google.com/file/d/17zyRsBWq9K9m2-3hd9UsK96SLm-tYAV7/view?usp=drive_link">
    <img src="https://img.shields.io/badge/Website Demo-Video-blue?style=for-the-badge">
  </a>
</p>

# 📌 Project Overview

SAIA (Stock and Arrows Investing App) is an AI-powered financial intelligence platform that combines traditional stock market analysis with news sentiment intelligence to provide more reliable investment insights

The platform continuously collects market data and financial news, processes them through automated data pipelines, applies machine learning and NLP models, and presents predictions, sentiment analysis, and interactive visualizations through a modern web application and Power BI dashboards


# 🧰 Tools & Technologies
[![Databricks](https://img.shields.io/badge/Databricks-FF3621?style=for-the-badge&logo=databricks&logoColor=white)]()
[![Apache Spark](https://img.shields.io/badge/Apache_Spark-E25A1C?style=for-the-badge&logo=apachespark&logoColor=white)]()
[![Delta Lake](https://img.shields.io/badge/Delta_Lake-00ADD8?style=for-the-badge)]()
[![Lakeflow](https://img.shields.io/badge/Lakeflow_Jobs-FF3621?style=for-the-badge)]()
[![Databricks Asset Bundles](https://img.shields.io/badge/Databricks_Asset_Bundles-FF3621?style=for-the-badge&logo=databricks&logoColor=white)]()
[![Power BI](https://img.shields.io/badge/Power_BI-F2C811?style=for-the-badge&logo=powerbi&logoColor=black)]()
[![DAX](https://img.shields.io/badge/DAX-217346?style=for-the-badge)]()
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)]()
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)]()
[![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)]()


# 🎯 Project Objectives

* Build an end-to-end financial intelligence platform for stock market analysis and prediction.
* Analyze and generate insights for 15,000+ stocks across multiple global stock exchanges.
* Collect and process market-moving financial news from relevant sources.
* Apply NLP-based sentiment analysis to measure the impact of news on stock prices.
* Combine traditional market analysis with AI-driven sentiment analysis into a unified prediction framework.
* Develop machine learning models to improve stock price forecasting.
* Deliver interactive investment insights through a modern web application and Power BI dashboards.

# ⚡ Challenges

* Processing large-scale market data for 15,000+ stocks across multiple exchanges.
* Collecting and filtering high-quality financial news from diverse sources.
* Linking news articles to the correct stocks and market events.
* Building scalable data pipelines for market data and news ingestion.
* Combining structured market data with unstructured text for unified analysis.
* Improving prediction reliability by integrating machine learning with NLP-based sentiment analysis.
* Delivering fast, interactive analytics through a web application and Power BI dashboards.




# 📊 Data Engineering

## Data Engineering Objectives

The data engineering layer was designed to:

* Collect and centralize stock market and financial news data.
* Ensure data quality through validation and cleansing.
* Build scalable ETL/ELT pipelines for automated data processing.
* Support large-scale analytics and machine learning workloads.
* Deliver analytics-ready datasets for prediction and visualization.

## Data Engineering Architecture

* Medallion Architecture (Bronze → Silver → Gold)

# Overall System Architecture
![](https://github.com/OmarAhmedWahby/SAIA/blob/main/docs/Screenshots/data_architeture.png)


---

## Components

* Stock Market Data APIs
* Financial News APIs
* Databricks Workspaces
* Apache Spark Processing
* Delta Lake Storage
* Lakeflow Job Orchestration
* Databricks Asset Bundles

---

## Data Warehouse Schema
* Star Schema
  
![](https://github.com/OmarAhmedWahby/SAIA/blob/main/docs/Screenshots/Data_screens/Star_Schema.png)

---

## Pipeline Stages

The data pipelines continuously ingest stock market data and financial news, validate and transform the incoming records using Apache Spark, enrich market data with sentiment information, and publish analytics-ready datasets for machine learning models, Power BI dashboards, and the web application.

* Market Data Extraction
* Financial News Collection
* Data Validation & Cleansing
* Data Transformation
* Sentiment Analysis
* Feature Engineering
* Gold Layer Publishing
* Analytical Serving

### Pipeline Screenshots

* Master Job
![](https://github.com/OmarAhmedWahby/SAIA/blob/main/docs/Screenshots/Data_screens/master_job.png)

* Extract Job
![](https://github.com/OmarAhmedWahby/SAIA/blob/main/docs/Screenshots/Data_screens/extract_job.png)

* Transformation Job
![](https://github.com/OmarAhmedWahby/SAIA/blob/main/docs/Screenshots/Data_screens/transformation_job.png)

* Serving Job
![](https://github.com/OmarAhmedWahby/SAIA/blob/main/docs/Screenshots/Data_screens/serving_job.png)

* Companies Job
![](https://github.com/OmarAhmedWahby/SAIA/blob/main/docs/Screenshots/Data_screens/companies_job.png)

* News Job
![](https://github.com/OmarAhmedWahby/SAIA/blob/main/docs/Screenshots/Data_screens/news_job.png)

---

## 📈 Data Engineering Summary

| Metric | Value |
|---------|------:|
| Data Sources | 5 |
| Stock Price Records | 17M+ |
| Stocks Covered | 17,000+ |
| News Articles Processed | 500K+ |
| Time Period | 2022 – Present |
| Storage Engine | Delta Lake |
| Loading Strategy | Incremental ETL |



# 👨‍💻 Team Members

### Omar Ahmed Wahby

Data Engineering • Business Intelligence • Power BI • Analytics

- [LinkedIn](https://www.linkedin.com/in/omarwahby)  
- [GitHub](https://github.com/OmarAhmedWahby)

### Abdallah Mohamed Abdelzaher

Data Engineering • Machine Learning • NLP


- [LinkedIn](https://www.linkedin.com/in/abdallahabdelzaher)  
- [GitHub](https://github.com/AbdallahAbdElzaher24)
