# Databricks notebook source
# MAGIC %md
# MAGIC # Data Exploration
# MAGIC - This notebook performs exploratory data analysis on the dataset.
# MAGIC - To expand on the analysis, attach this notebook to a cluster with runtime version **12.2.x-cpu-ml-scala2.12**,
# MAGIC edit [the options of pandas-profiling](https://pandas-profiling.ydata.ai/docs/master/rtd/pages/advanced_usage.html), and rerun it.
# MAGIC - Explore completed trials in the [MLflow experiment](#mlflow/experiments/3254325001238185).

# COMMAND ----------

import mlflow
import os
import uuid
import shutil
import pandas as pd
import databricks.automl_runtime

# Download input data from mlflow into a pandas DataFrame
# Create temporary directory to download data
temp_dir = os.path.join(os.environ["SPARK_LOCAL_DIRS"], "tmp", str(uuid.uuid4())[:8])
os.makedirs(temp_dir)

# Download the artifact and read it
training_data_path = mlflow.artifacts.download_artifacts(run_id="ce4164a854c94ffab5ac7a233376980c", artifact_path="data", dst_path=temp_dir)
df = pd.read_parquet(os.path.join(training_data_path, "training_data"))

# Delete the temporary data
shutil.rmtree(temp_dir)

target_col = "outcome"

# Drop columns created by AutoML before pandas-profiling
df = df.drop(['_automl_split_col_01ef'], axis=1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Semantic Type Detection Alerts
# MAGIC 
# MAGIC For details about the definition of the semantic types and how to override the detection, see
# MAGIC [Databricks documentation on semantic type detection](https://docs.databricks.com/applications/machine-learning/automl.html#semantic-type-detection).
# MAGIC 
# MAGIC - Semantic type `categorical` detected for columns `254761`, `260139`, `312437`, `40163554`, `40221901`, `40481087`, `4112343`, `4237458`, `4289517`, `432867`, `437663`, `80502`, `ETHNICITY_CONCEPT_ID`, `GENDER_CONCEPT_ID`, `RACE_CONCEPT_ID`. Training notebooks will encode features based on categorical transformations.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Profiling Results

# COMMAND ----------

from pandas_profiling import ProfileReport
df_profile = ProfileReport(df,
                           correlations={
                               "auto": {"calculate": True},
                               "pearson": {"calculate": True},
                               "spearman": {"calculate": True},
                               "kendall": {"calculate": True},
                               "phi_k": {"calculate": True},
                               "cramers": {"calculate": True},
                           }, title="Profiling Report", progress_bar=False, infer_dtypes=False)
profile_html = df_profile.to_html()

displayHTML(profile_html)

# COMMAND ----------


