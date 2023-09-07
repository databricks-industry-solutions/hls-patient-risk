# Databricks notebook source
# MAGIC %md
# MAGIC # Prepare the Environment

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Libraries and define user names

# COMMAND ----------

# DBTITLE 1,Import Libraries
import mlflow
import os
import requests
import numpy as np
import pandas as pd
import json

from databricks import feature_store
from databricks.feature_store import feature_table, FeatureLookup
fs = feature_store.FeatureStoreClient()

# COMMAND ----------

# DBTITLE 1,Create Class for making model serving endpoint
# MAGIC %run ./resources/00-init-modelserving 

# COMMAND ----------

# DBTITLE 1,Dynamically define user_name and schema_name
user_name=sql(f"SELECT current_user() as user").collect()[0]['user'].split('@')[0].replace('.','_')
schema_name = f"OMOP_{user_name}"
feature_schema = schema_name + '_features'
sql(f"USE {schema_name}")
print(schema_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set up Cosmos DB credentials
# MAGIC
# MAGIC In this section, you need to take some manual steps to make Cosmos DB accessible to this notebook. Databricks needs permission to create and update Cosmos DB containers so that Cosmos DB can work with Feature Store. The following steps stores Cosmos DB keys in Databricks Secrets.
# MAGIC
# MAGIC ### Look up the keys for Cosmos DB
# MAGIC 1. Go to Azure portal at https://portal.azure.com/
# MAGIC 2. Search and open "Cosmos DB", then create or select an account.
# MAGIC 3. Navigate to "keys" the view the URI and credentials.
# MAGIC 4. You'll paste the Cosmos DB "Read-write Keys" and "Read-only Keys" into the respective Databricks secret.
# MAGIC
# MAGIC ### Authenticate Databricks command line (CLI) with your workspace
# MAGIC
# MAGIC Use [Databricks CLI](https://learn.microsoft.com/en-us/azure/databricks/dev-tools/cli/databricks-cli) to create secrets. You'll have to [authenticate](https://learn.microsoft.com/en-us/azure/databricks/dev-tools/cli/databricks-cli#--set-up-authentication) between Databricks CLI and Databricks Workspace. 
# MAGIC 1. Using CLI enter this command to prompt Databricks Workspace authentication 
# MAGIC ```
# MAGIC     databricks configure --token
# MAGIC ```
# MAGIC ### Provide online store credentials using Databricks secrets
# MAGIC **Note:** For simplicity, the commands below use predefined names for the scope and secrets. To choose your own scope and secret names, follow the process in the Databricks [documentation](https://docs.microsoft.com/azure/databricks/applications/machine-learning/feature-store/online-feature-stores).
# MAGIC
# MAGIC 1. Using CLI, create two secret scopes in Databricks.
# MAGIC
# MAGIC     ```
# MAGIC     databricks secrets create-scope --scope one-env-dynamodb-fs-read
# MAGIC     databricks secrets create-scope --scope one-env-dynamodb-fs-write
# MAGIC     ```
# MAGIC
# MAGIC 2. Using CLI, add keys to the scopes. Then when prompted, copy/paste the Cosmos DB keys as secrets.
# MAGIC
# MAGIC    **Note:** The keys should follow the format `<prefix>-authorization-key`. Give the "prefix" a unique and meaningful name, but be sure the key ends with "-authorization-key". For simplicity, these commands use predefined names here. When the commands run, you will be prompted to copy your Cosmos DB "Read-write Keys" and "Read-only Keys" into an editor.
# MAGIC
# MAGIC     ```
# MAGIC     databricks secrets put --scope one-env-dynamodb-fs-read --key solutionaccelerator-authorization-key
# MAGIC     databricks secrets put --scope one-env-dynamodb-fs-write --key solutionaccelerator-authorization-key
# MAGIC     ```
# MAGIC     
# MAGIC Now the credentials are stored with Databricks Secrets. You will use them below to create the online feature store.

# COMMAND ----------

# MAGIC %md
# MAGIC # Publish features to the online feature store

# COMMAND ----------

# DBTITLE 1,Define Online Feature Store Spec. Publish First FS Table (drug_features)
from databricks.feature_store.online_store_spec import AzureCosmosDBSpec
#Fill in the account_uri of your Azure Cosmos DB account
account_uri = "https://field-demo.documents.azure.com:443/"

# Specify the online store.
# Note: These commands use the predefined secret prefix. If you used a different secret scope or prefix, edit these commands before running them.
#       Make sure you have a database created with same name as specified below.
#       Do not manually create the database or container in Cosmos DB. The publish_table() command creates it for you.

online_store_spec = AzureCosmosDBSpec(
  account_uri=account_uri,
  read_secret_prefix="one-env-dynamodb-fs-read/solutionaccelerator",
  write_secret_prefix="one-env-dynamodb-fs-write/solutionaccelerator",
  database_name=f"{user_name}_solution_accelerator_patient_risk_scoring_feature_store",
  container_name="drug_features"
)

# Push the feature table to online store.
fs.publish_table(f'{feature_schema}.drug_features', online_store_spec)

# COMMAND ----------

# DBTITLE 1,Publish Second FS Table (condition_history_features)
# Specify the online store.
# Note: These commands use the predefined secret prefix. If you used a different secret scope or prefix, edit these commands before running them.
#       Make sure you have a database created with same name as specified below.
#       Do not manually create the database or container in Cosmos DB. The publish_table() command creates it for you.

online_store_spec = AzureCosmosDBSpec(
  account_uri=account_uri,
  read_secret_prefix="one-env-dynamodb-fs-read/solutionaccelerator",
  write_secret_prefix="one-env-dynamodb-fs-write/solutionaccelerator",
  database_name=f"{user_name}_solution_accelerator_patient_risk_scoring_feature_store",
  container_name=f"condition_history_features"
)

# Push the feature table to online store.
fs.publish_table(f'{feature_schema}.condition_history_features', online_store_spec)

# COMMAND ----------

# DBTITLE 1,Publish Third Table (subject_demographics_features)
# Note: These commands use the predefined secret prefix. If you used a different secret scope or prefix, edit these commands before running them.
#       Make sure you have a database created with same name as specified below.
#       Do not manually create the database or container in Cosmos DB. The publish_table() command creates it for you.

online_store_spec = AzureCosmosDBSpec(
  account_uri=account_uri,
  read_secret_prefix="one-env-dynamodb-fs-read/solutionaccelerator",
  write_secret_prefix="one-env-dynamodb-fs-write/solutionaccelerator",
  database_name=f"{user_name}_solution_accelerator_patient_risk_scoring_feature_store",
  container_name="subject_demographics_features"
)

# Push the feature table to online store.
fs.publish_table(f'{feature_schema}.subject_demographics_features', online_store_spec)

# COMMAND ----------

# MAGIC %md
# MAGIC # Create Serverless Endpoint for Model Scoring

# COMMAND ----------

# DBTITLE 1,Load your model name from 03-AutoML
model_name = f"{user_name}_omop_patientrisk_model"

# COMMAND ----------

# DBTITLE 1, Set up & start a serverless model serving endpoint using the API
client = mlflow.tracking.MlflowClient()
latest_model = client.get_latest_versions(model_name, stages=["Production"])[0]

#See the 00-init-modelserving for the endpoint API details
serving_client = EndpointApiClient()

#Start the endpoint using the REST API (you can do it using the UI directly)
serving_client.create_endpoint_if_not_exists(f"{user_name}_patientrisk_feature_store_endpoint", model_name=model_name, model_version = latest_model.version, workload_size="Small", scale_to_zero_enabled=True, wait_start = True)

# COMMAND ----------

# Fill in the Databricks access token value.
# Note: You can generate a new Databricks access token by using databricks-cli or clicking your name in the top right > "User Settings" > "Developer" > Access Tokens" > "Generate new token". 
DATABRICKS_TOKEN = "FillinWithYourToken"

# COMMAND ----------

# DBTITLE 1,Functions to Score using Endpoint
workspacename = spark.conf.get("spark.databricks.workspaceUrl")

def create_tf_serving_json(data):
    return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

def score_model(dataset):
    url = f"https://{workspacename}/serving-endpoints/{user_name}_patientrisk_feature_store_endpoint/invocations"
    headers = {'Authorization': f'Bearer {DATABRICKS_TOKEN}', 'Content-Type': 'application/json'}
    ds_dict = {'dataframe_split': dataset.to_dict(orient='split')} if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
    data_json = json.dumps(ds_dict, allow_nan=True)
    response = requests.request(method='POST', headers=headers, url=url, data=data_json)
    if response.status_code != 200:
        raise Exception(f'Request failed with status {response.status_code}, {response.text}')

    return response.json()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Score patient records!

# COMMAND ----------

# DBTITLE 1,Simulate receiving patient records for scoring
newrecords = pd.DataFrame([68544,1460,5155,29660,52882,23215,32188,45457,86968], columns=["subject_id"])
display(newrecords)

# COMMAND ----------

# DBTITLE 1,Score patient records using Endpoint
print(score_model(newrecords))
