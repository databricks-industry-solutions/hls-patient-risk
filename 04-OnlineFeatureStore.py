# Databricks notebook source
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

# DBTITLE 1,Create Class for making serving endpoint
# MAGIC %run ./resources/00-init-modelserving 

# COMMAND ----------

# DBTITLE 1,Dynamically define user_name and schema_name
user_name=sql(f"SELECT current_user() as user").collect()[0]['user'].split('@')[0].replace('.','_')
schema_name = f"OMOP_{user_name}"
feature_schema = schema_name + '_features'
sql(f"USE {schema_name}")
print(schema_name)

# COMMAND ----------

# DBTITLE 1,Define Online Feature Store Spec, Publish First Table
from databricks.feature_store.online_store_spec import AzureCosmosDBSpec
account_uri = "https://field-demo.documents.azure.com:443/"

# Specify the online store.
# Note: These commands use the predefined secret prefix. If you used a different secret scope or prefix, edit these commands before running them.
#       Make sure you have a database created with same name as specified below.
#       Do not manually create the database or container in Cosmos DB. The publish_table() command creates it for you.

online_store_spec = AzureCosmosDBSpec(
  account_uri=account_uri,
  write_secret_prefix="one-env-dynamodb-fs-write/solutionaccelerator",
  read_secret_prefix="one-env-dynamodb-fs-read/solutionaccelerator",
  database_name="solution_accelerator_patient_risk_scoring_feature_store",
  container_name="subject_demographics_features_v3"
)

# Push the feature table to online store.
fs.publish_table(f'{feature_schema}.subject_demographics_features', online_store_spec)

# COMMAND ----------

# DBTITLE 1,Publish Second FS Table
# Specify the online store.
# Note: These commands use the predefined secret prefix. If you used a different secret scope or prefix, edit these commands before running them.
#       Make sure you have a database created with same name as specified below.
#       Do not manually create the database or container in Cosmos DB. The publish_table() command creates it for you.

online_store_spec = AzureCosmosDBSpec(
  account_uri=account_uri,
  write_secret_prefix="one-env-dynamodb-fs-write/solutionaccelerator",
  read_secret_prefix="one-env-dynamodb-fs-read/solutionaccelerator",
  database_name="solution_accelerator_patient_risk_scoring_feature_store",
  container_name="drug_features_v3"
)

# Push the feature table to online store.
fs.publish_table(f'{feature_schema}.drug_features', online_store_spec)

# COMMAND ----------

# DBTITLE 1,Publish Third Table
# Specify the online store.
# Note: These commands use the predefined secret prefix. If you used a different secret scope or prefix, edit these commands before running them.
#       Make sure you have a database created with same name as specified below.
#       Do not manually create the database or container in Cosmos DB. The publish_table() command creates it for you.

online_store_spec = AzureCosmosDBSpec(
  account_uri=account_uri,
  write_secret_prefix="one-env-dynamodb-fs-write/solutionaccelerator",
  read_secret_prefix="one-env-dynamodb-fs-read/solutionaccelerator",
  database_name="solution_accelerator_patient_risk_scoring_feature_store",
  container_name="condition_history_features_v3"
)

# Push the feature table to online store.
fs.publish_table(f'{feature_schema}.condition_history_features', online_store_spec)

# COMMAND ----------

# DBTITLE 1,Load your model name
model_name = "omop_patientrisk_model"

# COMMAND ----------

import time
import requests 

class EndpointApiClient:
    def __init__(self):
        self.base_url =dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
        self.token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
        self.headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}

    def create_inference_endpoint(self, endpoint_name, served_models):
        data = {"name": endpoint_name, "config": {"served_models": served_models}}
        return self._post("api/2.0/serving-endpoints", data)

    def get_inference_endpoint(self, endpoint_name):
        return self._get(f"api/2.0/serving-endpoints/{endpoint_name}", allow_error=True)
      
      
    def inference_endpoint_exists(self, endpoint_name):
      ep = self.get_inference_endpoint(endpoint_name)
      if 'error_code' in ep and ep['error_code'] == 'RESOURCE_DOES_NOT_EXIST':
          return False
      if 'error_code' in ep and ep['error_code'] != 'RESOURCE_DOES_NOT_EXIST':
          raise Exception(f"enpoint exists ? {ep}")
      return True

    def create_enpoint_if_not_exists(self, endpoint_name, model_name, model_version, workload_size, scale_to_zero_enabled=True, wait_start=True):
      models = [{
            "model_name": model_name,
            "model_version": model_version,
            "workload_size": workload_size,
            "scale_to_zero_enabled": scale_to_zero_enabled,
      }]
      if not self.inference_endpoint_exists(endpoint_name):
        r = self.create_inference_endpoint(endpoint_name, models)
      #Make sure we have the proper version deployed
      else:
        ep = self.get_inference_endpoint(endpoint_name)
        if 'pending_config' in ep:
            self.wait_endpoint_start(endpoint_name)
            ep = self.get_inference_endpoint(endpoint_name)
        if 'pending_config' in ep:
            model_deployed = ep['pending_config']['served_models'][0]
            print(f"Error with the model deployed: {model_deployed} - state {ep['state']}")
        else:
            model_deployed = ep['config']['served_models'][0]
        if model_deployed['model_version'] != model_version:
          print(f"Current model is version {model_deployed['model_version']}. Updating to {model_version}...")
          u = self.update_model_endpoint(endpoint_name, {"served_models": models})
      if wait_start:
        self.wait_endpoint_start(endpoint_name)
      
      
    def list_inference_endpoints(self):
        return self._get("api/2.0/serving-endpoints")

    def update_model_endpoint(self, endpoint_name, conf):
        return self._put(f"api/2.0/serving-endpoints/{endpoint_name}/config", conf)

    def delete_inference_endpoint(self, endpoint_name):
        return self._delete(f"api/2.0/serving-endpoints/{endpoint_name}")

    def wait_endpoint_start(self, endpoint_name):
      i = 0
      while self.get_inference_endpoint(endpoint_name)['state']['config_update'] == "IN_PROGRESS" and i < 500:
        print("waiting for endpoint to build model image and start")
        time.sleep(30)
        i += 1
      
    # Making predictions

    def query_inference_endpoint(self, endpoint_name, data):
        return self._post(f"realtime-inference/{endpoint_name}/invocations", data)

    # Debugging

    def get_served_model_build_logs(self, endpoint_name, served_model_name):
        return self._get(
            f"api/2.0/serving-endpoints/{endpoint_name}/served-models/{served_model_name}/build-logs"
        )

    def get_served_model_server_logs(self, endpoint_name, served_model_name):
        return self._get(
            f"api/2.0/serving-endpoints/{endpoint_name}/served-models/{served_model_name}/logs"
        )

    def get_inference_endpoint_events(self, endpoint_name):
        return self._get(f"api/2.0/serving-endpoints/{endpoint_name}/events")

    def _get(self, uri, data = {}, allow_error = False):
        r = requests.get(f"{self.base_url}/{uri}", params=data, headers=self.headers)
        return self._process(r, allow_error)

    def _post(self, uri, data = {}, allow_error = False):
        return self._process(requests.post(f"{self.base_url}/{uri}", json=data, headers=self.headers), allow_error)

    def _put(self, uri, data = {}, allow_error = False):
        return self._process(requests.put(f"{self.base_url}/{uri}", json=data, headers=self.headers), allow_error)

    def _delete(self, uri, data = {}, allow_error = False):
        return self._process(requests.delete(f"{self.base_url}/{uri}", json=data, headers=self.headers), allow_error)

    def _process(self, r, allow_error = False):
      if r.status_code == 500 or r.status_code == 403 or not allow_error:
        print(r.text)
        r.raise_for_status()
      return r.json()

# COMMAND ----------

# DBTITLE 1, Set up & start a Serverless model serving endpoint using the API: We will use the API to programmatically start the endpoint:
client = mlflow.tracking.MlflowClient()
latest_model = client.get_latest_versions(model_name, stages=["Production"])[0]

#See the 00-init-expert notebook for the endpoint API details
serving_client = EndpointApiClient()

#Start the enpoint using the REST API (you can do it using the UI directly)
serving_client.create_endpoint_if_not_exists("patientrisk_feature_store_endpoint_v3", model_name=model_name, model_version = latest_model.version, workload_size="Small", scale_to_zero_enabled=True, wait_start = True)

# COMMAND ----------

# Fill in the Databricks access token value.
# Note: You can generate a new Databricks access token by going to left sidebar "Settings" > "User Settings" > "Access Tokens", or using databricks-cli.

DATABRICKS_TOKEN = "dapid768198092de9b82bd55a50523b696d5"

# COMMAND ----------

# DBTITLE 1,Functions to Score using Endpoint
def create_tf_serving_json(data):
    return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

def score_model(dataset):
    url = 'https://e2-demo-field-eng.cloud.databricks.com/serving-endpoints/patientrisk_feature_store_endpoint_v3/invocations'
    headers = {'Authorization': f'Bearer {DATABRICKS_TOKEN}', 'Content-Type': 'application/json'}
    ds_dict = {'dataframe_split': dataset.to_dict(orient='split')} if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
    data_json = json.dumps(ds_dict, allow_nan=True)
    response = requests.request(method='POST', headers=headers, url=url, data=data_json)
    if response.status_code != 200:
        raise Exception(f'Request failed with status {response.status_code}, {response.text}')

    return response.json()

# COMMAND ----------

newrecords = pd.DataFrame([68544,1460,5155,29660,52882,23215,32188,45457,86968], columns=["subject_id"])
display(newrecords)

# COMMAND ----------

print(score_model(newrecords))
