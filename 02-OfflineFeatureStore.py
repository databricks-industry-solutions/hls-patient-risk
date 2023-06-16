# Databricks notebook source
# MAGIC %md
# MAGIC # Experiment parameters
# MAGIC First we set up the paramters for the experiment using databricks notebooks widgets utility.

# COMMAND ----------

dbutils.widgets.removeAll()

dbutils.widgets.dropdown('drop_schema','yes',['yes','no']) # set to no if you already have the OMOP data downlaoded and created the schema 

dbutils.widgets.text('target_condition_concept_id','4229440') # CHF
dbutils.widgets.text('outcome_concept_id','9203') # Emergency Room Visit

dbutils.widgets.text('drug1_concept_id','40163554') # Warfarin
dbutils.widgets.text('drug2_concept_id','40221901') # Acetaminophen

dbutils.widgets.text('min_observation_period','1095') # whashout period in days
dbutils.widgets.text('min_time_at_risk','7')

dbutils.widgets.text('max_time_at_risk','365')
dbutils.widgets.text('cond_history_years','5')
dbutils.widgets.text('max_n_commorbidities','5')

# COMMAND ----------

drop_schema = dbutils.widgets.get('drop_schema')

target_condition_concept_id = dbutils.widgets.get('target_condition_concept_id')
outcome_concept_id = dbutils.widgets.get('outcome_concept_id')

drug1_concept_id = dbutils.widgets.get('drug1_concept_id')
drug2_concept_id = dbutils.widgets.get('drug2_concept_id')

min_observation_period = dbutils.widgets.get('min_observation_period')
min_time_at_risk = dbutils.widgets.get('min_time_at_risk')
max_time_at_risk = dbutils.widgets.get('max_time_at_risk')

cond_history_years = dbutils.widgets.get('cond_history_years')
max_n_commorbidities = dbutils.widgets.get('max_n_commorbidities')

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE WIDGET text target_condition_concept_id DEFAULT "4229440"; -- CHF
# MAGIC CREATE WIDGET text outcome_concept_id DEFAULT "9203"; -- Emergency Room Visit
# MAGIC
# MAGIC CREATE WIDGET text drug1_concept_id DEFAULT "40163554"; -- Warfarin
# MAGIC CREATE WIDGET text drug2_concept_id DEFAULT "40221901"; -- Acetaminophen
# MAGIC
# MAGIC CREATE WIDGET text min_observation_period DEFAULT "1095"; -- whashout period in days
# MAGIC CREATE WIDGET text min_time_at_risk DEFAULT "7";
# MAGIC CREATE WIDGET text max_time_at_risk DEFAULT "365";
# MAGIC
# MAGIC CREATE WIDGET text cond_history_years DEFAULT "5";
# MAGIC CREATE WIDGET text max_n_commorbidities DEFAULT "5";

# COMMAND ----------

# DBTITLE 1,Set Cohort IDs
target_cohort_id = 1
outcome_cohort_id = 2

outcome_att_id = 0
condition_hist_att_id = 1
drug_hist_att_id = 2

# COMMAND ----------

# MAGIC %md
# MAGIC ## Add features
# MAGIC So far, we have leveraged OMOP's default schemas to store attributes associated to the target cohort that can be used for training our model. 
# MAGIC We now, leverage [databricks feature store](https://docs.databricks.com/machine-learning/feature-store/index.html) to create an offline feature store to store features that can be used (and re-used) for training our classifier.

# COMMAND ----------

from databricks import feature_store
from databricks.feature_store import feature_table, FeatureLookup
fs = feature_store.FeatureStoreClient()

# COMMAND ----------

user_name=sql(f"SELECT current_user() as user").collect()[0]['user'].split('@')[0].replace('.','_')
schema_name = f"OMOP_{user_name}"
sql(f"USE {schema_name}")
print(schema_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Drug Exposure History Feature

# COMMAND ----------

sql(f"SHOW TABLES IN {schema_name}").display()

# COMMAND ----------

# MAGIC %md
# MAGIC We create a new shema to hold the features as well as the training dataset for this analysis.

# COMMAND ----------

feature_schema = schema_name + '_features'
sql(f"DROP SCHEMA IF EXISTS {feature_schema} CASCADE")
sql(f"CREATE SCHEMA IF NOT EXISTS {feature_schema}")
print(feature_schema)

# COMMAND ----------

# DBTITLE 1,Add drug features to feature store
# fs.register_table(
#     delta_table="person_features_offline",
#     primary_keys=["person_id"],
#     description="Attributes related to a person identity",
# )

FEATURE_TABLE_NAME = f'{feature_schema}.drug_features'
description=f"drug features for drugs {drug1_concept_id} and {drug2_concept_id}"

try:
  fs.drop_table(FEATURE_TABLE_NAME)
except ValueError:
  pass
  
drug_features_df = sql(f"""
    select subject_id, VALUE_AS_CONCEPT_ID as drug_concept_id, VALUE_AS_NUMBER as drug_quantity 
    from cohort_attribute
    where 
    ATTRIBUTE_DEFINITION_ID = {drug_hist_att_id} and COHORT_DEFINITION_ID={target_cohort_id}
    """
    ).groupBy('subject_id').pivot('drug_concept_id').sum('drug_quantity').fillna(0)

fs.create_table(
    name=FEATURE_TABLE_NAME,
    primary_keys=["subject_id"],
    df=drug_features_df,
    schema=drug_features_df.schema,
    description=description
)

# #point in time relationship 
# fs.register_table(
#     delta_table="drug_features_offline",
#     primary_keys=["person_id"],
#     timestamp_keys=["drug_exposure_start_date"],
#     description="Attributes related to a person's drug history",
# )


# COMMAND ----------

sql(f'select * from {FEATURE_TABLE_NAME} limit 10').display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Commorbidity history features

# COMMAND ----------

# DBTITLE 1,top n commorbidities
# MAGIC %py
# MAGIC sql(f"""
# MAGIC select VALUE_AS_CONCEPT_ID as condition_concept_id, count(*) as cnt from cohort_attribute where ATTRIBUTE_DEFINITION_ID={condition_hist_att_id} group by 1
# MAGIC order by 2 desc
# MAGIC limit {max_n_commorbidities}
# MAGIC """).createOrReplaceTempView('top_comorbidities')

# COMMAND ----------

FEATURE_TABLE_NAME = f'{feature_schema}.condition_history_features'
description=f"condition history features for top {max_n_commorbidities} commorbidities"
try:
  fs.drop_table(FEATURE_TABLE_NAME)
except ValueError:
  pass
  
condition_history_df = sql(f"""
    select subject_id, VALUE_AS_CONCEPT_ID as condition_concept_id, VALUE_AS_NUMBER as n_condition_occurance 
    from cohort_attribute
    where 
    ATTRIBUTE_DEFINITION_ID = {condition_hist_att_id} and COHORT_DEFINITION_ID={target_cohort_id}
    and
    VALUE_AS_CONCEPT_ID in (select condition_concept_id from top_comorbidities)
    """
    ).groupBy('subject_id').pivot('condition_concept_id').sum('n_condition_occurance').fillna(0)

fs.create_table(
    name=FEATURE_TABLE_NAME,
    primary_keys=["subject_id"],
    df=condition_history_df,
    schema=condition_history_df.schema,
    description=description
)

# COMMAND ----------

sql(f'select * from {FEATURE_TABLE_NAME} limit 10').display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Demographics information 

# COMMAND ----------

# MAGIC %py
# MAGIC FEATURE_TABLE_NAME = f"{feature_schema}.subject_demographics_features"
# MAGIC description="demographic features"
# MAGIC try:
# MAGIC   fs.drop_table(FEATURE_TABLE_NAME)
# MAGIC except ValueError:
# MAGIC   pass
# MAGIC   
# MAGIC subject_demographics_features_df = sql(
# MAGIC     f"""select 
# MAGIC     c.subject_id,
# MAGIC     p.GENDER_CONCEPT_ID,
# MAGIC     date_diff(c.cohort_start_date, p.BIRTH_DATETIME) as age_in_days,
# MAGIC     p.RACE_CONCEPT_ID
# MAGIC   FROM cohort c
# MAGIC   join person p on c.subject_id = p.PERSON_ID
# MAGIC   where c.cohort_definition_id = {target_cohort_id}
# MAGIC   """
# MAGIC )
# MAGIC fs.create_table(
# MAGIC     name=FEATURE_TABLE_NAME,
# MAGIC     primary_keys=["subject_id"],
# MAGIC     df=subject_demographics_features_df,
# MAGIC     schema=subject_demographics_features_df.schema,
# MAGIC     description=description
# MAGIC )
# MAGIC
# MAGIC sql(f'select * from {FEATURE_TABLE_NAME} limit 10').display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create the Training Dataset
# MAGIC Now that our cohorts are in place, we can create the final dataset. we then use Databricks AutoML to train a model for predicting risk and also understand features impacting patient risk. 
# MAGIC To make it simpler, first we create a function that decides whether two cohorts overlap:

# COMMAND ----------

sql(f"SHOW TABLES IN {feature_schema}").display()

# COMMAND ----------

outcomes_df = sql(f"""
    select subject_id, VALUE_AS_CONCEPT_ID as outcome_concept_id, VALUE_AS_NUMBER as visited_emergency 
    from cohort_attribute
    where 
    ATTRIBUTE_DEFINITION_ID = {outcome_att_id} and COHORT_DEFINITION_ID={target_cohort_id}
    """
    ).groupBy('subject_id').pivot('outcome_concept_id').min('visited_emergency').fillna(0)
  
training_df = (
  sql('select subject_id from cohort')
  .filter(f'cohort_definition_id={target_cohort_id}')
  .join(outcomes_df, how='left',on='subject_id')
  .selectExpr('subject_id',f"CAST(`{outcome_concept_id}` AS INT) as outcome")
  .fillna(0)
)

# COMMAND ----------

display(outcomes_df)

# COMMAND ----------

display(training_df)

# COMMAND ----------

# DBTITLE 1,Create the Training Dataset
from databricks.feature_store import FeatureLookup

feature_lookups = [
    FeatureLookup(
      table_name = f'{feature_schema}.subject_demographics_features',
      lookup_key = 'subject_id'
    ),
    FeatureLookup(
      table_name = f'{feature_schema}.drug_features',
      lookup_key = 'subject_id'
    ),
    FeatureLookup(
      table_name = f'{feature_schema}.condition_history_features',
      lookup_key = 'subject_id'
    ),
  ]

training_set = fs.create_training_set(
  df=training_df,
  feature_lookups = feature_lookups,
  label = 'outcome',
 exclude_columns = ['subject_id']
)

# Load the training data from Feature Store
training_df = training_set.load_df()

training_df.selectExpr('avg(outcome)').display()

# COMMAND ----------

training_df.display()

# COMMAND ----------

# DBTITLE 1,Save the Training Dataset as a Delta Table
training_df.fillna(0).write.saveAsTable(f'{feature_schema}.training_data')

# COMMAND ----------

#print(f'{feature_schema}.training_data')

# COMMAND ----------

# DBTITLE 1,Use automl to build an ML model 
import databricks.automl as db_automl

summary_cl = db_automl.classify(training_df, target_col="outcome", primary_metric="f1", timeout_minutes=5, experiment_dir = "/patientrisk/experiments/feature_store")
print(f"Best run id: {summary_cl.best_trial.mlflow_run_id}")

# COMMAND ----------

import mlflow

# COMMAND ----------

import os

# COMMAND ----------

model_name = "omop_patientrisk_model"

# COMMAND ----------

# DBTITLE 1,Save best model in the registry & flag it as Production ready
# creating sample input to be logged (do not include the live features in the schema as they'll be computed within the model)
df_sample = training_df.limit(10).toPandas()
x_sample = df_sample.drop(columns=["outcome"])

# getting the model created by AutoML 
model = summary_cl.best_trial.load_model()

#Get the conda env from automl run
artifacts_path = mlflow.artifacts.download_artifacts(run_id=summary_cl.best_trial.mlflow_run_id)
env = mlflow.pyfunc.get_default_conda_env()
with open(artifacts_path+"model/requirements.txt", 'r') as f:
    env['dependencies'][-1]['pip'] = f.read().split('\n')

#Create a new run in the same experiment as our automl run.
with mlflow.start_run(run_name="best_fs_model", experiment_id=summary_cl.experiment.experiment_id) as run:
  #Use the feature store client to log our best model
  #TODO: need to add the conda env from the automl run
  fs.log_model(
              model=model, # object of your model
              artifact_path="model", #name of the Artifact under MlFlow
              flavor=mlflow.sklearn, # flavour of the model (our model has a SkLearn Flavour)
              training_set=training_set, # training set you used to train your model with AutoML
              input_example=x_sample, # Dataset example (Pandas dataframe)
              conda_env=env)

  #Copy automl images & params to our FS run
  for item in os.listdir(artifacts_path):
    if item.endswith(".png"):
      mlflow.log_artifact(artifacts_path+item)
  mlflow.log_metrics(summary_cl.best_trial.metrics)
  mlflow.log_params(summary_cl.best_trial.params)
  mlflow.log_param("automl_run_id", summary_cl.best_trial.mlflow_run_id)
  mlflow.set_tag(key='feature_store', value='expert_demo')
    
model_registered = mlflow.register_model(f"runs:/{run.info.run_id}/model", model_name)

#Move the model in production
client = mlflow.tracking.MlflowClient()
print("registering model version "+model_registered.version+" as production model")
client.transition_model_version_stage(model_name, model_registered.version, stage = "Production", archive_existing_versions=True)

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
serving_client.create_enpoint_if_not_exists("patientrisk_feature_store_endpoint", model_name=model_name, model_version = latest_model.version, workload_size="Small", scale_to_zero_enabled=True, wait_start = True)

# COMMAND ----------

# Fill in the Databricks access token value.
# Note: You can generate a new Databricks access token by going to left sidebar "Settings" > "User Settings" > "Access Tokens", or using databricks-cli.

DATABRICKS_TOKEN = "dapid768198092de9b82bd55a50523b696d5"
assert DATABRICKS_TOKEN.strip() != "dapid768198092de9b82bd55a50523b696d5"

# COMMAND ----------

# DBTITLE 1,Functions to Score using Endpoint
import os
import requests
import numpy as np
import pandas as pd
import json

def create_tf_serving_json(data):
    return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

def score_model(dataset):
    url = 'https://e2-demo-field-eng.cloud.databricks.com/serving-endpoints/patientrisk_feature_store_endpoint/invocations'
    headers = {'Authorization': f'Bearer {DATABRICKS_TOKEN}', 'Content-Type': 'application/json'}
    ds_dict = {'dataframe_split': dataset.to_dict(orient='split')} if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
    data_json = json.dumps(ds_dict, allow_nan=True)
    response = requests.request(method='POST', headers=headers, url=url, data=data_json)
    if response.status_code != 200:
        raise Exception(f'Request failed with status {response.status_code}, {response.text}')

    return response.json()

# COMMAND ----------

newdata = pd.DataFrame([(27412, 8532, 8527, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan), (29318, 8532, 8527, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan), (16746, 8532, 8527, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan), (13708, 8507, 8527, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan), (29306, 8532, 8527, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan), (19279, 8507, 8527, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan), (21520, 8532, 8527, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan), (20076, 8532, 8527, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan), (28445, 8507, 8527, 1.0, 0.0, np.nan, np.nan, np.nan, np.nan, np.nan), (19719, 8507, 8527, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)], columns=["age_in_days", "GENDER_CONCEPT_ID", "RACE_CONCEPT_ID", "40163554", "40221901", "260139", "40481087", "4112343", "4217975", "432867"])
display(newdata)

# COMMAND ----------

newrecords = pd.DataFrame([68544,1460,5155,29660,52882,23215,32188,45457,86968], columns=["subject_id"])
display(newrecords)

# COMMAND ----------

print(score_model(newrecords))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Training
# MAGIC Now that we have the training data ready, we will proceed to use databricks [AutoML]() to train a binary classifier that predicts the outcome (emergency room visit status) based on the selected features that we have stored in the feature store.
# MAGIC The next notebook ([02-automl-best-model]($./02-automl-best-model)) is an example notebook generated by AutoML based on the dataset prepared in this step.
# MAGIC
# MAGIC
# MAGIC <img src='https://hls-eng-data-public.s3.amazonaws.com/img/patient_risk_automl.gif'>

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📝 Note
# MAGIC As you can see in the autogenerated notebook, the accuracy of the model is very low. The primary reason for this is the fact that in the synthetically generated data, the emergency room visits are completely uncorrelated with CHF status or a patients disease history. However, we manually alterred the data to induce correlation between ethnicty, gender and drug exposure history with the outcome. This can be validated by the SHAP values generated in `Cmd 28` of the notebook. This is also reflected in the correlation matrix in [03-autoML-data-exploration]($./03-autoML-data-exploration) notebook. Also, note that most of the selected features have a very high skew (mostly zero).
