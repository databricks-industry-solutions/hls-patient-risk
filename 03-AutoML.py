# Databricks notebook source
import mlflow
import os
from databricks import feature_store
from databricks.feature_store import feature_table, FeatureLookup
fs = feature_store.FeatureStoreClient()

# COMMAND ----------

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

user_name=sql(f"SELECT current_user() as user").collect()[0]['user'].split('@')[0].replace('.','_')
schema_name = f"OMOP_{user_name}"
feature_schema = schema_name + '_features'
sql(f"USE {schema_name}")
print(schema_name)

# COMMAND ----------

# DBTITLE 1,Set Cohort IDs
target_cohort_id = 1
outcome_cohort_id = 2

outcome_att_id = 0
condition_hist_att_id = 1
drug_hist_att_id = 2

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

display(training_df)

# COMMAND ----------

# DBTITLE 1,Create the Training Dataset
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
training_df.fillna(0).write.mode("overwrite").saveAsTable(f'{feature_schema}.training_data')

# COMMAND ----------

# DBTITLE 1,Use automl to build an ML model 
import databricks.automl as db_automl

summary_cl = db_automl.classify(training_df, target_col="outcome", primary_metric="f1", timeout_minutes=5, experiment_dir = "/patientrisk/experiments/feature_store")
print(f"Best run id: {summary_cl.best_trial.mlflow_run_id}")

# COMMAND ----------

# DBTITLE 1,Name your model
model_name = "omop_patientrisk_model_v2"

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

# Load the ids we want to forecast
## For sake of simplicity, we will just predict using the same ids as during training, but this could be a different pipeline
subject_ids_to_forecast = spark.table(f'{feature_schema}.subject_demographics_features').select("subject_id").limit(100)
display(subject_ids_to_forecast)

# COMMAND ----------

# DBTITLE 1,Run inferences from a list of Subject IDs
scored_df = fs.score_batch(f"models:/{model_name}/Production", subject_ids_to_forecast, result_type="double")
display(scored_df)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC Notice that while we only selected a list of Subject IDs, the model returns our prediction (is this user likely to be admitted to the emergency room `True`/`False`) and the full list of features automatically retrieved from our feature table.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📝 Note
# MAGIC As you can see in the autogenerated notebook, the accuracy of the model is very low. The primary reason for this is the fact that in the synthetically generated data, the emergency room visits are completely uncorrelated with CHF status or a patients disease history. However, we manually alterred the data to induce correlation between ethnicty, gender and drug exposure history with the outcome. This can be validated by the SHAP values generated in `Cmd 28` of the notebook. This is also reflected in the correlation matrix in [03-autoML-data-exploration]($./03-autoML-data-exploration) notebook. Also, note that most of the selected features have a very high skew (mostly zero).
