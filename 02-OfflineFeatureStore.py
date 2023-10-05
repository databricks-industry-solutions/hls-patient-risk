# Databricks notebook source
# MAGIC %md
# MAGIC # Experiment parameters
# MAGIC First we set up the paramters for the experiment using databricks notebooks widgets utility.

# COMMAND ----------

# DBTITLE 1,Create widgets with values
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
dbutils.widgets.text('max_n_comorbidities','5')

# COMMAND ----------

# DBTITLE 1,Create Variables using widget values
drop_schema = dbutils.widgets.get('drop_schema')

target_condition_concept_id = dbutils.widgets.get('target_condition_concept_id')
outcome_concept_id = dbutils.widgets.get('outcome_concept_id')

drug1_concept_id = dbutils.widgets.get('drug1_concept_id')
drug2_concept_id = dbutils.widgets.get('drug2_concept_id')

min_observation_period = dbutils.widgets.get('min_observation_period')
min_time_at_risk = dbutils.widgets.get('min_time_at_risk')
max_time_at_risk = dbutils.widgets.get('max_time_at_risk')

cond_history_years = dbutils.widgets.get('cond_history_years')
max_n_comorbidities = dbutils.widgets.get('max_n_comorbidities')

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

# DBTITLE 1,Import Feature Store Libraries
from databricks import feature_store
from databricks.feature_store import feature_table, FeatureLookup
fs = feature_store.FeatureStoreClient()

# COMMAND ----------

# DBTITLE 1,Dynamically define user_name and schema_name
user_name=sql(f"SELECT current_user() as user").collect()[0]['user'].split('@')[0].replace('.','_')
schema_name = f"OMOP_{user_name}"
sql(f"USE {schema_name}")
print(schema_name)

# COMMAND ----------

# DBTITLE 1,List tables made in 01-Engineering notebook
sql(f"SHOW TABLES IN {schema_name}").display()

# COMMAND ----------

# DBTITLE 1,Create a new schema to hold the features as well as the training dataset for this analysis
feature_schema = schema_name + '_features'
sql(f"DROP SCHEMA IF EXISTS {feature_schema} CASCADE")
sql(f"CREATE SCHEMA IF NOT EXISTS {feature_schema}")
print(feature_schema)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Feature Store: Drug Exposure History

# COMMAND ----------

# DBTITLE 1,Create a feature store for the drug history features
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

# DBTITLE 1,Select from Feature Store
sql(f'select * from {FEATURE_TABLE_NAME} limit 10').display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Feature Store: Comorbidity History

# COMMAND ----------

# DBTITLE 1,View top n comorbidities
# MAGIC %py
# MAGIC sql(f"""
# MAGIC select VALUE_AS_CONCEPT_ID as condition_concept_id, count(*) as cnt from cohort_attribute where ATTRIBUTE_DEFINITION_ID={condition_hist_att_id} group by 1
# MAGIC order by 2 desc
# MAGIC limit {max_n_comorbidities}
# MAGIC """).createOrReplaceTempView('top_comorbidities')

# COMMAND ----------

FEATURE_TABLE_NAME = f'{feature_schema}.condition_history_features'
description=f"condition history features for top {max_n_comorbidities} comorbidities"
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
# MAGIC ### Create Feature Store: Demographics Information 

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

# DBTITLE 1,Select from Feature Store
sql(f'select * from {FEATURE_TABLE_NAME} limit 10').display()
