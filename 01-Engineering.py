# Databricks notebook source
# MAGIC %md This solution accelerator notebook is also available at https://github.com/databricks-industry-solutions/hls-patient-risk

# COMMAND ----------

# MAGIC %md
# MAGIC # Patient Level Risk Prediction: Cohorts and Features
# MAGIC In this [01-Engineering]($01-Engineering) notebook we use data already available in OMOP 5.3 to create:
# MAGIC  1. Target cohort (patinets recently diagnosed with CHF)
# MAGIC  2. Outcome Cohort (patients adimitted to emergency room)
# MAGIC  3. Drug exposure history features
# MAGIC  4. Comorbidity history features
# MAGIC  5. Demographics features
# MAGIC
# MAGIC  The [02-OfflineFeatureStore]($02-OfflineFeatureStore)  notebook will create Offline Feature store tables from our OMOP schema.
# MAGIC
# MAGIC  The [03-AutoML]($03-AutoML) notebook will create a training dataset for "risk prediction" and train then register a classification model.
# MAGIC
# MAGIC  The [04-OnlineFeatureStore]($04-OnlineFeatureStore) notebook will create Online Feature store tables and a serverless model endpoint for real-time serving and automated feature lookup.
# MAGIC
# MAGIC ### Requirements
# MAGIC
# MAGIC * Cluster Runtime & Access Mode Requirements
# MAGIC   - Unity Catalog enabled workspace: Databricks Runtime 13.0 ML or above with the access mode **Assigned** or **No Isolation Shared**. 
# MAGIC   - Non-Unity Catalog enabled workspace: Databricks Runtime 13.0 ML or above with cluster mode **Standard**.
# MAGIC * Connection to Azure Cosmos DB for Online Feature Store
# MAGIC   - The cluster must have the Azure Cosmos DB connector for Spark installed. See the instructions in the section **Prepare the compute cluster**.
# MAGIC   - The last notebook uses Cosmos DB as the online store and guides you through how to generate secrets and register them with Databricks Secret Management.  

# COMMAND ----------

# MAGIC %md ## Prepare the compute cluster
# MAGIC
# MAGIC 1. When creating the compute cluster select Databricks Runtime ML 13.0 or above and choose access mode **Assigned** or **No Isolation Shared** for Unity Catalog workspaces, or **Standard** cluster mode for Non-Unity Catalog workspaces.
# MAGIC 1. After creating the cluster, follow these steps to install the [latest Azure Cosmos DB connector for Spark 3.2](https://github.com/Azure/azure-sdk-for-java/tree/main/sdk/cosmos/azure-cosmos-spark_3-3_2-12): 
# MAGIC     - Navigate to the compute cluster and click the **Libraries** tab.
# MAGIC     - Click **Install new**.
# MAGIC     - Click **Maven** and enter the coordinates of the latest version. For example: `com.azure.cosmos.spark:azure-cosmos-spark_3-2_2-12:4.18.2`
# MAGIC     - Click **Install**. 
# MAGIC 1. Attach the notebooks to the cluster.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Experiment parameters
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

# MAGIC %md
# MAGIC ## Create the OMOP Schema
# MAGIC First we create our OMOP schema based on the data available from databircks. This dataset is generated using synthea and our [OMOP solution accelerator](https://d1r5llqwmkrl74.cloudfront.net/notebooks/HLS/1-omop-cdm/index.html#1-omop-cdm_1.html).
# MAGIC Since we don't need all tables for this excersie, we only load those tables that are needed.

# COMMAND ----------

# DBTITLE 1,Define path with raw datasets
data_path= "s3://hls-eng-data-public/omop/synthetic-data/omop-gzip/"

# COMMAND ----------

# DBTITLE 1,List of available datasets
display(dbutils.fs.ls(data_path))

# COMMAND ----------

# DBTITLE 1,List of tables to load
tables = ["condition_occurrence","concept","concept_ancestor","observation_period","visit_occurrence","condition_era","drug_exposure","person"]

# COMMAND ----------

# DBTITLE 1,Dynamically define user_name based on who's running the notebook
user_name=sql(f"SELECT current_user() as user").collect()[0]['user'].split('@')[0].replace('.','_')
print(user_name)

# COMMAND ----------

# DBTITLE 1,Create new OMOP schema and load tables
schema_name = f"OMOP_{user_name}"
if drop_schema=='yes':
  sql(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE")
  sql(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")

  for table_name in tables:
    spark.read.csv(f"{data_path}{table_name}.csv",header=True,sep="\t", inferSchema=True).write.saveAsTable(f"{schema_name}.{table_name}")
    
print(schema_name)

# COMMAND ----------

# DBTITLE 1,List the tables created in our new OMOP schema
sql(f"SHOW TABLES in {schema_name}").display()

# COMMAND ----------

# DBTITLE 1,Set new OMOP as default schema to simplify code later
sql(f"USE {schema_name}")

# COMMAND ----------

# DBTITLE 1,Sample data from visit_occurrence
# MAGIC %sql
# MAGIC SELECT * FROM visit_occurrence limit 10

# COMMAND ----------

# MAGIC %md
# MAGIC ### Add derived elements
# MAGIC Now we add `cohort`, `cohort_attributes` and `cohort_definitions` to the schema

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE
# MAGIC OR REPLACE TABLE cohort (
# MAGIC   cohort_definition_id LONG,
# MAGIC   subject_id LONG,
# MAGIC   cohort_start_date DATE,
# MAGIC   cohort_end_date DATE
# MAGIC ) USING DELTA;

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE
# MAGIC OR REPLACE TABLE cohort_definition (
# MAGIC   cohort_definition_id LONG,
# MAGIC   cohort_definition_name STRING,
# MAGIC   cohort_definition_description STRING,
# MAGIC   definition_type_concept_id LONG,
# MAGIC   cohort_definition_syntax STRING,
# MAGIC   cohort_initiation_date DATE
# MAGIC ) USING DELTA;

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE
# MAGIC OR REPLACE TABLE COHORT_ATTRIBUTE (
# MAGIC   COHORT_DEFINITION_ID LONG,
# MAGIC   SUBJECT_ID LONG,
# MAGIC   COHORT_START_DATE DATE,
# MAGIC   COHORT_END_DATE DATE,
# MAGIC   ATTRIBUTE_DEFINITION_ID LONG,
# MAGIC   VALUE_AS_NUMBER DOUBLE,
# MAGIC   VALUE_AS_CONCEPT_ID LONG
# MAGIC ) USING DELTA;

# COMMAND ----------

# DBTITLE 1,Set up ids for cohorts
target_cohort_id = 1
outcome_cohort_id = 2

outcome_att_id = 0
condition_hist_att_id = 1
drug_hist_att_id = 2

# COMMAND ----------

# DBTITLE 1,List the concept names
# MAGIC %python
# MAGIC input_concepts = sql(f"""
# MAGIC select concept_id, concept_name from concept 
# MAGIC where concept_id in ({target_condition_concept_id},{drug1_concept_id},{drug2_concept_id}, {outcome_concept_id})
# MAGIC """)
# MAGIC display(input_concepts)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cohort Selection
# MAGIC In this section we define main cohorts, namely the target and outcome cohorts.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Target Cohort
# MAGIC
# MAGIC First we define the [target cohort](https://ohdsi.github.io/TheBookOfOhdsi/Cohorts.html), which is determind based on the following criteria:
# MAGIC
# MAGIC Patients who:
# MAGIC - are newly diagnosed with chronic congestive heart failure (CCHF)
# MAGIC - have a condition occurrence record of CCHF or any descendants, indexed at the first diagnosis (cohort entry date)
# MAGIC - have at least three years (1095 days) of prior observation before their first diagnosis
# MAGIC
# MAGIC For the target condition, we choose only to allow patients enter once at the earliest time they have been diagnosed. In the following query we also include ancestor concepts.

# COMMAND ----------

# DBTITLE 1,Create the condition cohort
# MAGIC %py
# MAGIC sql(f"""
# MAGIC CREATE OR REPLACE TABLE earliest_condition_onset AS (
# MAGIC   SELECT
# MAGIC     person_id,
# MAGIC     min(condition_start_date) as condition_start_date
# MAGIC   FROM
# MAGIC     condition_occurrence
# MAGIC   WHERE condition_concept_id IN (
# MAGIC       SELECT
# MAGIC         descendant_concept_id
# MAGIC       FROM
# MAGIC         concept_ancestor
# MAGIC       WHERE
# MAGIC         ancestor_concept_id = '{target_condition_concept_id}'
# MAGIC     )
# MAGIC   GROUP BY
# MAGIC     person_id
# MAGIC );
# MAGIC """)
# MAGIC display(sql("""SELECT count(*) from earliest_condition_onset"""))

# COMMAND ----------

# DBTITLE 1,Create the target cohort
# MAGIC %py
# MAGIC #Minimum of 3 years prior observation
# MAGIC #Minimum of 1 year available after observation
# MAGIC #Patients who have CHF
# MAGIC target_cohort_query =  f"""
# MAGIC SELECT
# MAGIC     {target_cohort_id} as cohort_definition_id,
# MAGIC     earliest_condition_onset.person_id AS subject_id,
# MAGIC     DATE_ADD(earliest_condition_onset.condition_start_date, ( -1 * {min_observation_period} ) )  AS cohort_start_date,
# MAGIC     earliest_condition_onset.condition_start_date AS cohort_end_date
# MAGIC   from
# MAGIC     earliest_condition_onset
# MAGIC     INNER JOIN (SELECT person_id, min(visit_start_date) as observation_period_start_date, max(visit_start_date) as observation_period_end_date  FROM visit_occurrence GROUP BY person_id) observation_period 
# MAGIC       ON earliest_condition_onset.person_id = observation_period.person_id
# MAGIC     AND earliest_condition_onset.condition_start_date > date_add(
# MAGIC       observation_period.observation_period_start_date, ( -1 * {min_observation_period} )
# MAGIC     )
# MAGIC     AND date_add(earliest_condition_onset.condition_start_date, {max_time_at_risk} ) < observation_period.observation_period_end_date
# MAGIC """
# MAGIC
# MAGIC cnt = sql(f'select count(*) as cnt from cohort').collect()[0]['cnt']
# MAGIC if cnt==0:
# MAGIC   sql(f'INSERT INTO cohort {target_cohort_query}')
# MAGIC else:
# MAGIC   sql(f'INSERT INTO cohort REPLACE WHERE cohort_definition_id = {target_cohort_id} {target_cohort_query}')

# COMMAND ----------

# DBTITLE 1,Sample from the target cohort
# MAGIC %py
# MAGIC sql(f"""select * 
# MAGIC from cohort 
# MAGIC where cohort_definition_id = {target_cohort_id}
# MAGIC limit 10""").display()

# COMMAND ----------

# DBTITLE 1,Count the subjects in target cohort
# MAGIC %py
# MAGIC sql(f"""select count(*) 
# MAGIC from cohort 
# MAGIC where cohort_definition_id = {target_cohort_id}
# MAGIC """).display()

# COMMAND ----------

# DBTITLE 1,Add target cohort information 
# MAGIC %py
# MAGIC target_cohort_concept_name = input_concepts.filter(f'concept_id = {target_condition_concept_id}' ).collect()[0]['concept_name']
# MAGIC target_cohort_description = f"""
# MAGIC persons with a condition occurrence record of {target_cohort_concept_name} or any descendants, indexed at the first diagnosis
# MAGIC who have >{min_observation_period} days of prior observation before their first diagnosis
# MAGIC """
# MAGIC
# MAGIC insert_query = f"select {target_cohort_id}, '{target_cohort_concept_name} Cohort', '{target_cohort_description}', 1, '{target_cohort_query}', current_date()"
# MAGIC cnt = sql(f'select count(*) as cnt from cohort_definition').collect()[0]['cnt']
# MAGIC
# MAGIC if cnt==0:
# MAGIC   sql(f"INSERT INTO cohort_definition {insert_query}")
# MAGIC else:
# MAGIC   sql(f"INSERT INTO cohort_definition REPLACE WHERE cohort_definition_id = {target_cohort_id} {insert_query}")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * from cohort_definition LIMIT 10 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Outcome Cohort
# MAGIC Similary we can create an outcome cohort from our target cohort

# COMMAND ----------

# MAGIC %py
# MAGIC sql(f'SET outcome_cohort_id = {outcome_cohort_id}')
# MAGIC
# MAGIC
# MAGIC outcome_cohort_query =  f"""
# MAGIC   SELECT
# MAGIC     {outcome_cohort_id} AS cohort_definition_id,
# MAGIC     visit_occurrence.person_id AS subject_id,
# MAGIC     cohort.cohort_end_date AS cohort_start_date, --Diagnosis of CHF 
# MAGIC     MIN(visit_occurrence.visit_end_date) AS cohort_end_date --first er admission
# MAGIC   FROM
# MAGIC     visit_occurrence
# MAGIC       INNER JOIN cohort 
# MAGIC         ON cohort_definition_id = {target_cohort_id}
# MAGIC         AND visit_occurrence.person_id = cohort.subject_id
# MAGIC   WHERE
# MAGIC     visit_occurrence.visit_concept_id IN ({outcome_concept_id}) --er admission
# MAGIC       and visit_occurrence.visit_start_date BETWEEN cohort.cohort_end_date AND date_add(cohort.cohort_end_date, {max_time_at_risk})   
# MAGIC       --Inp admission after CHF diagnosis and before max time at risk
# MAGIC   GROUP BY
# MAGIC     cohort_definition_id,
# MAGIC     visit_occurrence.person_id,
# MAGIC     cohort.cohort_end_date
# MAGIC """
# MAGIC
# MAGIC cnt = sql(f'select count(*) as cnt from cohort where cohort_definition_id = {outcome_cohort_id}').collect()[0]['cnt']
# MAGIC
# MAGIC if cnt==0:
# MAGIC   sql(f'INSERT INTO cohort {outcome_cohort_query}')
# MAGIC else:
# MAGIC   sql(f'INSERT INTO cohort REPLACE WHERE cohort_definition_id = {outcome_cohort_id} {outcome_cohort_query}')

# COMMAND ----------

# DBTITLE 1,Inspect the outcome cohort
# MAGIC %py
# MAGIC sql(f"""select * 
# MAGIC from cohort 
# MAGIC where cohort_definition_id ={outcome_cohort_id}
# MAGIC limit(10)""").display()

# COMMAND ----------

#number of events of ER admission from cohort
sql(f"""
select count(*) 
from cohort 
where cohort_definition_id = {outcome_cohort_id}""").display()

# COMMAND ----------

# DBTITLE 1,Update our Cohort_Definition to include "Outcome Cohort" aka ER Visits
# MAGIC %py
# MAGIC outcome_cohort_concept_name = input_concepts.filter(f'concept_id = {outcome_concept_id}' ).collect()[0]['concept_name']
# MAGIC outcome_cohort_description = f"""
# MAGIC persons with at lease once occurance of {outcome_cohort_concept_name}
# MAGIC """
# MAGIC insert_query = f"select {outcome_cohort_id}, '{outcome_cohort_concept_name} Cohort', '{outcome_cohort_description}', 1, '{outcome_cohort_query}' , current_date()"
# MAGIC cnt = sql(f'select count(*) as cnt from cohort_definition where cohort_definition_id = {outcome_cohort_id}').collect()[0]['cnt']
# MAGIC if cnt==0:
# MAGIC   sql(f"INSERT INTO cohort_definition {insert_query}")
# MAGIC else:
# MAGIC   sql(f"INSERT OVERWRITE cohort_definition WHERE cohort_id = {outcome_cohort_id} {insert_query}")

# COMMAND ----------

# DBTITLE 1,View Cohort_Definition
# MAGIC %sql
# MAGIC select * from cohort_definition limit 10  

# COMMAND ----------

# MAGIC %md
# MAGIC ## Patient Cohort Attributes
# MAGIC Now that we have our cohorts in place, we add data to the [cohort_attributes](https://www.ohdsi.org/web/wiki/doku.php?id=documentation:cdm:cohort_attribute) table, these attributes are selected based on the downstream analysis for this excersise, which is creating a feature store for risk prediction. 

# COMMAND ----------

# MAGIC %md
# MAGIC Since we will be comparing dates often, it is better to simplify downstream queries by defining a sql function that retruns whether a given date is within a range

# COMMAND ----------

# DBTITLE 1,Function to validate time overlap
# MAGIC %sql
# MAGIC CREATE
# MAGIC OR REPLACE TEMPORARY FUNCTION is_valid_time_overlap (
# MAGIC   cohort1_start DATE,
# MAGIC   cohort1_end DATE,
# MAGIC   cohort2_start DATE,
# MAGIC   cohort2_end DATE,
# MAGIC   start_date_offset INT,
# MAGIC   end_date_offset INT)
# MAGIC RETURNS BOOLEAN 
# MAGIC RETURN
# MAGIC DATE_ADD(cohort1_start, start_date_offset) < cohort2_start
# MAGIC AND 
# MAGIC cohort2_end < DATE_ADD(cohort1_end, end_date_offset)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Outcome Attribute
# MAGIC First, we add the outcome attributes for patients in the target cohort, this is where we determine which of the patients within the cohort experienced the outcome in question within the timeframe of interest. 

# COMMAND ----------

# MAGIC %py
# MAGIC insert_query = f"""
# MAGIC select
# MAGIC   distinct
# MAGIC   tc.cohort_definition_id,
# MAGIC   tc.subject_id,
# MAGIC   tc.cohort_start_date,
# MAGIC   tc.cohort_end_date,
# MAGIC   {outcome_att_id} as ATTRIBUTE_DEFINITION_ID,
# MAGIC   1 as VALUE_AS_NUMBER,
# MAGIC   vo.VISIT_CONCEPT_ID as VALUE_AS_CONCEPT_ID
# MAGIC from
# MAGIC   cohort tc
# MAGIC   join visit_occurrence vo on tc.subject_id = vo.PERSON_ID
# MAGIC where
# MAGIC   tc.cohort_definition_id = {target_cohort_id}
# MAGIC   AND vo.VISIT_CONCEPT_ID = {outcome_concept_id}
# MAGIC   AND is_valid_time_overlap(tc.cohort_start_date, tc.cohort_start_date, vo.visit_start_date, vo.visit_start_date, {min_time_at_risk}, {max_time_at_risk})
# MAGIC """
# MAGIC
# MAGIC cnt = sql(f'select count(*) as cnt from COHORT_ATTRIBUTE where cohort_definition_id = {target_cohort_id}').collect()[0]['cnt']
# MAGIC if cnt==0:
# MAGIC   sql(f"INSERT INTO COHORT_ATTRIBUTE {insert_query}")
# MAGIC else:
# MAGIC   sql(f"INSERT INTO COHORT_ATTRIBUTE WHERE ATTRIBUTE_DEFINITION_ID={outcome_att_id} {insert_query}")

# COMMAND ----------

# DBTITLE 1,Inspect the subset of target cohort which experienced the outcome within the timeframe
sql(f'select * from COHORT_ATTRIBUTE where ATTRIBUTE_DEFINITION_ID={outcome_att_id}').display()

# COMMAND ----------

# DBTITLE 1,Count them
sql(f'select count(*) from COHORT_ATTRIBUTE where ATTRIBUTE_DEFINITION_ID={outcome_att_id}').display()


# COMMAND ----------

# MAGIC %md
# MAGIC ### Condition History
# MAGIC Now we add condition history attributes, corresponding to the top `n` comorbidities found among the target cohort members. Note that a cohort member is considered to have the given attribute if they were diagnosed with the condition within the observation period. 

# COMMAND ----------

# MAGIC %py
# MAGIC insert_query = f"""
# MAGIC select
# MAGIC   tc.cohort_definition_id,
# MAGIC   tc.subject_id,
# MAGIC   tc.cohort_start_date,
# MAGIC   tc.cohort_end_date,
# MAGIC   {condition_hist_att_id} as ATTRIBUTE_DEFINITION_ID,
# MAGIC   sum(ce.CONDITION_OCCURRENCE_COUNT) as VALUE_AS_NUMBER,
# MAGIC   ce.CONDITION_CONCEPT_ID as VALUE_AS_CONCEPT_ID
# MAGIC from
# MAGIC   cohort tc
# MAGIC   join condition_era ce on tc.subject_id = ce.PERSON_ID
# MAGIC where
# MAGIC   tc.cohort_definition_id = {target_cohort_id}
# MAGIC   AND is_valid_time_overlap(
# MAGIC     ce.CONDITION_ERA_START_DATE,
# MAGIC     ce.CONDITION_ERA_START_DATE,
# MAGIC     tc.cohort_start_date,
# MAGIC     tc.cohort_start_date,
# MAGIC     0,
# MAGIC     5*{cond_history_years}
# MAGIC   )
# MAGIC group by
# MAGIC   tc.cohort_definition_id,
# MAGIC   tc.subject_id,
# MAGIC   tc.cohort_start_date,
# MAGIC   tc.cohort_end_date,
# MAGIC   ce.CONDITION_CONCEPT_ID
# MAGIC """
# MAGIC cnt = sql(f'select count(*) as cnt from COHORT_ATTRIBUTE where cohort_definition_id = {target_cohort_id}').collect()[0]['cnt']
# MAGIC if cnt==0:
# MAGIC   sql(f"INSERT INTO COHORT_ATTRIBUTE {insert_query}")
# MAGIC else:
# MAGIC   sql(f"INSERT INTO COHORT_ATTRIBUTE REPLACE WHERE ATTRIBUTE_DEFINITION_ID={condition_hist_att_id} {insert_query}")

# COMMAND ----------

sql(f'select * from COHORT_ATTRIBUTE where ATTRIBUTE_DEFINITION_ID={condition_hist_att_id}').display()

# COMMAND ----------

# DBTITLE 1,Count them
sql(f'select count(*) from COHORT_ATTRIBUTE where ATTRIBUTE_DEFINITION_ID={condition_hist_att_id}').display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Drug Exposure History
# MAGIC Similar to the condition history, we add the drug exposure history:

# COMMAND ----------

# MAGIC %py
# MAGIC insert_query = f"""select
# MAGIC   tc.cohort_definition_id,
# MAGIC   tc.subject_id,
# MAGIC   tc.cohort_start_date,
# MAGIC   tc.cohort_end_date,
# MAGIC   {drug_hist_att_id} as ATTRIBUTE_DEFINITION_ID,
# MAGIC   sum(de.QUANTITY)+1 as VALUE_AS_NUMBER,
# MAGIC   de.DRUG_CONCEPT_ID as VALUE_AS_CONCEPT_ID
# MAGIC from
# MAGIC   cohort tc
# MAGIC   join drug_exposure de on tc.subject_id = de.PERSON_ID
# MAGIC where
# MAGIC   tc.cohort_definition_id = {target_cohort_id}
# MAGIC   and de.DRUG_CONCEPT_ID in ({drug1_concept_id}, {drug2_concept_id})
# MAGIC   AND is_valid_time_overlap(
# MAGIC     tc.cohort_start_date,
# MAGIC     tc.cohort_end_date,
# MAGIC     de.DRUG_EXPOSURE_START_DATE,
# MAGIC     de.DRUG_EXPOSURE_END_DATE,
# MAGIC     0,
# MAGIC     0
# MAGIC   )
# MAGIC group by
# MAGIC   tc.cohort_definition_id,
# MAGIC   tc.subject_id,
# MAGIC   tc.cohort_start_date,
# MAGIC   tc.cohort_end_date,
# MAGIC   de.DRUG_CONCEPT_ID
# MAGIC   """
# MAGIC
# MAGIC cnt = sql(f'select count(*) as cnt from COHORT_ATTRIBUTE where ATTRIBUTE_DEFINITION_ID = {drug_hist_att_id}').collect()[0]['cnt']
# MAGIC if cnt==0:
# MAGIC   sql(f"INSERT INTO COHORT_ATTRIBUTE {insert_query}")
# MAGIC else:
# MAGIC   sql(f"INSERT INTO COHORT_ATTRIBUTE REPLACE WHERE ATTRIBUTE_DEFINITION_ID={drug_hist_att_id} {insert_query}")

# COMMAND ----------

# DBTITLE 1,Count them
# MAGIC %py
# MAGIC sql(f'select count(*) as cnt from COHORT_ATTRIBUTE where ATTRIBUTE_DEFINITION_ID = {drug_hist_att_id}').display()
