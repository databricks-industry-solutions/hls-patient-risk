# Databricks notebook source
# MAGIC %md This solution accelerator notebook is also available at https://github.com/databricks-industry-solutions/hls-patient-risk

# COMMAND ----------

dbutils.widgets.removeAll()

# COMMAND ----------

# MAGIC %md
# MAGIC # Patient Level Risk Prediction: Cohorts and Features
# MAGIC In this notebook we use data already available in OMOP 5.3 to create:
# MAGIC  1. Target cohort (patinets recently diagnosed with CHF)
# MAGIC  2. Outcome Cohort (patients adimitted to emergency room)
# MAGIC  3. Drug exposure history features
# MAGIC  4. Commorbidity history features
# MAGIC  5. Demographics features
# MAGIC 
# MAGIC  and create a training dataset that will be used for risk prediction.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Experiment parameters
# MAGIC First we set up the paramters for the experiment using databricks notebooks widgets utility.

# COMMAND ----------

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

# MAGIC %md
# MAGIC ## Create the OMOP Schema
# MAGIC First we create our OMOP schema based on the data available from databircks. This dataset is generated using synthea and our [OMOP solution accelerator](https://d1r5llqwmkrl74.cloudfront.net/notebooks/HLS/1-omop-cdm/index.html#1-omop-cdm_1.html).
# MAGIC Since we don't need all tables for this excersie, we only load those tables that are needed.

# COMMAND ----------

data_path= "s3://hls-eng-data-public/omop/synthetic-data/omop-gzip/"

# COMMAND ----------

# DBTITLE 1,list of available datasets
display(dbutils.fs.ls(data_path))

# COMMAND ----------

# DBTITLE 1,list of tables to load
tables = ["condition_occurrence","concept","concept_ancestor","observation_period","visit_occurrence","condition_era","drug_exposure","person"]

# COMMAND ----------


user_name=sql(f"SELECT current_user() as user").collect()[0]['user'].split('@')[0].replace('.','_')

# COMMAND ----------

# DBTITLE 1,create a new omop schema and load tables
schema_name = f"OMOP_{user_name}"

if drop_schema=='yes':
  sql(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE")
  sql(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")

  for table_name in tables:
    spark.read.csv(f"{data_path}{table_name}.csv",header=True,sep="\t", inferSchema=True).write.saveAsTable(f"{schema_name}.{table_name}")
  

# COMMAND ----------

# DBTITLE 1,list tables
sql(f"SHOW TABLES in {schema_name}").display()

# COMMAND ----------

sql(f"USE {schema_name}")

# COMMAND ----------

# DBTITLE 1,example data for visit_occurrence
# MAGIC %sql
# MAGIC SELECT * FROM visit_occurrence limit 10

# COMMAND ----------

# MAGIC %md
# MAGIC ### Add derived elements
# MAGIC Now we add `cohort`, `cohort_attributes` and `cohort_definitions` to the scjhema

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

# DBTITLE 1,set up ids for cohorts
target_cohort_id = 1
outcome_cohort_id = 2

outcome_att_id = 0
condition_hist_att_id = 1
drug_hist_att_id = 2

# COMMAND ----------

# DBTITLE 1,list the concept names
# MAGIC %py
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
# MAGIC Patients who are newly:
# MAGIC - diagnosed with chronic congestive heart failure (CCHF)
# MAGIC - persons with a condition occurrence record of CCHF or any descendants, indexed at the first diagnosis (cohort entry date)
# MAGIC - who have at least three years (1095 days) of prior observation before their first diagnosis
# MAGIC 
# MAGIC For the target condition, we choose only to allow patients enter once at the earliest time they have been diagnosed. In the following query we also include ancestor concepts.

# COMMAND ----------

# DBTITLE 1,Create condition cohort
# MAGIC %py
# MAGIC sql(f"""DROP TABLE IF EXISTS earliest_condition_onset;""")
# MAGIC sql(f"""
# MAGIC CREATE TABLE earliest_condition_onset AS (
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

# DBTITLE 1,create target Cohort
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

# DBTITLE 1,target cohort
# MAGIC %py
# MAGIC sql(f"""select * 
# MAGIC from cohort 
# MAGIC where cohort_definition_id = {target_cohort_id}
# MAGIC limit 10""").display()

# COMMAND ----------

# DBTITLE 1,count of subject in target control
# MAGIC %py
# MAGIC sql(f"""select count(*) 
# MAGIC from cohort 
# MAGIC where cohort_definition_id = {target_cohort_id}
# MAGIC """).display()

# COMMAND ----------

# DBTITLE 1,add target cohort information 
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
# MAGIC ### Outcome cohort
# MAGIC similary we can create an outcome cohort

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
# MAGIC       INNER JOIN omop_patient_risk.cohort 
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

# DBTITLE 1,function to validate time overlap
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
# MAGIC First, we add the outcome attributes for patinets in the target cohort, this is where we determine which of the pateints withing the cohort, have expereinced the outcome in question within the timeframe of interest. 

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

sql(f'select * from COHORT_ATTRIBUTE where ATTRIBUTE_DEFINITION_ID={outcome_att_id}').display()

# COMMAND ----------

sql(f'select count(*) from COHORT_ATTRIBUTE where ATTRIBUTE_DEFINITION_ID={outcome_att_id}').display()


# COMMAND ----------

# MAGIC %md
# MAGIC ### Condition History
# MAGIC Now we add condition history attributes, corresponding to the top `n` comorbidities found among the target cohort members. Note that a cohrot member is considered to have the given attribute if she has been diagnosed with the condition within the observation period. 

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

sql(f'select count(*) from COHORT_ATTRIBUTE where ATTRIBUTE_DEFINITION_ID={condition_hist_att_id}').display()


# COMMAND ----------

sql(f'select * from COHORT_ATTRIBUTE where ATTRIBUTE_DEFINITION_ID={condition_hist_att_id}').display()


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

# MAGIC %py
# MAGIC sql(f'select count(*) as cnt from COHORT_ATTRIBUTE where ATTRIBUTE_DEFINITION_ID = {drug_hist_att_id}').display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Add features
# MAGIC So far, we have leveraged OMOP's default schemas to store attributes associated to the target cohort that can be used for training our model. 
# MAGIC We now, leverage [databricks feature store](https://docs.databricks.com/machine-learning/feature-store/index.html) to create an offline feature store to store features that can be used (and re-used) for training our classifier.

# COMMAND ----------

from databricks import feature_store
from databricks.feature_store import feature_table, FeatureLookup

# COMMAND ----------

fs = feature_store.FeatureStoreClient()

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

# COMMAND ----------

# DBTITLE 1,add drug features to feature store
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
# MAGIC ## Create Training Dataset
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

training_df = training_set.load_df()

training_df.selectExpr('avg(outcome)').display()

# COMMAND ----------

training_df.display()

# COMMAND ----------

# DBTITLE 1,store the dataset
training_df.fillna(0).write.saveAsTable(f'{feature_schema}.training_data')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Tarining
# MAGIC Now that we have the training data ready, we will proceed to use databricks [AutoML]() to train a binary classifier that predicts the outcome (emergency room visit status) based on the selected features that we have stored in the feature store.
# MAGIC The next notebook ([02-automl-best-model]($./02-automl-best-model)) is an example notebook generated by AutoML based on the dataset prepard in this step.
# MAGIC 
# MAGIC 
# MAGIC <img src='https://hls-eng-data-public.s3.amazonaws.com/img/patient_risk_automl.gif'>

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📝 Note
# MAGIC As you can see in the autogenerated notebook, the accuracy of the model is very low. The primary reason for this is the fact that in the synthetically generated data, the emergency room visits are completely uncorrelated with CHF status or a patients disease history. However, we manually alterred the data to induce correlation between ethnicty, gender and drug exposure history with the outcome. This can be validated by the SHAP values generated in `Cmd 28` of the notebook. This is also reflected in the correlation matrix in [03-autoML-data-exploration]($./03-autoML-data-exploration) notebook. Also, note that most of the selected features have a very high skew (mostly zero).
