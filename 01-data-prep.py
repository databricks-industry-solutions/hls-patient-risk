# Databricks notebook source
# MAGIC %md
# MAGIC # Patient Level Risk Prediction: Cohorts and Features
# MAGIC In this notebook we use data already available in OMOP 5.3 to create:
# MAGIC  1. Target cohort (patinets recently diagnosed with CHF)
# MAGIC  2. Outcome Cohort (patients adimitted to emergency room)
# MAGIC  3. Drug exposure hostory features
# MAGIC  4. Commorbidity history features
# MAGIC  5. Demographics features

# COMMAND ----------

# MAGIC %md
# MAGIC dbutils.widgets.removeAll()

# COMMAND ----------

# MAGIC %sql
# MAGIC USE OMOP531 -- todo maybe change 

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP SCHEMA IF EXISTS omop_patient_risk CASCADE

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE SCHEMA IF NOT EXISTS omop_patient_risk

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE
# MAGIC OR REPLACE TABLE omop_patient_risk.cohort (
# MAGIC   cohort_definition_id LONG,
# MAGIC   subject_id LONG,
# MAGIC   cohort_start_date DATE,
# MAGIC   cohort_end_date DATE
# MAGIC ) USING DELTA;

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE
# MAGIC OR REPLACE TABLE omop_patient_risk.cohort_definition (
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
# MAGIC OR REPLACE TABLE omop_patient_risk.COHORT_ATTRIBUTE (
# MAGIC   COHORT_DEFINITION_ID LONG,
# MAGIC   SUBJECT_ID LONG,
# MAGIC   COHORT_START_DATE DATE,
# MAGIC   COHORT_END_DATE DATE,
# MAGIC   ATTRIBUTE_DEFINITION_ID LONG,
# MAGIC   VALUE_AS_NUMBER DOUBLE,
# MAGIC   VALUE_AS_CONCEPT_ID LONG
# MAGIC ) USING DELTA;

# COMMAND ----------

# DBTITLE 1,Set up parameters
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
# MAGIC CREATE WIDGET text max_n_commorbidities DEFAULT "10";

# COMMAND ----------

# MAGIC %py
# MAGIC drug1_concept_id = dbutils.widgets.get('drug1_concept_id')
# MAGIC drug2_concept_id = dbutils.widgets.get('drug2_concept_id')
# MAGIC outcome_concept_id = dbutils.widgets.get('outcome_concept_id')
# MAGIC min_observation_period = dbutils.widgets.get('min_observation_period')
# MAGIC 
# MAGIC min_time_at_risk = dbutils.widgets.get('min_time_at_risk')
# MAGIC max_time_at_risk= dbutils.widgets.get('max_time_at_risk')
# MAGIC cond_history_years= dbutils.widgets.get('cond_history_years')
# MAGIC max_n_commorbidities= dbutils.widgets.get('max_n_commorbidities')
# MAGIC 
# MAGIC target_condition_concept_id = dbutils.widgets.get('target_condition_concept_id')

# COMMAND ----------

target_cohort_id = 1
outcome_cohort_id = 2

outcome_att_id = 0
condition_hist_att_id = 1
drug_hist_att_id = 2

# COMMAND ----------

# MAGIC %py
# MAGIC input_concepts = sql(f"""
# MAGIC select concept_id, concept_name from concept 
# MAGIC where concept_id in ({target_condition_concept_id},{drug1_concept_id},{drug2_concept_id}, {outcome_concept_id})
# MAGIC """)
# MAGIC display(input_concepts)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Target Cohort
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
# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS earliest_condition_onset;
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
# MAGIC         ancestor_concept_id = ${target_condition_concept_id}
# MAGIC     )
# MAGIC   GROUP BY
# MAGIC     person_id
# MAGIC );
# MAGIC SELECT
# MAGIC   count(*)
# MAGIC from
# MAGIC   earliest_condition_onset

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
# MAGIC cnt = sql(f'select count(*) as cnt from omop_patient_risk.cohort').collect()[0]['cnt']
# MAGIC if cnt==0:
# MAGIC   sql(f'INSERT INTO omop_patient_risk.cohort {target_cohort_query}')
# MAGIC else:
# MAGIC   sql(f'INSERT INTO omop_patient_risk.cohort REPLACE WHERE cohort_definition_id = {target_cohort_id} {target_cohort_query}')

# COMMAND ----------

# DBTITLE 1,target cohort
# MAGIC %py
# MAGIC sql(f"""select * 
# MAGIC from omop_patient_risk.cohort 
# MAGIC where cohort_definition_id = {target_cohort_id}
# MAGIC limit 10""").display()

# COMMAND ----------

# DBTITLE 1,count of subject in target control
# MAGIC %py
# MAGIC sql(f"""select count(*) 
# MAGIC from omop_patient_risk.cohort 
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
# MAGIC cnt = sql(f'select count(*) as cnt from omop_patient_risk.cohort_definition').collect()[0]['cnt']
# MAGIC 
# MAGIC if cnt==0:
# MAGIC   sql(f"INSERT INTO omop_patient_risk.cohort_definition {insert_query}")
# MAGIC else:
# MAGIC   sql(f"INSERT INTO omop_patient_risk.cohort_definition REPLACE WHERE cohort_definition_id = {target_cohort_id} {insert_query}")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * from omop_patient_risk.cohort_definition LIMIT 10 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Outcome cohort
# MAGIC Similary we can create an outcome cohort.
# MAGIC 
# MAGIC Everyone who has been diagnosised with CHF, had appropriate observability window, and has been admitted to ER

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
# MAGIC     MIN(visit_occurrence.visit_end_date) AS cohort_end_date --First ER Admission
# MAGIC   FROM
# MAGIC     visit_occurrence
# MAGIC       INNER JOIN omop_patient_risk.cohort 
# MAGIC         ON cohort_definition_id = {target_cohort_id}
# MAGIC         AND visit_occurrence.person_id = cohort.subject_id
# MAGIC   WHERE
# MAGIC     visit_occurrence.visit_concept_id IN ({outcome_concept_id}) --inpatient admission
# MAGIC       and visit_occurrence.visit_start_date BETWEEN cohort.cohort_end_date AND date_add(cohort.cohort_end_date, {max_time_at_risk})   
# MAGIC       --Inp admission after CHF diagnosis and before max time at risk
# MAGIC   GROUP BY
# MAGIC     cohort_definition_id,
# MAGIC     visit_occurrence.person_id,
# MAGIC     cohort.cohort_end_date
# MAGIC """
# MAGIC 
# MAGIC cnt = sql(f'select count(*) as cnt from omop_patient_risk.cohort where cohort_definition_id = {outcome_cohort_id}').collect()[0]['cnt']
# MAGIC 
# MAGIC if cnt==0:
# MAGIC   sql(f'INSERT INTO omop_patient_risk.cohort {outcome_cohort_query}')
# MAGIC else:
# MAGIC   sql(f'INSERT INTO omop_patient_risk.cohort REPLACE WHERE cohort_definition_id = {outcome_cohort_id} {outcome_cohort_query}')

# COMMAND ----------

# MAGIC %py
# MAGIC sql(f"""select * 
# MAGIC from omop_patient_risk.cohort 
# MAGIC where cohort_definition_id ={outcome_cohort_id}
# MAGIC limit(10)""").display()

# COMMAND ----------

#number of events of ER admission from cohort
sql(f"""
select count(*) as num_events, count(distinct subject_id) as num_members
from omop_patient_risk.cohort 
where cohort_definition_id = {outcome_cohort_id}
""").display()

# COMMAND ----------

# MAGIC %py
# MAGIC outcome_cohort_concept_name = input_concepts.filter(f'concept_id = {outcome_concept_id}' ).collect()[0]['concept_name']
# MAGIC outcome_cohort_description = f"""
# MAGIC persons with at lease once occurance of {outcome_cohort_concept_name}
# MAGIC """
# MAGIC insert_query = f"select {outcome_cohort_id}, '{outcome_cohort_concept_name} Cohort', '{outcome_cohort_description}', 1, '{outcome_cohort_query}' , current_date()"
# MAGIC cnt = sql(f'select count(*) as cnt from omop_patient_risk.cohort_definition where cohort_definition_id = {outcome_cohort_id}').collect()[0]['cnt']
# MAGIC if cnt==0:
# MAGIC   sql(f"INSERT INTO omop_patient_risk.cohort_definition {insert_query}")
# MAGIC else:
# MAGIC   sql(f"INSERT OVERWRITE omop_patient_risk.cohort_definition WHERE cohort_id = {outcome_cohort_id} {insert_query}")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from omop_patient_risk.cohort_definition limit 10

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Engineering
# MAGIC At this point, we've created our Target & Outcome cohorts  
# MAGIC 
# MAGIC Now we want to develop features that are relevant to CHF + ER admission. This is centered around patient attributes and relevant medical history

# COMMAND ----------

# MAGIC %md
# MAGIC ### Patient, Condition, and Drug  Features

# COMMAND ----------

# MAGIC %sql
# MAGIC   --Person Attributes 
# MAGIC CREATE TABLE IF NOT EXISTS omop531.person_features_offline as
# MAGIC  select  person_id
# MAGIC  ,year_of_birth
# MAGIC  ,race_concept_id
# MAGIC  ,gender_source_value
# MAGIC  FROM omop531.person
# MAGIC  ;
# MAGIC  
# MAGIC  --Medical Condition Attributes
# MAGIC CREATE TABLE IF NOT EXISTS omop531.condition_features_offline as
# MAGIC SELECT person_id, condition_era_start_date
# MAGIC ,SUM(`4112343`) as `4112343`, SUM(`4289517`) as `4289517`
# MAGIC ,SUM(`432867`) as `432867`, SUM(`4237458`) as `4237458`, SUM(`260139`) as `260139`
# MAGIC ,SUM(`312437`) as `312437`, SUM(`254761`) as `254761`, SUM(`40481087`) as `40481087`
# MAGIC ,SUM(`80502`) as `80502`, SUM(`437663`) as `437663`
# MAGIC FROM (
# MAGIC SELECT person_id, condition_era_start_date, NVL(`4112343`, 0) as `4112343`, NVL(`4289517`,0)  as `4289517`
# MAGIC ,NVL(`432867`, 0) as `432867`, NVL(`4237458`, 0) as `4237458`, NVL(`260139`, 0) as `260139`
# MAGIC ,NVL(`312437`, 0) as `312437`, NVL(`254761`, 0) as `254761`, NVL(`40481087`, 0) as `40481087`
# MAGIC ,NVL(`80502`, 0) as  `80502`, NVL(`437663`, 0) as `437663`
# MAGIC FROM omop531.CONDITION_ERA
# MAGIC PIVOT(
# MAGIC 	SUM(CONDITION_OCCURRENCE_COUNT) as VALUE_AS_NUMBER
# MAGIC 	FOR CONDITION_CONCEPT_ID IN ("4112343", "4289517", "432867", "4237458", "260139", "312437", "254761", "40481087", "80502", "437663")
# MAGIC )
# MAGIC ) FOO 
# MAGIC GROUP BY person_id, condition_era_start_date
# MAGIC ;
# MAGIC --Drug Attributes
# MAGIC CREATE TABLE IF NOT EXISTS omop531.drug_features_offline as
# MAGIC select person_id, drug_exposure_start_date, SUM(`40163554`) as `40163554`, SUM(`40221901`) as `40221901`
# MAGIC FROM
# MAGIC (
# MAGIC select person_id, drug_exposure_start_date, NVL(`40163554`, 0) as `40163554` , NVL(`40221901`, 0) as `40221901`
# MAGIC from omop531.drug_exposure person_rx
# MAGIC PIVOT(
# MAGIC 	count(drug_concept_id)
# MAGIC 	FOR drug_concept_id IN  ("40163554", "40221901")
# MAGIC )
# MAGIC ) FOO
# MAGIC GROUP BY person_id, drug_exposure_start_date
# MAGIC ;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Display our Model Feature Tables

# COMMAND ----------

display(spark.table("omop531.person_features_offline"))
display(spark.table("omop531.condition_features_offline"))
display(spark.table("omop531.drug_features_offline"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create our Offline Feature Store (Point in Time Datasets)

# COMMAND ----------

from databricks import feature_store
from databricks.feature_store.client import FeatureStoreClient
from databricks.feature_store import FeatureLookup

person_df = spark.table("omop531.person_features_offline")
person_medical_condition_df = spark.table("omop531.condition_features_offline")
person_rx_df = spark.table("omop531.drug_features_offline")

fs = FeatureStoreClient()
#no point in time information needed
fs.create_table(
    "omop531.person_features",
    primary_keys=["person_id"],
    df=person_df,
    description="Attributes related to a person identity",
)

#point in time relationship 
fs.create_table(
    "omop531.condition_features",
    primary_keys=["person_id"],
    df=person_medical_condition_df,
    timestamp_keys=["condition_era_start_date"],
    description="Attributes related to a person's medical history",
)

#point in time relationship 
fs.create_table(
    "omop531.drug_features",
    primary_keys=["person_id"],
    df=person_rx_df,
    timestamp_keys=["drug_exposure_start_date"],
    description="Attributes related to a person's drug history",
)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Training Dataset TODO

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Online Feature Stores (In Real Time Data)

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS omop531.person_features_online
# MAGIC AS 
# MAGIC SELECT person.person_id 
# MAGIC --person features
# MAGIC ,person.year_of_birth
# MAGIC ,person.race_concept_id
# MAGIC ,person.gender_source_value
# MAGIC --condition features
# MAGIC ,SUM(NVL(`4112343`, 0)) as `4112343`, SUM(NVL(`4289517`, 0)) as `4289517`
# MAGIC ,SUM(NVL(`432867`, 0)) as `432867`, SUM(NVL(`4237458`, 0)) as `4237458`, SUM(NVL(`260139`, 0)) as `260139`
# MAGIC ,SUM(NVL(`312437`, 0)) as `312437`, SUM(NVL(`254761`, 0)) as `254761`, SUM(NVL(`40481087`, 0)) as `40481087`
# MAGIC ,SUM(NVL(`80502`, 0)) as `80502`, SUM(NVL(`437663`, 0)) as `437663`
# MAGIC --drug features
# MAGIC ,SUM(NVL(`40163554`, 0)) as `40163554`, SUM(NVL(`40221901`, 0)) as `40221901`
# MAGIC FROM omop531.person_features_offline person
# MAGIC LEFT OUTER JOIN omop531.condition_features_offline cond
# MAGIC   on cond.person_id=person.person_id
# MAGIC LEFT OUTER JOIN omop531.drug_features_offline rx
# MAGIC   on rx.person_id=person.person_id
# MAGIC GROUP BY person.person_id
# MAGIC ,person.year_of_birth
# MAGIC ,person.race_concept_id
# MAGIC ,person.gender_source_value
# MAGIC   

# COMMAND ----------

display( spark.table("omop531.person_features_online") )

# COMMAND ----------

person_online_df = spark.table("omop531.person_features_online")

fs.create_table(
    "omop531.person_features_online",
    primary_keys=["person_id"],
    df=person_online_df,
    description="Person attribute features for online serving",
)

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW TABLES IN omop_patient_risk

# COMMAND ----------

FEATURE_TABLE_NAME = 'omop_patient_risk.drug_features'
description="drug features"


# COMMAND ----------

# DBTITLE 1,add drug features to feature store
# MAGIC %py
# MAGIC FEATURE_TABLE_NAME = 'omop_patient_risk.drug_features'
# MAGIC description="drug features"
# MAGIC try:
# MAGIC   fs.drop_table(FEATURE_TABLE_NAME)
# MAGIC except ValueError:
# MAGIC   pass
# MAGIC 
# MAGIC #TODO Add date parameter to "offline" feature store for model training 
# MAGIC drug_features_df = sql(f"""
# MAGIC     select subject_id, VALUE_AS_CONCEPT_ID as drug_concept_id, VALUE_AS_NUMBER as drug_quantity  
# MAGIC     from omop_patient_risk.cohort_attribute
# MAGIC     where 
# MAGIC     ATTRIBUTE_DEFINITION_ID = {drug_hist_att_id} and COHORT_DEFINITION_ID={target_cohort_id}
# MAGIC     """
# MAGIC     ).groupBy('subject_id').pivot('drug_concept_id').sum('drug_quantity').fillna(0)
# MAGIC 
# MAGIC fs.create_table(
# MAGIC     name=FEATURE_TABLE_NAME,
# MAGIC     primary_keys=["subject_id"],
# MAGIC     df=drug_features_df,
# MAGIC     schema=drug_features_df.schema,
# MAGIC     description=description
# MAGIC )

# COMMAND ----------

sql(f'SELECT * from {FEATURE_TABLE_NAME} limit 10').display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Commorbidity history features

# COMMAND ----------

# DBTITLE 1,top n commorbidities
# MAGIC %py
# MAGIC sql(f"""
# MAGIC select VALUE_AS_CONCEPT_ID as condition_concept_id, count(*) as cnt from omop_patient_risk.cohort_attribute where ATTRIBUTE_DEFINITION_ID={condition_hist_att_id} group by 1
# MAGIC order by 2 desc
# MAGIC limit {max_n_commorbidities}
# MAGIC """).createOrReplaceTempView('top_comorbidities')

# COMMAND ----------

# MAGIC %py
# MAGIC FEATURE_TABLE_NAME = 'omop_patient_risk.condition_history_features'
# MAGIC description="condition history features"
# MAGIC try:
# MAGIC   fs.drop_table(FEATURE_TABLE_NAME)
# MAGIC except ValueError:
# MAGIC   pass
# MAGIC   
# MAGIC condition_history_df = sql(f"""
# MAGIC     select subject_id, VALUE_AS_CONCEPT_ID as condition_concept_id, VALUE_AS_NUMBER as n_condition_occurance 
# MAGIC     from omop_patient_risk.cohort_attribute
# MAGIC     where 
# MAGIC     ATTRIBUTE_DEFINITION_ID = {condition_hist_att_id} and COHORT_DEFINITION_ID={target_cohort_id}
# MAGIC     and
# MAGIC     VALUE_AS_CONCEPT_ID in (select condition_concept_id from top_comorbidities)
# MAGIC     """
# MAGIC     ).groupBy('subject_id').pivot('condition_concept_id').sum('n_condition_occurance').fillna(0)
# MAGIC 
# MAGIC fs.create_table(
# MAGIC     name=FEATURE_TABLE_NAME,
# MAGIC     primary_keys=["subject_id"],
# MAGIC     df=condition_history_df,
# MAGIC     schema=condition_history_df.schema,
# MAGIC     description=description
# MAGIC )

# COMMAND ----------

sql(f'select * from {FEATURE_TABLE_NAME} limit 10').display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Demographics information 

# COMMAND ----------

# MAGIC %py
# MAGIC FEATURE_TABLE_NAME = "omop_patient_risk.subject_demographics_features"
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
# MAGIC     p.YEAR_OF_BIRTH,
# MAGIC     date_diff(c.cohort_start_date, p.BIRTH_DATETIME) as age_in_days,
# MAGIC     p.RACE_CONCEPT_ID,
# MAGIC     p.ETHNICITY_CONCEPT_ID
# MAGIC   from cohort c
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
# MAGIC # select * from subject_demographics limit 10

# COMMAND ----------

sql(f'select * from {FEATURE_TABLE_NAME} limit 10').display()

# COMMAND ----------

# MAGIC %md
# MAGIC ##  3. Training Dataset
# MAGIC Now that our cohorts are in place, we can create the final dataset. we then use Databricks AutoML to train a model for predicting risk and also understand features impacting patient risk. 
# MAGIC To make it simpler, first we create a function that decides whether two cohorts overlap:

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW TABLES FROM omop_patient_risk

# COMMAND ----------

# DBTITLE 1,add outcome labels
# MAGIC %py
# MAGIC outcomes_df = sql(f"""
# MAGIC     select subject_id, VALUE_AS_CONCEPT_ID as outcome_concept_id, VALUE_AS_NUMBER as visited_emergency 
# MAGIC     from omop_patient_risk.cohort_attribute
# MAGIC     where 
# MAGIC     ATTRIBUTE_DEFINITION_ID = {outcome_att_id} and COHORT_DEFINITION_ID={target_cohort_id}
# MAGIC     """
# MAGIC     ).groupBy('subject_id').pivot('outcome_concept_id').min('visited_emergency').fillna(0)

# COMMAND ----------

# DBTITLE 1,create training data
# MAGIC %py
# MAGIC training_df = (
# MAGIC   sql('select subject_id from omop_patient_risk.cohort')
# MAGIC   .filter(f'cohort_definition_id={target_cohort_id}')
# MAGIC   .join(outcomes_df, how='left',on='subject_id')
# MAGIC   .selectExpr('subject_id',f"CAST(`{outcome_concept_id}` AS INT) as outcome")
# MAGIC   .fillna(0)
# MAGIC   )

# COMMAND ----------

# DBTITLE 1,proportion of patients with the outcome
training_df.selectExpr('avg(outcome)').display()

# COMMAND ----------

# DBTITLE 1,add features to the training dataset
from databricks.feature_store import FeatureLookup


feature_lookups = [
    FeatureLookup(
      table_name = 'omop_patient_risk.subject_demographics_features',
      lookup_key = 'subject_id'
    ),
    FeatureLookup(
      table_name = 'omop_patient_risk.drug_features',
      lookup_key = 'subject_id'
    ),
    FeatureLookup(
      table_name = 'omop_patient_risk.condition_history_features',
      lookup_key = 'subject_id'
    )
  ]

training_set = fs.create_training_set(
  df=training_df,
  feature_lookups = feature_lookups,
  label = 'outcome',
 exclude_columns = ['subject_id']
)

training_df = training_set.load_df().fillna(0)

# COMMAND ----------

training_df.selectExpr('avg(outcome)').display()

# COMMAND ----------

training_df.write.saveAsTable('omop_patient_risk.training_data')

# COMMAND ----------

# MAGIC %md
# MAGIC ```
# MAGIC with drugs as(
# MAGIC SELECT
# MAGIC 
# MAGIC                 C.concept_id Drug_concept_id,
# MAGIC 
# MAGIC                 C.concept_name Drug_concept_name,
# MAGIC 
# MAGIC                 C.concept_code Drug_concept_code,
# MAGIC 
# MAGIC                 C.concept_class_id Drug_concept_class,
# MAGIC 
# MAGIC                 C.standard_concept Drug_concept_level,
# MAGIC 
# MAGIC                 C.vocabulary_id Drug_concept_vocab_id,
# MAGIC 
# MAGIC                 V.vocabulary_name Drug_concept_vocab_code,
# MAGIC 
# MAGIC                ( CASE C.vocabulary_id
# MAGIC 
# MAGIC                         WHEN 'RxNorm' THEN
# MAGIC 
# MAGIC                                 CASE lower(C.concept_class_id)
# MAGIC 
# MAGIC                                 WHEN 'clinical drug' THEN 'Yes'
# MAGIC 
# MAGIC                                 WHEN 'branded drug' THEN 'Yes'
# MAGIC 
# MAGIC                                 WHEN 'ingredient' THEN 'Yes'
# MAGIC 
# MAGIC                                 WHEN 'branded pack' THEN 'Yes'
# MAGIC 
# MAGIC                                 WHEN 'clinical pack' THEN 'Yes'
# MAGIC 
# MAGIC                                 ELSE 'No' END
# MAGIC 
# MAGIC                         ELSE 'No' END) Is_Drug_Concept_flag
# MAGIC 
# MAGIC                 -- (CASE C.domain_id WHEN 'Drug' THEN 'Yes' ELSE 'No' END) Is_Drug_Concept_flag
# MAGIC 
# MAGIC         FROM
# MAGIC 
# MAGIC                 concept C,
# MAGIC 
# MAGIC                 vocabulary V
# MAGIC 
# MAGIC         WHERE
# MAGIC 
# MAGIC                 C.vocabulary_id = V.vocabulary_id
# MAGIC                 )
# MAGIC select * from drugs
# MAGIC where drug_concept_id in (SELECT distinct(drug_concept_id) FROM drug_exposure)
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC Copyright / License info of the notebook. Copyright Databricks, Inc. [2021].  The source in this notebook is provided subject to the [Databricks License](https://databricks.com/db-license-source).  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC 
# MAGIC |Library Name|Library License|Library License URL|Library Source URL| 
# MAGIC | :-: | :-:| :-: | :-:|
# MAGIC |Smolder |Apache-2.0 License| https://github.com/databrickslabs/smolder | https://github.com/databrickslabs/smolder/blob/master/LICENSE|
# MAGIC |Synthea|Apache License 2.0|https://github.com/synthetichealth/synthea/blob/master/LICENSE| https://github.com/synthetichealth/synthea|
# MAGIC | OHDSI/CommonDataModel| Apache License 2.0 | https://github.com/OHDSI/CommonDataModel/blob/master/LICENSE | https://github.com/OHDSI/CommonDataModel |
# MAGIC | OHDSI/ETL-Synthea| Apache License 2.0 | https://github.com/OHDSI/ETL-Synthea/blob/master/LICENSE | https://github.com/OHDSI/ETL-Synthea |
# MAGIC |OHDSI/OMOP-Queries|||https://github.com/OHDSI/OMOP-Queries|
# MAGIC |The Book of OHDSI | Creative Commons Zero v1.0 Universal license.|https://ohdsi.github.io/TheBookOfOhdsi/index.html#license|https://ohdsi.github.io/TheBookOfOhdsi/|
