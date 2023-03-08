# Databricks notebook source
# MAGIC %md This solution accelerator notebook is also available at https://github.com/databricks-industry-solutions/hls-patient-risk

# COMMAND ----------

# MAGIC %md
# MAGIC ## Patient-Level Risk Scoring Based on Comorbidity History
# MAGIC Longitudinal health records, contain tremendous amount of information with regard to a patients risk factors. For example, using standrad ML techniques, one can investigate the correlation between a patient's health history and a given outcome such as heart attack. In this case, we use a patient’s condition history, durgs taken and demographics information as input, and  use a patinet's encounter's history to identify patients who have had a heart attack within a given window of time. 
# MAGIC 
# MAGIC In this solution accelerator, we assume that the patients data are already stored as in an OMOP 5.3 CDM on Delta Lake and use the OHDSI's patient-level risk prediction methodology to train a classifier that predicts the risk of a given outcome. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Experiment Design
# MAGIC In this section we outline the terminology used in this solution accelerator, based on on the experiment design outlined in the [Book of OHDSI](https://ohdsi.github.io/TheBookOfOhdsi/PatientLevelPrediction.html#designing-a-patient-level-prediction-study)
# MAGIC 
# MAGIC <img src='https://ohdsi.github.io/TheBookOfOhdsi/images/PatientLevelPrediction/Figure1.png'>
# MAGIC 
# MAGIC |||
# MAGIC |-----|-----|
# MAGIC |Choice|Description|
# MAGIC |Target cohort|How do we define the cohort of persons for whom we wish to predict?|
# MAGIC |Outcome cohort|	How do we define the outcome we want to predict?|
# MAGIC |Time-at-risk|In which time window relative to t=0 do we want to make the prediction?|
# MAGIC |Model	|What algorithms do we want to use, and which potential predictor variables do we include?|

# COMMAND ----------

# MAGIC %md
# MAGIC #### Washout Period: days (int):
# MAGIC 
# MAGIC > The minimum amount of observation time required before the start of the target cohort. This choice could depend on the available patient time in the training data, but also on the time we expect to be available in the data sources we want to apply the model on in the future. The longer the minimum observation time, the more baseline history time is available for each person to use for feature extraction, but the fewer patients will qualify for analysis. Moreover, there could be clinical reasons to choose a short or longer look-back period. 
# MAGIC 
# MAGIC For our example, we will use a _365-day prior history as look-back period (washout period)_

# COMMAND ----------

# MAGIC %md
# MAGIC #### Allowed in cohort multiple times? (boolean):
# MAGIC 
# MAGIC >Can patients enter the target cohort multiple times? In the target cohort definition, a person may qualify for the cohort multiple times during different spans of time, for example if they had different episodes of a disease or separate periods of exposure to a medical product. The cohort definition does not necessarily apply a restriction to only let the patients enter once, but in the context of a particular patient-level prediction problem we may want to restrict the cohort to the first qualifying episode. 
# MAGIC 
# MAGIC In our example, _a person can only enter the target cohort once_, i.e. patients who have been diagnosed with CHF most recently. 

# COMMAND ----------

# MAGIC %md
# MAGIC #### Inlcude if they have experienced the outcome before? (boolean):
# MAGIC 
# MAGIC >Do we allow persons to enter the cohort if they experienced the outcome before? Do we allow persons to enter the target cohort if they experienced the outcome before qualifying for the target cohort? Depending on the particular patient-level prediction problem, there may be a desire to predict incident first occurrence of an outcome, in which case patients who have previously experienced the outcome are not at risk for having a first occurrence and therefore should be excluded from the target cohort. In other circumstances, there may be a desire to predict prevalent episodes, whereby patients with prior outcomes can be included in the analysis and the prior outcome itself can be a predictor of future outcomes. 
# MAGIC 
# MAGIC For our prediction example, we allow all patients who have expereinced the outcome - emergency room visits - to be allowed in the target cohort. 

# COMMAND ----------

# MAGIC %md
# MAGIC #### time at risk period (start,end), start should be greater or equal than the target cohort start date**:
# MAGIC > How do we define the period in which we will predict our outcome relative to the target cohort start? We have to make two decisions to answer this question. First, does the time-at-risk window start at the date of the start of the target cohort or later? Arguments to make it start later could be that we want to avoid outcomes that were entered late in the record that actually occurred before the start of the target cohort or we want to leave a gap where interventions to prevent the outcome could theoretically be implemented. Second, we need to define the time-at-risk by setting the risk window end, as some specification of days offset relative to the target cohort start or end dates.
# MAGIC 
# MAGIC For our problem we will predict in a time-at-risk window starting 7 days after the start of the target cohort (`min_time_at_risk=7`) up to 365 days later (`max_time_at_risk = 365`)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Parameter descriptions
# MAGIC |parameter name|description|default value|
# MAGIC |-----|-----|-----|
# MAGIC |`target_condition_concept_id`|qualifying condition to enter the target cohort| 4229440 (CHF)|
# MAGIC |`outcome_concept_id`|outcome to predict| 9203 (Emergency Room Visit)|
# MAGIC |`drug1_concept_id`|concept id for drug exposure history | 40163554 (Warfarin)|
# MAGIC |`drug2_concept_id`|concept id for drug exposure history | 40221901 (Acetaminophen)|
# MAGIC |`cond_history_years`| years of patient history to look up| 5 |
# MAGIC |`max_n_commorbidities`| max number of commorbid conditions to use for the prediction probelm| 5 |
# MAGIC |`min_observation_period`| whashout period in days| 1095 |
# MAGIC |`min_time_at_risk`| days since target cohort start to start the time at risk| 7 |
# MAGIC |`max_time_at_risk`| days since target cohort start to end time at risk window| 365 |

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Preparation
# MAGIC TODO:
# MAGIC 1. Using OMOP we create target cohorts
# MAGIC 2. add attributes to attributes table
# MAGIC 3. use attributes to create feature store

# COMMAND ----------

# MAGIC %md
# MAGIC ### ML 
# MAGIC TODO: add verbiage arround AutoML
# MAGIC <img src='https://hls-eng-data-public.s3.amazonaws.com/img/patient_risk_automl.gif'>

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Flow 
# MAGIC TODO
