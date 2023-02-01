## Patient-Level Risk Scoring Based on Comorbidity History

Longitudinal health records, contain tremendous amount of information with regard to a patients risk factors. For example, using standrad ML techniques, one can investigate the correlation between a patient's health history and a given outcome such as heart attack. In this case, we use a patient’s condition history, durgs taken and demographics information as input, and  use a patinet's encounter's history to identify patients who have had a heart attack within a given window of time. 
In this solution accelerator, we assume that the patients data are already stored as in an OMOP 5.3 CDM on Delta Lake and use the OHDSI's patient-level risk prediction methodology to train a classifier that predicts the risk of a given outcome. 


### Experiment Design

In this section we outline the terminology used in this solution accelerator, based on on the experiment design outlined in the [Book of OHDSI](https://ohdsi.github.io/TheBookOfOhdsi/PatientLevelPrediction.html#designing-a-patient-level-prediction-study)
<img src='https://ohdsi.github.io/TheBookOfOhdsi/images/PatientLevelPrediction/Figure1.png'>
|||
|-----|-----|
|Choice|Description|
|Target cohort|How do we define the cohort of persons for whom we wish to predict?|
|Outcome cohort|	How do we define the outcome we want to predict?|
|Time-at-risk|In which time window relative to t=0 do we want to make the prediction?|
|Model	|What algorithms do we want to use, and which potential predictor variables do we include?|


#### Washout Period: days (int):

> The minimum amount of observation time required before the start of the target cohort. This choice could depend on the available patient time in the training data, but also on the time we expect to be available in the data sources we want to apply the model on in the future. The longer the minimum observation time, the more baseline history time is available for each person to use for feature extraction, but the fewer patients will qualify for analysis. Moreover, there could be clinical reasons to choose a short or longer look-back period. 
For our example, we will use a _365-day prior history as look-back period (washout period)_

#### Allowed in cohort multiple times? (boolean):

>Can patients enter the target cohort multiple times? In the target cohort definition, a person may qualify for the cohort multiple times during different spans of time, for example if they had different episodes of a disease or separate periods of exposure to a medical product. The cohort definition does not necessarily apply a restriction to only let the patients enter once, but in the context of a particular patient-level prediction problem we may want to restrict the cohort to the first qualifying episode. 
In our example, _a person can only enter the target cohort once_, i.e. patients who have been diagnosed with CHF most recently. 

#### Inlcude if they have experienced the outcome before? (boolean):

>Do we allow persons to enter the cohort if they experienced the outcome before? Do we allow persons to enter the target cohort if they experienced the outcome before qualifying for the target cohort? Depending on the particular patient-level prediction problem, there may be a desire to predict incident first occurrence of an outcome, in which case patients who have previously experienced the outcome are not at risk for having a first occurrence and therefore should be excluded from the target cohort. In other circumstances, there may be a desire to predict prevalent episodes, whereby patients with prior outcomes can be included in the analysis and the prior outcome itself can be a predictor of future outcomes. 
For our prediction example, we allow all patients who have expereinced the outcome - emergency room visits - to be allowed in the target cohort. 

#### time at risk period (start,end), start should be greater or equal than the target cohort start date**:

> How do we define the period in which we will predict our outcome relative to the target cohort start? We have to make two decisions to answer this question. First, does the time-at-risk window start at the date of the start of the target cohort or later? Arguments to make it start later could be that we want to avoid outcomes that were entered late in the record that actually occurred before the start of the target cohort or we want to leave a gap where interventions to prevent the outcome could theoretically be implemented. Second, we need to define the time-at-risk by setting the risk window end, as some specification of days offset relative to the target cohort start or end dates.
For our problem we will predict in a time-at-risk window starting 7 days after the start of the target cohort (`min_time_at_risk=7`) up to 365 days later (`max_time_at_risk = 365`)

#### Parameter descriptions

|parameter name|description|default value|
|-----|-----|-----|
|`target_condition_concept_id`|qualifying condition to enter the target cohort| 4229440 (CHF)|
|`outcome_concept_id`|outcome to predict| 9203 (Emergency Room Visit)|
|`drug1_concept_id`|concept id for drug exposure history | 40163554 (Warfarin)|
|`drug2_concept_id`|concept id for drug exposure history | 40221901 (Acetaminophen)|
|`cond_history_years`| years of patient history to look up| 5 |
|`max_n_commorbidities`| max number of commorbid conditions to use for the prediction probelm| 5 |
|`min_observation_period`| whashout period in days| 1095 |
|`min_time_at_risk`| days since target cohort start to start the time at risk| 7 |
|`max_time_at_risk`| days since target cohort start to end time at risk window| 365 |

### Data Preparation

TODO:
1. Using OMOP we create target cohorts
2. add attributes to attributes table
3. use attributes to create feature store

### ML 

TODO: add verbiage arround AutoML
<img src='https://hls-eng-data-public.s3.amazonaws.com/img/patient_risk_automl.gif'>

### Data Flow 

TODO
