\# Random Forest CDP Load Profile Imputation



Machine learning pipeline for imputing missing half-hourly load profile (LP)

data from AMI/CDP energy meters using Random Forest Regression.



\## Project Status



\*\*Current version:\*\* V1 Baseline



\*\*Status:\*\* Working baseline completed



The current implementation uses:



\- One selected CDP meter

\- One unidirectional energy direction

\- 30-minute load profile data

\- Fixed-event gap experiments

\- One Random Forest prediction per missing LP

\- Gap lengths of 1, 6, 24 and 48 LP

\- 96 LP gap experiment removed

\- Chronological train/validation/test split

\- Deterministic calendar and cyclic features

\- 96 LP observed context on both sides of each gap



\---



\## Problem



AMI/CDP meters can contain missing half-hourly load profile observations.



For a missing interval, the objective is to estimate the missing energy value using

information that would legitimately be available around the missing event.



The project evaluates Random Forest Regression for this imputation problem.



\---



\## Dataset



The source dataset contains:



\- 17,520 half-hourly observations

\- 50 CDP meters

\- 101 columns

\- One timestamp column

\- A+ and A- measurements for each CDP



The current selected meter is:



```text

CDP\_00526\_P\_01

