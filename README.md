# Student-Performance-Analysis
Analysis of student performance data using Python

Note: Aashish sent me his part of the code and also the parts he reviewed and I was the one to put it on Github.
Also, some of the code (heatmap and machine learning part) comes from our Machine Learning Lab solution pdf files. So all of this must be credited to our Machine Learning teacher.

## Context

A secondary school wants to investigate social, demographic, and school related causes linked to student failure in order to proactively identify students at risk and provide them with adequate counseling and support.

## Features

The dataset contains the following features.

sex: The student's gender.

age: The student's age (in years).

famsize: Family size (LE3: less or equal to 3; GT3: greater than 3).

Pstatus: Parents' cohabitation status (T: living together; A: living apart).

Mjob: Mother's job.

Fjob: Father's job.

guardian: The student's guardian.

studytime: Weekly study time:

1: <2 hours;

2: 2 to 5 hours;

3: 5 to 10 hours;

4: > 10 hours.

schoolsup: Whether the student has extra educational support or not.

famsup: Whether the student has family educational support or not.

paid: Whether the student attends extra paid classes or not.

activities: Whether the student has extra-curricular activities or not.

internet: Whether the student has internet access at home or not.

romantic: Whether the student is in a romantic relationship or not.

famrel: Quality of family relationships, from 1 (very bad) to 5 (excellent).

goout: Frequency of going out with friends, from 1 (very low) to 5 (very high).

Dalc: Workday alcohol consumption, from 1 (very low) to 5 (very high).

Walc: Weekend alcohol consumption, from 1 (very low) to 5 (very high).

health: Current health condition, from 1 (very bad) to 5 (very good).

absences: Number of school absences.

success (target): Whether the student passed or failed.


## Questions

What is the proportion of students that failed the class?

How are different variables (age, gender, internet access, family composition, etc.) distributed in the dataset?

How do variables interact with each other?

- Are age, absences, family composition, etc. distributed similarly for both genders?

- What are the age, gender, internet access, alcohol consumption, etc. distributions by parents cohabitation status? etc.

What are the principal factors that play a key role in student failure?

(Optional) Build a simple machine learning classification model that predicts if a student will fail the class.

