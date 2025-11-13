#=====================================================================================================================#
#                              Student performance analysis project                                                   #
#=====================================================================================================================#

# For github:
""" pathname: cd '/Users/yesminehachana/Desktop/Classes/Dauphine/2nd Year/1st Semester/Python/Student Performance Analysis Project'"""

#================================================================================#
# Initialisation
#================================================================================#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import csv

#================================================================================#
# Getting familiar with the dataset
#================================================================================#

students_raw = pd.read_csv(
    "/Users/yesminehachana/Desktop/Classes/Dauphine/2nd Year/1st Semester/Python/Student Performance Analysis Project/students.csv",  sep=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)

print(students_raw.head())

print(students_raw.shape)
print(students_raw.dtypes)

students = students_raw.rename(columns={
    "Mjob": "Mother_Job",
    "Fjob": "Father_Job",
    "Pstatus": "Parent_Status",
    "famsize": "Family_Size",
    "studytime": "Study_Time",
    "schoolsup": "School_Support",
    "famsup": "Family_Support",
    "paid" : "Paid_classes",
    "famrel" : "Relationship_with_family",
    "goout": "Go_Out",
    "Dalc": "Daily_Alcohol",
    "Walc": "Weekend_Alcohol"
})

students["sex"] = students["sex"].replace({
    "M": "Male",
    "F": "Female"
})
students["Parent_Status"] = students["Parent_Status"].replace({
    "T": "living together",
    "A": "living apart"
})

students["Family_Size"] = students["Family_Size"].replace({
    "LE3" : "less or equal to 3",
    "GT3": "greater than 3"
})

students.to_csv(
    "/Users/yesminehachana/Desktop/Classes/Dauphine/2nd Year/1st Semester/Python/Student Performance Analysis Project/students.csv",
    index=False
)

#================================================================================#
# Question 1: What is the proportion of students that failed the class?
#================================================================================#

fails = (students["success"] == False).sum()
total = len(students)
print(f"Failure rate: {fails / total * 100:.2f}%")

#================================================================================#
# QUESTION 2: Variable Distributions
#================================================================================#
# How are different variables (age, gender, internet access, family composition,
# etc.) distributed in the dataset?
#================================================================================#

#--------------------------------------------------------------------------------
# PART 2.1: Summary Statistics Tables
#--------------------------------------------------------------------------------

# Define variable categories
categorical_cols = ["sex", "Family_Size", "Parent_Status", "Mother_Job",
                    "Father_Job", "guardian", "School_Support", "Family_Support",
                    "Paid_classes", "activities", "internet", "romantic", "success"]

numeric_cols = ["age", "Study_Time", "Relationship_with_family", "Go_Out",
                "Daily_Alcohol", "Weekend_Alcohol", "health", "absences"]

# Numeric variables summary (mean, median, min, max)

# Select only the numerical columns
numeric_data = students[numeric_cols]

# Compute summary statistics separately
mean_values = numeric_data.mean()
median_values = numeric_data.median()
min_values = numeric_data.min()
max_values = numeric_data.max()

# Combine everything into a single DataFrame
numeric_summary = pd.DataFrame({
    'mean': mean_values,
    'median': median_values,
    'min': min_values,
    'max': max_values
})

# Round
numeric_summary = numeric_summary.round(2)

numeric_summary.to_latex("numeric_summary.tex", index=True,
                         caption="Descriptive Statistics of Numeric Variables",
                         label="tab:numeric_summary")
print("Numeric Variables Summary:")
print(numeric_summary)
print()

# Categorical variables summary (unique values, most frequent, frequency %)

# Create an empty dictionary to store results
cat_summary = {
    'Unique Values': [],
    'Most Frequent': [],
    'Frequency (%) of Most Frequent': []
}

# Loop through each categorical column
for col in categorical_cols:
    column_data = students[col] #for each categorical column (like sex), we get all values of that category

    # Count how many unique values
    unique_vals = column_data.nunique() # we count unique values for each category

    # Find the most frequent value
    most_frequent = column_data.mode()[0]   # mode() returns a list of values from most to least frequent, we take the first

    # Compute the frequency of the most frequent value by category in percentage
    freq_percent = column_data.value_counts(normalize=True).iloc[0] * 100 #iloc selects the value at position 0
    freq_percent = round(freq_percent, 2) # we used normalize=True above to get proportions

    # Store results
    cat_summary['Unique Values'].append(unique_vals)
    cat_summary['Most Frequent'].append(most_frequent)
    cat_summary['Frequency (%) of Most Frequent'].append(freq_percent)

# Convert to a DataFrame
categorical_summary = pd.DataFrame(cat_summary, index=categorical_cols)

print("Categorical Variables Summary:")
print(categorical_summary)
print()

#--------------------------------------------------------------------------------
# PART 2.2: Demographic & Background Variables
#--------------------------------------------------------------------------------

# Age distribution
plt.figure(figsize=(6, 4))
ax = sns.histplot(data=students, x="age", bins=range(15, 23),
                  color="skyblue", edgecolor="black")
plt.title("Distribution of Student Age (15–22)", fontsize=13)
plt.xlabel("Age")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("age_distribution.pdf", format="pdf", bbox_inches="tight")
plt.show()

# Gender distribution
plt.figure(figsize=(4, 4))
sns.countplot(data=students, x="sex", palette=["#8ecae6", "#f4a261"])
plt.title("Distribution of Students by Gender")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("gender_distribution.pdf", format="pdf", bbox_inches="tight")
plt.show()

# Parent occupation comparison

# List of job categories we want to appear in the plots
job_categories = ["at_home", "health", "other", "services", "teacher"]

# Mother job counts
mother_raw_counts = students["Mother_Job"].value_counts() # how many times each job appears for mothers
# We create a list of counts for each job category, in the order of job_categories, 0 if one of the label is missing
mother_counts = mother_raw_counts.reindex(job_categories, fill_value=0)

# Father job counts
father_raw_counts = students["Father_Job"].value_counts()
father_counts = father_raw_counts.reindex(job_categories, fill_value=0)

# Positions for the bars in the bar plot
x = np.arange(len(job_categories))  # This creates an array with 5 positions for the five job categories
width = 0.35  # width of the bars

plt.figure(figsize=(8, 5))
plt.bar(x - width/2, mother_counts, width, color="#8ecae6", label="Mother's Job") # we move the first bar a litte to the left (to avoid overlapping)
plt.bar(x + width/2, father_counts, width, color="#f4a261", label="Father's Job") # we move the second bar a little to the right

display_labels = ["At Home", "Health", "Other", "Services", "Teacher"]
plt.xticks(x, display_labels, rotation=15)
plt.xlabel("Job Category")
plt.ylabel("Count of Parents")
plt.title("Comparison of Mother's and Father's Occupations", fontsize=14)
plt.legend(title="Parent Job")
plt.tight_layout()
plt.savefig("parent_jobs_comparison.pdf", format="pdf", bbox_inches="tight")
plt.show()

#--------------------------------------------------------------------------------
# PART 2.3: Academic Engagement Variables
#--------------------------------------------------------------------------------

# Study time distribution
plt.figure(figsize=(5, 4))
ax = sns.histplot(data=students, x="Study_Time", bins=range(1, 6),
                  color="skyblue", edgecolor="black")
ax.set_xticks([1.5, 2.5, 3.5, 4.5])
ax.set_xticklabels(["<2 hours", "2 to 5 hours", "5 to 10 hours", "> 10 hours"])
plt.title("Weekly Study Time Distribution", fontsize=13)
plt.xlabel("Study Time")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("study_time_distribution.pdf", format="pdf", bbox_inches="tight")
plt.show()

# School support
support_counts = students["School_Support"].value_counts()
colors = ["#8ecae6" if label == "yes" else "#f4a261" for label in support_counts.index]

plt.figure(figsize=(5, 5))
plt.pie(support_counts, labels=support_counts.index, autopct='%1.1f%%',
        colors=colors, startangle=90, explode=(0.05, 0),
        wedgeprops={"edgecolor": "white"})
plt.title("Proportion of Students Receiving School Support", fontsize=13)
plt.tight_layout()
plt.savefig("school_support_distribution.pdf", format="pdf", bbox_inches="tight")
plt.show()

# Family support
support_counts = students["Family_Support"].value_counts()
colors = ["#8ecae6" if label == "yes" else "#f4a261" for label in support_counts.index]

plt.figure(figsize=(5, 5))
plt.pie(support_counts, labels=support_counts.index, autopct='%1.1f%%',
        colors=colors, startangle=90, explode=(0.05, 0),
        wedgeprops={"edgecolor": "white"})
plt.title("Proportion of Students Receiving Family Support", fontsize=13)
plt.tight_layout()
plt.savefig("family_support_distribution.pdf", format="pdf", bbox_inches="tight")
plt.show()

# Paid classes
support_counts = students["Paid_classes"].value_counts()
colors = ["#8ecae6" if label == "yes" else "#f4a261" for label in support_counts.index]

plt.figure(figsize=(5, 5))
plt.pie(support_counts, labels=support_counts.index, autopct='%1.1f%%',
        colors=colors, startangle=90, explode=(0.05, 0),
        wedgeprops={"edgecolor": "white"})
plt.title("Proportion of Students Receiving Paid Tuition", fontsize=13)
plt.tight_layout()
plt.savefig("paid_classes_distribution.pdf", format="pdf", bbox_inches="tight")
plt.show()

# Internet access
support_counts = students["internet"].value_counts()
colors = ["#8ecae6" if label == "yes" else "#f4a261" for label in support_counts.index]

plt.figure(figsize=(5, 5))
plt.pie(support_counts, labels=support_counts.index, autopct='%1.1f%%',
        colors=colors, startangle=90, explode=(0.05, 0),
        wedgeprops={"edgecolor": "white"})
plt.title("Proportion of Students with Access to Internet", fontsize=13)
plt.tight_layout()
plt.savefig("internet_access_distribution.pdf", format="pdf", bbox_inches="tight")
plt.show()

#--------------------------------------------------------------------------------
# PART 2.4: Social & Lifestyle Variables
#--------------------------------------------------------------------------------

# Going out frequency vs health status comparison
out_counts = students["Go_Out"].value_counts().sort_index() # we sort the values by index (1 to 5), not by frequency
health_counts = students["health"].value_counts().sort_index()

x = np.arange(1, 6)  # produces an x axis scale from 1 to 5, for each level of the variable
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x - width/2, out_counts, width, color="#8ecae6", label="Going Out Frequency")
plt.bar(x + width/2, health_counts, width, color="#f4a261", label="Health Status")

plt.xticks(x, ["Very Low", "Low", "Moderate", "High", "Very High"])
plt.xlabel("Rating Level (1-5)")
plt.ylabel("Count of Students")
plt.title("Distribution of Going Out Frequency vs Health Status", fontsize=14)
plt.legend(title="Variable")
plt.tight_layout()
plt.savefig("goout_vs_health_distribution.pdf", format="pdf", bbox_inches="tight")
plt.show()

# Alcohol consumption: daily vs weekend
daily_counts = students["Daily_Alcohol"].value_counts().sort_index()
weekend_counts = students["Weekend_Alcohol"].value_counts().sort_index()

x = np.arange(1, 6)  # 1 to 5 levels
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x - width/2, daily_counts, width, color="#8ecae6", label="Daily Alcohol")
plt.bar(x + width/2, weekend_counts, width, color="#f4a261", label="Weekend Alcohol")

plt.xticks(x, ["Very Low", "Low", "Moderate", "High", "Very High"])
plt.xlabel("Alcohol Consumption Level")
plt.ylabel("Count of Students")
plt.title("Daily vs Weekend Alcohol Consumption", fontsize=14)
plt.legend(title="Type of Day")
plt.tight_layout()
plt.savefig("alcohol_consumption_distribution.pdf", format="pdf", bbox_inches="tight")
plt.show()

#================================================================================#
# QUESTION 3: Variable Interactions
#================================================================================#
# How do variables interact with each other?
# - Are variables distributed similarly for both genders?
# - How do variables differ by parent cohabitation status?
# - What other interesting interactions exist?
#================================================================================#

#--------------------------------------------------------------------------------
# PART 3.1: Distributions by Gender
#--------------------------------------------------------------------------------

# Variables to analyze
vars_to_summarize = [
    "age", "absences", "Go_Out", "Weekend_Alcohol", "Daily_Alcohol",
    "Relationship_with_family", "Study_Time",
    "School_Support", "Family_Support", "Paid_classes",
    "activities", "internet", "romantic", "success"
]

numeric_vars = [
    "age", "absences", "Go_Out", "Weekend_Alcohol", "Daily_Alcohol",
    "Relationship_with_family", "Study_Time"
]

categorical_vars = [
    "School_Support", "Family_Support", "Paid_classes",
    "activities", "internet", "romantic", "success"
]

# Create an empty DataFrame to store the results
summary_by_gender = pd.DataFrame(index=vars_to_summarize) # we create a data frame with each var to summarize as a row

# Separate the data into two groups
male_students = students[students["sex"] == "Male"] # we take the rows of students where sex = male
female_students = students[students["sex"] == "Female"]

# Compute means for numeric variables
for var in numeric_vars:
    # Mean for males
    male_mean = male_students[var].mean()

    # Mean for females
    female_mean = female_students[var].mean()

    # Store the results in the summary table
    summary_by_gender.loc[var, "Male"] = male_mean # panda create a column "Male" and assignes male_mean to each var (row)
    summary_by_gender.loc[var, "Female"] = female_mean # loc says go to this specific row and this specific column. It creates the Male of Female column if it does not exist.

# Calculate percentage of "yes" or True for categorical variables by gender
for var in categorical_vars:
    # Extract the column for males
    male_values = male_students[var]

    # Count "yes" or True for males
    male_yes = male_values.isin(["yes", True]).mean() * 100

    # Extract the column for females
    female_values = female_students[var]

    # Count "yes" or True for females
    female_yes = female_values.isin(["yes", True]).mean() * 100

    # Store the results in the summary table
    summary_by_gender.loc[var, "Male"] = male_yes
    summary_by_gender.loc[var, "Female"] = female_yes

summary_by_gender = summary_by_gender.round(2)

print("Summary Table - Variables by Gender:")
print(summary_by_gender)
print()

# Family size distribution by gender
print("Family Size Distribution by Gender (%):")
print(pd.crosstab(students['sex'], students['Family_Size'], normalize='index') * 100)
print()
# We create a table that counts how many times combinations of sex and family size appear (crosstab)
# We then create percentages by dividing each value by the row's total

# Visualize age by gender
plt.figure(figsize=(6, 5))
age_by_gender = students.groupby('sex')['age'].mean() # We group the dataset by gender, take the age column for each group, and compute the average age for males and females separately
plt.bar(range(len(age_by_gender)), age_by_gender.values,
        color=['#8ecae6', '#f4a261'], edgecolor='black')
plt.xticks(range(len(age_by_gender)), age_by_gender.index)
plt.ylabel('Average Age')
plt.xlabel('Gender')
plt.title('Average Age by Gender')
plt.tight_layout()
plt.savefig("age_by_gender.pdf", format="pdf", bbox_inches="tight")
plt.show()

# Visualize absences by gender
plt.figure(figsize=(6, 5))
absences_by_gender = students.groupby('sex')['absences'].mean()
plt.bar(range(len(absences_by_gender)), absences_by_gender.values,
        color=['#8ecae6', '#f4a261'], edgecolor='black')
plt.xticks(range(len(absences_by_gender)), absences_by_gender.index)
plt.ylabel('Average Absences')
plt.xlabel('Gender')
plt.title('Average Absences by Gender')
plt.tight_layout()
plt.savefig("absences_by_gender.pdf", format="pdf", bbox_inches="tight")
plt.show()

#--------------------------------------------------------------------------------
# PART 3.2: Distributions by Parent Cohabitation Status
#--------------------------------------------------------------------------------

# Variables to analyze
vars_to_summarize = [
    "guardian", "Study_Time", "health",
    "Family_Support", "Daily_Alcohol", "Weekend_Alcohol",
    "Relationship_with_family", "romantic", "absences", "success"
]

numeric_vars = [
    "Study_Time", "health", "Daily_Alcohol",
    "Weekend_Alcohol", "Relationship_with_family", "absences"
]

proportion_vars = ["Family_Support", "romantic", "success"]

# We create empty summary table (each row is one variable to summarize)
summary_by_parent_status = pd.DataFrame(index=vars_to_summarize)

# Split the data into two groups

living_together = students[students["Parent_Status"] == "living together"]
living_apart    = students[students["Parent_Status"] == "living apart"]

# Means for NUMERIC variables

for var in numeric_vars:
    # Mean for students whose parents live together
    mean_together = living_together[var].mean()

    # Mean for students whose parents live apart
    mean_apart = living_apart[var].mean()

    # Store in the summary table
    summary_by_parent_status.loc[var, "Living Together"] = mean_together
    summary_by_parent_status.loc[var, "Living Apart"] = mean_apart

# Percent "yes" / True for CATEGORICAL variables

for var in proportion_vars:
    # Column values for each group
    values_together = living_together[var]
    values_apart    = living_apart[var]

    # Percentage of "yes" or True in each group
    percent_together = values_together.isin(["yes", True]).mean() * 100
    percent_apart    = values_apart.isin(["yes", True]).mean() * 100

    # Store in the summary table
    summary_by_parent_status.loc[var, "Living Together"] = percent_together
    summary_by_parent_status.loc[var, "Living Apart"] = percent_apart

# Most frequent 'guardian' for each group

guardian_together = living_together["guardian"].mode()[0]
guardian_apart    = living_apart["guardian"].mode()[0]

summary_by_parent_status.loc["guardian", "Living Together"] = guardian_together
summary_by_parent_status.loc["guardian", "Living Apart"] = guardian_apart

# Round and print

summary_by_parent_status = summary_by_parent_status.round(2)

print("Summary Table - Variables by Parent Cohabitation Status:")
print(summary_by_parent_status)
print()

# Gender distribution by parent cohabitation status
print("Gender Distribution by Parent Cohabitation Status (%):")
gender_by_parent = pd.crosstab(students['Parent_Status'], students['sex'], normalize='index') * 100
print(gender_by_parent.round(2))
print()

# Visualize age by parent status
plt.figure(figsize=(6, 5))
age_by_parent = students.groupby('Parent_Status')['age'].mean()
plt.bar(range(len(age_by_parent)), age_by_parent.values,
        color=['#f4a261', '#8ecae6'], edgecolor='black')
plt.xticks(range(len(age_by_parent)), age_by_parent.index, rotation=15)
plt.ylabel('Average Age')
plt.xlabel('Parent Cohabitation Status')
plt.title('Average Age by Parent Cohabitation Status')
plt.tight_layout()
plt.savefig("age_by_parent_status.pdf", format="pdf", bbox_inches="tight")
plt.show()

#--------------------------------------------------------------------------------
# PART 3.3: Parental Job Type vs Student Success
#--------------------------------------------------------------------------------

# Compute success rates by parental job type

# Get list of job categories for mothers
mother_jobs = students["Mother_Job"].unique()

# Create an empty list to store results
mother_results = []

# Loop through each job type
for job in mother_jobs:
    # Select students whose mother has this job
    subset = students[students["Mother_Job"] == job] # for each job, it says if mother job is this job or not (true, false), then it only keeps the specific job (true). It does that for each job (loop).
    # We create a new DataFrame called subset that contains only the students whose mother has job type = job.

    # Compute the % of success ("yes") in this subset (for this specific mother's job)
    success_rate = subset["success"].isin(["yes", True]).mean() * 100

    # Add the result to the list
    mother_results.append([job, success_rate])

# Convert the results into a DataFrame
mother_success = pd.DataFrame(mother_results, columns=["Mother_Job", "Success_Rate"])

# Get list of job categories for fathers
father_jobs = students["Father_Job"].unique()

# Create empty list
father_results = []

# Loop through each job type
for job in father_jobs:
    subset = students[students["Father_Job"] == job]

    success_rate = subset["success"].isin(["yes", True]).mean() * 100

    father_results.append([job, success_rate])

# Convert to DataFrame
father_success = pd.DataFrame(father_results, columns=["Father_Job", "Success_Rate"])

# Visualize parental job type vs success rate
plt.figure(figsize=(8, 5))
x = range(len(mother_success))
width = 0.35


plt.bar([i - width/2 for i in x], mother_success["Success_Rate"], # we take each value i (category position) in x and shift it a bit to the left for mothers
        width=width, label="Mother's Job", color="#8ecae6")
plt.bar([i + width/2 for i in x], father_success["Success_Rate"], # we take each value i (category position) in x and shift it a bit to the right for fathers
        width=width, label="Father's Job", color="#f4a261")

plt.xticks(ticks=x, labels=mother_success["Mother_Job"], rotation=30, ha="right")
plt.xlabel("Parental Job Type")
plt.ylabel("Success Rate (%)")
plt.title("Student Success Rate by Parental Job Type")
plt.legend(title="Parent")
plt.tight_layout()
plt.savefig("success_by_parent_job.pdf", format="pdf", bbox_inches="tight")
plt.show()

#--------------------------------------------------------------------------------
# PART 3.4: Study Time vs Multiple Variables
#--------------------------------------------------------------------------------

# Variables to analyze
vars_to_summarize = [
    "health", "Daily_Alcohol", "Weekend_Alcohol",
    "absences", "Relationship_with_family", "Go_Out", "success"
]

numeric_vars = [
    "health", "Daily_Alcohol", "Weekend_Alcohol",
    "absences", "Relationship_with_family", "Go_Out"
]

# Compute mean for numeric variables by Study_Time
numeric_summary = students.groupby("Study_Time")[numeric_vars].mean().round(2)

# Compute proportion (%) of success by Study_Time
success_summary = (
    students.groupby("Study_Time")["success"]
    .apply(lambda x: x.isin(["yes", True]).mean() * 100)
    .round(2)
)

# Combine results
count_summary = students.groupby("Study_Time")["success"].count()
summary_by_studytime = numeric_summary.copy()
summary_by_studytime["success (%)"] = success_summary
summary_by_studytime["count"] = count_summary

# Label study time categories
summary_by_studytime.index = ["<2 hours", "2–5 hours", "5–10 hours", ">10 hours"]

# Transpose for better readability
summary_by_studytime = summary_by_studytime.T

print("Summary Table - Variables by Study Time:")
print(summary_by_studytime)
print()

#--------------------------------------------------------------------------------
# PART 3.5: Correlation Analysis
#--------------------------------------------------------------------------------

# Correlation heatmap (numeric variables only)
corr = students.select_dtypes(include=['number']).corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

plt.figure(figsize=(8, 6))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            square=True, cbar_kws={"shrink": 0.8})
plt.title("Correlation Heatmap - Numeric Variables (Lower Triangle)", fontsize=13)
plt.tight_layout()
plt.savefig("correlation_heatmap_numeric.pdf", format="pdf", bbox_inches="tight")
plt.show()

# Full correlation heatmap (including encoded binary variables)
students_corr = students.copy()
binary_map = {
    "yes": 1, "no": 0, True: 1, False: 0,
    "Male": 1, "Female": 0,
    "living together": 1, "living apart": 0
}
binary_vars = ["School_Support", "Family_Support", "Paid_classes",
               "activities", "internet", "romantic", "sex", "Parent_Status"]
students_corr[binary_vars] = students_corr[binary_vars].replace(binary_map)

corr = students_corr.select_dtypes(include="number").corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            square=True, cbar_kws={"shrink": 0.8})
plt.title("Correlation Heatmap - All Variables (Including Encoded Categoricals)", fontsize=13)
plt.tight_layout()
plt.savefig("correlation_heatmap_full.pdf", format="pdf", bbox_inches="tight")
plt.show()

#--------------------------------------------------------------------------------
# PART 3.6: Success vs Multiple Factors Comparison
#--------------------------------------------------------------------------------

# Variables to compare
vars_to_summarize = [
    "Family_Size", "guardian", "Parent_Status", "Family_Support",
    "Relationship_with_family", "Study_Time", "Daily_Alcohol",
    "Weekend_Alcohol", "Go_Out", "health", "absences"
]

categorical_vars = ["Family_Size", "guardian", "Parent_Status"]
binary_vars = ["Family_Support"]
numeric_vars = [
    "Relationship_with_family", "Study_Time", "Daily_Alcohol",
    "Weekend_Alcohol", "Go_Out", "health", "absences"
]

# Create comparison table
summary_by_success = pd.DataFrame(index=vars_to_summarize)

# Most frequent category for categorical variables
for var in categorical_vars:
    summary_by_success.loc[var, "Passed"] = students.loc[students["success"] == True, var].mode()[0]
    summary_by_success.loc[var, "Failed"] = students.loc[students["success"] == False, var].mode()[0]

# Proportion (%) of "yes" for binary variables
for var in binary_vars:
    summary_by_success.loc[var, "Passed"] = (
        students.loc[students["success"] == True, var].isin(["yes", True]).mean() * 100
    )
    summary_by_success.loc[var, "Failed"] = (
        students.loc[students["success"] == False, var].isin(["yes", True]).mean() * 100
    )

# Mean for numeric variables
for var in numeric_vars:
    summary_by_success.loc[var, "Passed"] = students.loc[students["success"] == True, var].mean()
    summary_by_success.loc[var, "Failed"] = students.loc[students["success"] == False, var].mean()

summary_by_success = summary_by_success.round(2)

print("Summary Table - Passed vs Failed Students:")
print(summary_by_success)
print()

#--------------------------------------------------------------------------------
# PART 3.7: Support Systems Interaction
#--------------------------------------------------------------------------------

# Analyze interaction of paid classes, family support, and school support
support_summary = (
    students
    .groupby(["Paid_classes", "Family_Support", "School_Support"])
    .agg(
        total_students=("success", "count"),
        pass_rate=("success", lambda x: x.isin(["yes", True]).mean() * 100)
    )
    .reset_index()
    .sort_values("pass_rate", ascending=False)
)

print("Success Rates by Support System Combinations:")
print(support_summary.round(2))
print()

#================================================================================#
# QUESTION 4: Principal Factors in Student Failure
#================================================================================#
# What are the principal factors that play a key role in student failure?
#================================================================================#

#--------------------------------------------------------------------------------
# PART 4.1: Success Rate by Absences
#--------------------------------------------------------------------------------

# Create absence groups
students['absence_group'] = 'Low (0-5)'
students.loc[students['absences'] > 5, 'absence_group'] = 'Medium (6-10)'
students.loc[students['absences'] > 10, 'absence_group'] = 'High (>10)'

absence_success = students.groupby('absence_group')['success'].apply(lambda x: (x == True).mean() * 100)

# Order to show logical progression
absence_order = ['Low (0-5)', 'Medium (6-10)', 'High (>10)']
absence_success = absence_success.reindex(absence_order)

plt.figure(figsize=(8, 5))
plt.bar(range(len(absence_success)), absence_success.values, color='#8ecae6', edgecolor='black')
plt.xlabel('Number of Absences')
plt.ylabel('Success Rate (%)')
plt.title('Success Rate by Absences')
plt.xticks(range(len(absence_success)), absence_order)
plt.tight_layout()
plt.savefig("success_by_absences.pdf", format="pdf", bbox_inches="tight")
plt.show()

#--------------------------------------------------------------------------------
# PART 4.2: Success Rate by Study Time
#--------------------------------------------------------------------------------

study_success = students.groupby('Study_Time')['success'].apply(lambda x: (x == True).mean() * 100)

plt.figure(figsize=(8, 5))
plt.bar(range(len(study_success)), study_success.values, color='#8ecae6', edgecolor='black')
plt.xlabel('Study Time')
plt.ylabel('Success Rate (%)')
plt.title('Success Rate by Weekly Study Time')
plt.xticks(range(len(study_success)), ['<2 hrs', '2-5 hrs', '5-10 hrs', '>10 hrs'])
plt.tight_layout()
plt.savefig("success_by_studytime.pdf", format="pdf", bbox_inches="tight")
plt.show()

#--------------------------------------------------------------------------------
# PART 4.3: Success Rate by Going Out Frequency
#--------------------------------------------------------------------------------

goout_success = students.groupby('Go_Out')['success'].apply(lambda x: (x == True).mean() * 100)

plt.figure(figsize=(8, 5))
plt.bar(range(len(goout_success)), goout_success.values, color='#f4a261', edgecolor='black')
plt.xlabel('Going Out Frequency')
plt.ylabel('Success Rate (%)')
plt.title('Success Rate by Going Out Frequency')
plt.xticks(range(len(goout_success)), ['Very Low', 'Low', 'Moderate', 'High', 'Very High'])
plt.tight_layout()
plt.savefig("success_by_goout.pdf", format="pdf", bbox_inches="tight")
plt.show()

#--------------------------------------------------------------------------------
# PART 4.4: Success Rate by Weekend Alcohol Consumption
#--------------------------------------------------------------------------------

alcohol_success = students.groupby('Weekend_Alcohol')['success'].apply(lambda x: (x == True).mean() * 100)

plt.figure(figsize=(8, 5))
plt.bar(range(len(alcohol_success)), alcohol_success.values, color='#f4a261', edgecolor='black')
plt.xlabel('Weekend Alcohol Consumption')
plt.ylabel('Success Rate (%)')
plt.title('Success Rate by Weekend Alcohol Consumption')
plt.xticks(range(len(alcohol_success)), ['Very Low', 'Low', 'Moderate', 'High', 'Very High'])
plt.tight_layout()
plt.savefig("success_by_alcohol.pdf", format="pdf", bbox_inches="tight")
plt.show()

#--------------------------------------------------------------------------------
# PART 4.5: Success Rate by Internet Access
#--------------------------------------------------------------------------------

internet_success = students.groupby('internet')['success'].apply(lambda x: (x == True).mean() * 100)

plt.figure(figsize=(6, 5))
plt.bar(range(len(internet_success)), internet_success.values,
        color=['#f4a261', '#8ecae6'], edgecolor='black')
plt.xlabel('Internet Access')
plt.ylabel('Success Rate (%)')
plt.title('Success Rate by Internet Access')
plt.xticks(range(len(internet_success)), internet_success.index)
plt.tight_layout()
plt.savefig("success_by_internet.pdf", format="pdf", bbox_inches="tight")
plt.show()

#--------------------------------------------------------------------------------
# PART 4.6: Success Rate by Family Relationship Quality
#--------------------------------------------------------------------------------

famrel_success = students.groupby('Relationship_with_family')['success'].apply(lambda x: (x == True).mean() * 100)

plt.figure(figsize=(8, 5))
plt.bar(range(len(famrel_success)), famrel_success.values, color='#8ecae6', edgecolor='black')
plt.xlabel('Quality of Family Relationships')
plt.ylabel('Success Rate (%)')
plt.title('Success Rate by Family Relationship Quality')
plt.xticks(range(len(famrel_success)), ['Very Bad', 'Bad', 'Moderate', 'Good', 'Excellent'])
plt.tight_layout()
plt.savefig("success_by_famrel.pdf", format="pdf", bbox_inches="tight")
plt.show()

#--------------------------------------------------------------------------------
# PART 4.7: Summary Comparison Chart
#--------------------------------------------------------------------------------

# Create a summary DataFrame showing the range of success rates for each factor
factor_ranges = pd.DataFrame({
    'Factor': ['Going Out', 'Absences', 'Alcohol', 'Family Relations', 'Study Time', 'Internet'],
    'Min Success Rate': [
        goout_success.min(),
        absence_success.min(),
        alcohol_success.min(),
        famrel_success.min(),
        study_success.min(),
        internet_success.min()
    ],
    'Max Success Rate': [
        goout_success.max(),
        absence_success.max(),
        alcohol_success.max(),
        famrel_success.max(),
        study_success.max(),
        internet_success.max()
    ]
})

factor_ranges['Impact (Range)'] = factor_ranges['Max Success Rate'] - factor_ranges['Min Success Rate']
factor_ranges = factor_ranges.sort_values('Impact (Range)', ascending=False)

# Visualize the impact of each factor
plt.figure(figsize=(10, 6))
plt.barh(range(len(factor_ranges)), factor_ranges['Impact (Range)'].values,
         color='#8ecae6', edgecolor='black')
plt.yticks(range(len(factor_ranges)), factor_ranges['Factor'].values)
plt.xlabel('Impact on Success Rate (Percentage Point Difference)')
plt.ylabel('Factor')
plt.title('Comparison of Factor Impact on Student Success')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("factor_impact_comparison.pdf", format="pdf", bbox_inches="tight")
plt.show()

print("\nFactor Impact Summary:")
print(factor_ranges)
print()

#--------------------------------------------------------------------------------
# Print Success Rates Summary
#--------------------------------------------------------------------------------

print("SUCCESS RATES SUMMARY BY FACTOR")

print("\nAbsences:")
print(absence_success)
print("\nStudy Time:")
print(study_success)
print("\nGoing Out:")
print(goout_success)
print("\nWeekend Alcohol:")
print(alcohol_success)
print("\nInternet Access:")
print(internet_success)
print("\nFamily Relationships:")
print(famrel_success)

#--------------------------------------------------------------------------------
# Written Conclusion from looking at the graphs
#--------------------------------------------------------------------------------

print("ANSWER: PRINCIPAL FACTORS IN STUDENT FAILURE")

print("""
Based on the analysis including correlation heatmaps, comparison tables,
and success rate visualizations, the principal factors in student failure are:

TOP RISK FACTORS (factors with strongest negative impact on success):
-----------------------------------------------------------------------
1. FREQUENT GOING OUT (Most Important Risk Factor)
   - Success rate drops from 80% (very low) to 51% (very high)
   - Nearly 30 percentage point difference - the largest effect observed
   - Students who go out frequently with friends are at highest risk

2. HIGH ABSENCES
   - High absences (>10): only 56% success rate
   - Low absences (0-5): 68% success rate
   - Clear pattern: more absences = lower success

3. HIGH ALCOHOL CONSUMPTION
   - Students with high weekend alcohol consumption have ~55% success rate
   - Those with very low consumption have ~67% success rate
   - Negative correlation visible in correlation heatmap

4. POOR FAMILY RELATIONSHIPS
   - Students with bad family relationships: 53% success rate
   - Students with excellent family relationships: 74% success rate
   - 21 percentage point difference shows importance of family support

5. LACK OF INTERNET ACCESS
   - No internet: 63.5% success rate
   - Has internet: 68.4% success rate
   - 5 percentage point difference, significant for equal opportunity

PROTECTIVE FACTORS (factors that support student success):
-----------------------------------------------------------
1. LOW GOING OUT FREQUENCY
   - Students who rarely go out have 80% success rate
   - Staying focused on academics rather than socializing is protective

2. GOOD/EXCELLENT FAMILY RELATIONSHIPS
   - Students with excellent family relationships: 74% success rate
   - Family support is crucial for student success

3. INTERNET ACCESS AT HOME
   - Essential resource for homework and research
   - Particularly important when combined with adequate study time

4. LOW ABSENCES
   - Regular attendance is one of the most important factors
   - Students who attend regularly perform much better

5. ADEQUATE STUDY TIME
   - Students studying 5-10 hours or >10 hours: ~74% success rate
   - Students studying <2 hours: 70% success rate
   - Effect is modest but becomes stronger with internet access

NOTES FROM CORRELATION HEATMAP:
-------------------------------
- Going out frequency shows strong negative correlation with success
- Alcohol consumption (both daily and weekend) negatively correlates with success
- Weekend and daily alcohol are highly correlated with each other (0.66)
- Age shows negative correlation, suggesting older students struggle more

MOST SURPRISING FINDING:
------------------------
Study time alone shows only modest impact on success (70% to 74% range).
However, from earlier analysis combining internet access and study time,
we see that internet access amplifies the benefit of study time significantly.

CONCLUSION:
-----------
The SINGLE MOST IMPORTANT risk factor for student failure is FREQUENT GOING OUT,
with nearly a 30 percentage point drop in success rates. The second most important
factor is HIGH ABSENCES. Together, these behavioral factors (going out and missing
class) represent the strongest predictors of failure.

Students at highest risk are those who: go out very frequently, have high absences,
lack internet access at home, consume alcohol regularly, and have weak family support.
Early identification of these factors can help schools provide timely interventions.
""")

# ================================================================================#
# QUESTION 5 (Optional): Machine Learning Classification Model
# ================================================================================#
# Build a simple machine learning classification model that predicts if a
# student will fail the class.
#
# We follow the methodology we learned from our Machine Learning Decision Tree lab:
# 1. Build a default decision tree and assess overfitting
# 2. Investigate the effect of key parameters (max_depth, min_samples_split)
# 3. Use post-pruning with cost complexity pruning
# 4. Use automatic model selection with GridSearchCV
# ================================================================================#

# --------------------------------------------------------------------------------
# PART 5.1: Data Preparation
# --------------------------------------------------------------------------------

# Select features for the model (all variables except success)
feature_columns = ['age', 'Study_Time', 'Relationship_with_family', 'Go_Out',
                   'Daily_Alcohol', 'Weekend_Alcohol', 'health', 'absences',
                   'sex', 'Family_Size', 'Parent_Status', 'School_Support',
                   'Family_Support', 'Paid_classes', 'activities', 'internet', 'romantic']

X = students[feature_columns].copy()
Y = students['success'].copy()

# Encode categorical variables (binary: yes/no, Male/Female, etc.)
binary_map = {
    'yes': 1, 'no': 0,
    'Male': 1, 'Female': 0,
    'living together': 1, 'living apart': 0,
    'greater than 3': 1, 'less or equal to 3': 0
}

# Apply encoding to categorical columns
categorical_cols = ['sex', 'Family_Size', 'Parent_Status', 'School_Support',
                    'Family_Support', 'Paid_classes', 'activities', 'internet', 'romantic']

for col in categorical_cols:
    X[col] = X[col].replace(binary_map)

# Convert success to binary (True=1, False=0)
Y = Y.astype(int)

# Split data into training and test sets (70% train, 30% test)
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.3,
    random_state=42,
    stratify=Y
)

# --------------------------------------------------------------------------------
# PART 5.2: Default Decision Tree and Overfitting Assessment
# --------------------------------------------------------------------------------

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Create and train a default decision tree
dt_default = DecisionTreeClassifier(random_state=42)
dt_default.fit(X_train, Y_train)

# Evaluate on both training and test sets
train_pred = dt_default.predict(X_train)
test_pred = dt_default.predict(X_test)

train_accuracy = accuracy_score(Y_train, train_pred)
test_accuracy = accuracy_score(Y_test, test_pred)

print("Default Decision Tree Performance:")
print("  Training accuracy: {:.2f}%".format(train_accuracy * 100))
print("  Test accuracy: {:.2f}%".format(test_accuracy * 100))

# --------------------------------------------------------------------------------
# PART 5.3: Investigating the Effect of max_depth
# --------------------------------------------------------------------------------

max_depths = range(1, 16)
train_accuracies = []
test_accuracies = []

for depth in max_depths:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X_train, Y_train)

    train_accuracies.append(accuracy_score(Y_train, dt.predict(X_train)))
    test_accuracies.append(accuracy_score(Y_test, dt.predict(X_test)))

# Create DataFrame for visualization
max_depth_results = pd.DataFrame({
    'max_depth': max_depths,
    'train': train_accuracies,
    'test': test_accuracies
})

# Plot results
plt.figure(figsize=(10, 6))
sns.lineplot(data=max_depth_results.melt('max_depth', var_name='set', value_name='accuracy'),
             x='max_depth', y='accuracy', hue='set', marker='o')
plt.xlabel('Maximum Depth')
plt.ylabel('Accuracy')
plt.title('Effect of max_depth on Model Performance')
plt.legend(title='Dataset')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("ml_max_depth_effect.pdf", format="pdf", bbox_inches="tight")
plt.show()

# --------------------------------------------------------------------------------
# PART 5.4: Investigating the Effect of min_samples_split
# --------------------------------------------------------------------------------

min_splits = [2, 5, 10, 20, 30, 40, 50, 60, 75, 100]
train_accuracies = []
test_accuracies = []

for min_split in min_splits:
    dt = DecisionTreeClassifier(min_samples_split=min_split, random_state=42)
    dt.fit(X_train, Y_train)

    train_accuracies.append(accuracy_score(Y_train, dt.predict(X_train)))
    test_accuracies.append(accuracy_score(Y_test, dt.predict(X_test)))

# Create DataFrame for visualization
min_split_results = pd.DataFrame({
    'min_samples_split': min_splits,
    'train': train_accuracies,
    'test': test_accuracies
})

# Plot results
plt.figure(figsize=(10, 6))
sns.lineplot(data=min_split_results.melt('min_samples_split', var_name='set', value_name='accuracy'),
             x='min_samples_split', y='accuracy', hue='set', marker='o')
plt.xlabel('Minimum Samples to Split')
plt.ylabel('Accuracy')
plt.title('Effect of min_samples_split on Model Performance')
plt.legend(title='Dataset')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("ml_min_samples_split_effect.pdf", format="pdf", bbox_inches="tight")
plt.show()

#--------------------------------------------------------------------------------
# PART 5.5: Automatic Model Selection with GridSearchCV
#--------------------------------------------------------------------------------

# Following the lab methodology: We already did the first train_test_split
# Now create a validation split object for GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit

v_split = StratifiedShuffleSplit(n_splits=1, train_size=0.75, random_state=42)

# Prepare parameter grid
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=42)

param_grid = {
    'max_depth': range(1, 11),
    'min_samples_split': [2, 10, 25, 50, 75, 85, 100, 200]
}

# Create GridSearchCV and fit
from sklearn.model_selection import GridSearchCV

dt_search = GridSearchCV(dt, param_grid, cv=v_split, n_jobs=-1)
dt_res = dt_search.fit(X_train, Y_train)

# Visualize results
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.DataFrame(dt_res.cv_results_['params'])
df['mean_test_score'] = dt_res.cv_results_['mean_test_score']

sns.set_theme()
sns.set(rc={'figure.figsize': (8, 6)})
sns.heatmap(df.pivot_table(index='max_depth', columns='min_samples_split',
                           values='mean_test_score'),
            cmap='viridis', annot=True)
plt.title('Grid Search Results - Decision Tree Classifier')
plt.xlabel('min_samples_split')
plt.ylabel('max_depth')
plt.tight_layout()
plt.savefig("ml_grid_search_heatmap.pdf", format="pdf", bbox_inches="tight")
plt.show()

# For classification, we use accuracy
print("\nBest parameters found: {}".format(dt_res.best_params_))
print("Best validation score: {:.4f}".format(dt_res.best_score_))

# Evaluate on test set (adapting the lab's final evaluation)
test_accuracy = accuracy_score(Y_test, dt_res.predict(X_test))
print("Test accuracy: {:.2f}%\n".format(test_accuracy * 100))

#--------------------------------------------------------------------------------
# PART 5.6: Visualize the Final Decision Tree
#--------------------------------------------------------------------------------

from sklearn.tree import plot_tree

# Get the best estimator from grid search
best_dt = dt_res.best_estimator_

# Visualize the tree
plt.figure(figsize=(20, 10))
plot_tree(best_dt,
          feature_names=feature_columns,
          class_names=['Failed', 'Passed'],
          filled=True,
          rounded=True,
          fontsize=10)
plt.title('Optimized Decision Tree from GridSearchCV', fontsize=14)
plt.tight_layout()
plt.savefig("ml_final_decision_tree.pdf", format="pdf", bbox_inches="tight")
plt.show()

""" GINI IMPURITY 
from 0 to 0.5 (for binary classification)
measures how "mixed" or "impure" the node is
gini = 0: Perfectly pure (all samples are the same class)
gini = 0.5: Maximum impurity (50% Failed, 50% Passed)
gini = 1 - (p_failed² + p_passed²)"""

""" VALUE
for example, value = [79, 165]
the meaning is [number of Failed students, number of Passed students] at this node
for example, 79 students failed, 165 students passed """

""" CLASS
determined by majority vote"""

# After visualizing the tree
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, dt_res.predict(X_test))
print(cm)