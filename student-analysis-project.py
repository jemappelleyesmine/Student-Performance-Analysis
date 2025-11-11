""" pathname: cd '/Users/yesminehachana/Desktop/Classes/Dauphine/2nd Year/1st Semester/Python/Student Performance Analysis Project'"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import csv

students = pd.read_csv(
    "/Users/yesminehachana/Desktop/Classes/Dauphine/2nd Year/1st Semester/Python/Student Performance Analysis Project/students.csv",  sep=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
#    sep=";",
 #   quotechar='"',
  #  quoting=csv.QUOTE_MINIMAL
    # quatotations in text values except numbers
    # seperator being used also appears in text panda wont be abel to load it.


print(students.head())

print(students.shape)
print(students.dtypes)

students = students.rename(columns={
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

# Question 1:
# What is the proportion of students that failed the class?
# Proportion of failed students
fails = (students["success"] == False).sum()
total = len(students)
print(f"Failure rate: {fails / total * 100:.2f}%")

# Question 2:
# How are different variables (age, gender, internet access, family composition, etc.) distributed in the dataset?

categorical_cols = ["sex","Family_Size","Parent_Status","Mother_Job","Father_Job","guardian","School_Support","Family_Support","Paid_classes","activities","internet","romantic","success"]

numeric_cols = ["age","Study_Time","Relationship_with_family","Go_Out","Daily_Alcohol","Weekend_Alcohol","health","absences"]

numeric_summary = students[numeric_cols].agg(['mean', 'median', 'min', 'max']).T.round(2)
numeric_summary.to_latex("numeric_summary.tex", index=True, caption="Descriptive Statistics of Numeric Variables", label="tab:numeric_summary") # latex save
numeric_summary

categorical_summary = pd.DataFrame({
    'Unique Values': [students[col].nunique() for col in categorical_cols],
    'Most Frequent': [students[col].mode()[0] for col in categorical_cols],
    'Frequency (%) of the Most Frequent': [
        round(students[col].value_counts(normalize=True).iloc[0] * 100, 2)
        for col in categorical_cols
    ]
}, index=categorical_cols)

categorical_summary

# Demographic & Background Variables
plt.figure(figsize=(6, 4))
ax = sns.histplot(
    data=students,x="age", bins=range(15, 23), color="skyblue", edgecolor="black")
plt.title("Distribution of Student Age (15–22)", fontsize=13)
plt.xlabel("Age")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("gender_distribution.pdf", format="pdf", bbox_inches="tight") # pdf save
plt.show()

plt.figure(figsize=(4, 4))
sns.countplot(data=students, x="sex", palette=["#8ecae6", "#f4a261"])
plt.title("Distribution of Students by Gender")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Prepare counts for each category
mother_counts = students["Mother_Job"].value_counts().reindex(
    ["at_home", "health", "other", "services", "teacher"], fill_value=0
)
father_counts = students["Father_Job"].value_counts().reindex(
    ["at_home", "health", "other", "services", "teacher"], fill_value=0
)

# Define x positions and bar width
x = np.arange(len(mother_counts))
width = 0.35

# Plot bars side by side
plt.figure(figsize=(8, 5))
plt.bar(x - width/2, mother_counts, width, color="#8ecae6", label="Mother's Job")
plt.bar(x + width/2, father_counts, width, color="#f4a261", label="Father's Job")

# Relabeled tick names for clarity
display_labels = ["At Home", "Health", "Other", "Services", "Teacher"]
plt.xticks(x, display_labels, rotation=15)

# Aesthetic details
plt.xlabel("Job Category")
plt.ylabel("Count of Parents")
plt.title("Comparison of Mother's and Father's Occupations", fontsize=14)
plt.legend(title="Parent Job")
plt.tight_layout()
plt.show()

# Academic Engagement Variables

plt.figure(figsize=(5, 4))
ax = sns.histplot(
    data=students,
    x="Study_Time",
    bins=range(1, 6),  # 1 to 4 inclusive
    color="skyblue",
    edgecolor="black"
)
# Set custom labels for numeric categories
ax.set_xticks([1.5, 2.5, 3.5, 4.5])
ax.set_xticklabels(["<2 hours", "2 to 5 hours", "5 to 10 hours", "> 10 hours"])

plt.title("Weekly Study Time", fontsize=13)
plt.xlabel("Study Time")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Count the categories
support_counts = students["School_Support"].value_counts()

colors = ["#8ecae6" if label == "yes" else "#f4a261" for label in support_counts.index]

# Pie chart
plt.figure(figsize=(5, 5))
plt.pie(
    support_counts,
    labels=support_counts.index,
    autopct='%1.1f%%',
    colors=colors,
    startangle=90,
    explode=(0.05, 0),
    wedgeprops={"edgecolor": "white"}
)

plt.title("Proportion of Students Receiving School Support", fontsize=13)
plt.tight_layout()
plt.show()

# Count the categories
support_counts = students["Family_Support"].value_counts()

colors = ["#8ecae6" if label == "yes" else "#f4a261" for label in support_counts.index]

# Pie chart
plt.figure(figsize=(5, 5))
plt.pie(
    support_counts,
    labels=support_counts.index,
    autopct='%1.1f%%',
    colors=colors,
    startangle=90,
    explode=(0.05, 0),
    wedgeprops={"edgecolor": "white"}
)

plt.title("Proportion of Students Receiving Family Support", fontsize=13)
plt.tight_layout()
plt.show()

# Count the categories
support_counts = students["Paid_classes"].value_counts()
colors = ["#8ecae6" if label == "yes" else "#f4a261" for label in support_counts.index]

# Pie chart
plt.figure(figsize=(5, 5))
plt.pie(
    support_counts,
    labels=support_counts.index,
    autopct='%1.1f%%',
    colors=colors,
    startangle=90,
    explode=(0.05, 0),
    wedgeprops={"edgecolor": "white"}
)

plt.title("Proportion of Students Receiving Paid Tuition", fontsize=13)
plt.tight_layout()
plt.show()

# Count the categories
support_counts = students["internet"].value_counts()
colors = ["#8ecae6" if label == "yes" else "#f4a261" for label in support_counts.index]

# Pie chart
plt.figure(figsize=(5, 5))
plt.pie(
    support_counts,
    labels=support_counts.index,
    autopct='%1.1f%%',
    colors=colors,
    startangle=90,
    explode=(0.05, 0),
    wedgeprops={"edgecolor": "white"}
)

plt.title("Proportion of Students with Access to Internet", fontsize=13)
plt.tight_layout()
plt.show()

# Social & Lifestyle Variables; activities and romance, write about them from the table
# Health & Well-being Variables

# Prepare data
out_counts = students["Go_Out"].value_counts().sort_index()
health_counts = students["health"].value_counts().sort_index()

# Define x positions and bar width
x = np.arange(1, 6)  # 1 to 5 levels
width = 0.35

plt.figure(figsize=(7, 5))
plt.bar(x - width/2, out_counts, width, color="#8ecae6", label="Meeting friends")
plt.bar(x + width/2, health_counts, width, color="#f4a261", label="Healthy")

# Customize
plt.xticks(x, ["Very Low", "Low", "Moderate", "High", "Very High"])
plt.xlabel("")
plt.ylabel("Count of Students")
plt.title("", fontsize=14)
plt.legend(title="")
plt.tight_layout()
plt.show()
# maybe they cannot concentrate more when there health is not best, but they dont go outwhen is better, they study nmore
# at home using. lets see with health and no of hours

# Prepare data
daily_counts = students["Daily_Alcohol"].value_counts().sort_index()
weekend_counts = students["Weekend_Alcohol"].value_counts().sort_index()

# Define x positions and bar width
x = np.arange(1, 6)  # 1 to 5 levels
width = 0.35

plt.figure(figsize=(7, 5))
plt.bar(x - width/2, daily_counts, width, color="#8ecae6", label="Daily Alcohol")
plt.bar(x + width/2, weekend_counts, width, color="#f4a261", label="Weekend Alcohol")

# Customize
plt.xticks(x, ["Very Low", "Low", "Moderate", "High", "Very High"])
plt.xlabel("Alcohol Consumption Level")
plt.ylabel("Count of Students")
plt.title("Daily vs Weekend Alcohol Consumption", fontsize=14)
plt.legend(title="Type of Day")
plt.tight_layout()
plt.show()

numeric_cols = ["age","Study_Time","Relationship_with_family","Go_Out","Daily_Alcohol","Weekend_Alcohol","health","absences"]

# How do variables interact with each other?
# 1. Are age, absences, family composition, etc. distributed similarly for both genders?
# 2. What are the age, gender, internet access, alcohol consumption, etc. distributions by parents cohabitation status?
# ...
# etc..

# 1. Are age, absences, family composition, etc. distributed similarly for both genders?

# Variables to include
vars_to_summarize = [
    "age", "absences", "Go_Out", "Weekend_Alcohol", "Daily_Alcohol",
    "Relationship_with_family", "Study_Time",
    "School_Support", "Family_Support", "Paid_classes",
    "activities", "internet", "romantic", "success"
]

# Separate numeric and categorical variables
numeric_vars = [
    "age", "absences", "Go_Out", "Weekend_Alcohol", "Daily_Alcohol",
    "Relationship_with_family", "Study_Time"
]

categorical_vars = [
    "School_Support", "Family_Support", "Paid_classes",
    "activities", "internet", "romantic", "success"
]

# Initialize summary DataFrame
summary_by_gender = pd.DataFrame(index=vars_to_summarize)

# Calculate means for numeric variables by gender
for var in numeric_vars:
    summary_by_gender.loc[var, "Male"] = students.loc[students["sex"] == "Male", var].mean()
    summary_by_gender.loc[var, "Female"] = students.loc[students["sex"] == "Female", var].mean()

# Calculate percentage of "yes" or True for categorical variables by gender
for var in categorical_vars:
    summary_by_gender.loc[var, "Male"] = (
        students.loc[students["sex"] == "Male", var].isin(["yes", True]).mean() * 100
    )
    summary_by_gender.loc[var, "Female"] = (
        students.loc[students["sex"] == "Female", var].isin(["yes", True]).mean() * 100
    )

# Round results to 2 decimals
summary_by_gender = summary_by_gender.round(2)

# Display final summary
summary_by_gender

# Additional analysis for Question 4: Family composition by gender

print("\n\nFamily Size Distribution by Gender:")
print(pd.crosstab(students['sex'], students['Family_Size'], normalize='index') * 100)

# Visualize age by gender
plt.figure(figsize=(6, 5))
age_by_gender = students.groupby('sex')['age'].mean()
plt.bar(range(len(age_by_gender)), age_by_gender.values, color=['#8ecae6', '#f4a261'], edgecolor='black')
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
plt.bar(range(len(absences_by_gender)), absences_by_gender.values, color=['#8ecae6', '#f4a261'], edgecolor='black')
plt.xticks(range(len(absences_by_gender)), absences_by_gender.index)
plt.ylabel('Average Absences')
plt.xlabel('Gender')
plt.title('Average Absences by Gender')
plt.tight_layout()
plt.savefig("absences_by_gender.pdf", format="pdf", bbox_inches="tight")
plt.show()


# 2. What are the age, gender, internet access, alcohol consumption, guardian etc. distributions by parents cohabitation status?

# Variables to include
vars_to_summarize = [
    "guardian", "Study_Time", "health",
    "Family_Support", "Daily_Alcohol", "Weekend_Alcohol",
    "Relationship_with_family", "romantic", "absences", "success"
]

# Separate numeric and categorical variables
numeric_vars = [
    "Study_Time", "health", "Daily_Alcohol",
    "Weekend_Alcohol", "Relationship_with_family", "absences"
]

# Variables for proportions (%)
proportion_vars = ["Family_Support", "romantic", "success"]

# Initialize table
summary_by_parent_status = pd.DataFrame(index=vars_to_summarize)

# Compute mean values for numeric variables by Parent_Status
for var in numeric_vars:
    summary_by_parent_status.loc[var, "Living Together"] = students.loc[
        students["Parent_Status"] == "living together", var
    ].mean()
    summary_by_parent_status.loc[var, "Living Apart"] = students.loc[
        students["Parent_Status"] == "living apart", var
    ].mean()

# Compute proportions (%) for selected categorical variables
for var in proportion_vars:
    summary_by_parent_status.loc[var, "Living Together"] = (
        students.loc[students["Parent_Status"] == "living together", var]
        .isin(["yes", True]).mean() * 100
    )
    summary_by_parent_status.loc[var, "Living Apart"] = (
        students.loc[students["Parent_Status"] == "living apart", var]
        .isin(["yes", True]).mean() * 100
    )

# Compute most frequent category for 'guardian'
summary_by_parent_status.loc["guardian", "Living Together"] = students.loc[
    students["Parent_Status"] == "living together", "guardian"
].mode()[0]
summary_by_parent_status.loc["guardian", "Living Apart"] = students.loc[
    students["Parent_Status"] == "living apart", "guardian"
].mode()[0]

# Round numeric results to 2 decimals
summary_by_parent_status = summary_by_parent_status.round(2)

# Display the final summary
summary_by_parent_status

# Additional analysis for Question 5: Gender distribution by parent status

print("\n\nGender Distribution by Parent Cohabitation Status:")
gender_by_parent = pd.crosstab(students['Parent_Status'], students['sex'], normalize='index') * 100
print(gender_by_parent.round(2))

# Visualize age distribution by parent status
plt.figure(figsize=(6, 5))
age_by_parent = students.groupby('Parent_Status')['age'].mean()
plt.bar(range(len(age_by_parent)), age_by_parent.values, color=['#f4a261', '#8ecae6'], edgecolor='black')
plt.xticks(range(len(age_by_parent)), age_by_parent.index, rotation=15)
plt.ylabel('Average Age')
plt.xlabel('Parent Cohabitation Status')
plt.title('Average Age by Parent Cohabitation Status')
plt.tight_layout()
plt.savefig("age_by_parent_status.pdf", format="pdf", bbox_inches="tight")
plt.show()

# 3. Parental Job Type vs Student Success

# Compute success rates (or counts) by parental job type
mother_success = (
    students.groupby("Mother_Job")["success"]
    .apply(lambda x: x.isin(["yes", True]).mean() * 100)
    .reset_index()
    .rename(columns={"success": "Success_Rate"})
)

father_success = (
    students.groupby("Father_Job")["success"]
    .apply(lambda x: x.isin(["yes", True]).mean() * 100)
    .reset_index()
    .rename(columns={"success": "Success_Rate"})
)

# Create figure and axes
plt.figure(figsize=(8, 5))

# Bar width and positions
x = range(len(mother_success))
width = 0.35

# Plot both bars side by side
plt.bar(
    [i - width/2 for i in x],
    mother_success["Success_Rate"],
    width=width,
    label="Mother's Job",
    color="#8ecae6"
)
plt.bar(
    [i + width/2 for i in x],
    father_success["Success_Rate"],
    width=width,
    label="Father's Job",
    color="#f4a261"
)

# Customize chart
plt.xticks(ticks=x, labels=mother_success["Mother_Job"], rotation=30, ha="right")
plt.xlabel("Parental Job Type")
plt.ylabel("Success Rate (%)")
plt.title("Student Success Rate by Parental Job Type")
plt.legend(title="")
plt.tight_layout()
plt.show()

# 4. “Do students who study more hours per week also show better health, fewer absences, stronger family relationships,
# and higher success rates — and does their alcohol consumption or social activity differ?”

# Variables to summarize
vars_to_summarize = [
    "health", "Daily_Alcohol", "Weekend_Alcohol",
    "absences", "Relationship_with_family", "Go_Out", "success"
]

# Separate numeric and categorical variables
numeric_vars = [
    "health", "Daily_Alcohol", "Weekend_Alcohol",
    "absences", "Relationship_with_family", "Go_Out"
]

categorical_vars = ["success"]

# Compute mean for numeric variables by Study_Time
numeric_summary = (
    students.groupby("Study_Time")[numeric_vars]
    .mean()
    .round(2)
)

# Compute proportion (%) of "yes" for categorical variable (success)
success_summary = (
    students.groupby("Study_Time")["success"]
    .apply(lambda x: x.isin(["yes", True]).mean() * 100)
    .round(2)
)

# Combine both results into one DataFrame
count_summary = students.groupby("Study_Time")["success"].count()

summary_by_studytime = numeric_summary.copy()
summary_by_studytime["success (%)"] = success_summary
summary_by_studytime["count"] = count_summary


# Label the study time categories
summary_by_studytime.index = ["<2 hours", "2–5 hours", "5–10 hours", ">10 hours"]

# Transpose to make columns the study time categories
summary_by_studytime = summary_by_studytime.T # (transposed for column)

# Display final summary
summary_by_studytime #  cross-tabulation table

# 5. Heatmap

# Compute correlation matrix
corr = students.select_dtypes(include=['number']).corr()
# Create a mask for the upper triangle — but keep the diagonal visible
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)  # k=1 hides only upper triangle

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
    square=True, cbar_kws={"shrink": 0.8}
)
plt.title("Correlation Heatmap (Lower Triangle with Diagonal)", fontsize=13)
plt.tight_layout()
plt.show()

# Make a copy and encode binary variables
students_corr = students.copy()
binary_map = {
    "yes": 1, "no": 0, True: 1, False: 0,
    "Male": 1, "Female": 0,
    "living together": 1, "living apart": 0
}
binary_vars = ["School_Support", "Family_Support", "Paid_classes",
               "activities", "internet", "romantic", "sex", "Parent_Status"]
students_corr[binary_vars] = students_corr[binary_vars].replace(binary_map)

# Compute correlation matrix
corr = students_corr.select_dtypes(include="number").corr()

# Full heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar_kws={"shrink": 0.8})
plt.title("Correlation Heatmap", fontsize=13)
plt.tight_layout()
plt.show()

# success result by factors. a lot of vars are not significant.

# Variables to include
vars_to_summarize = [
    "Family_Size", "guardian", "Parent_Status", "Family_Support",
    "Relationship_with_family", "Study_Time", "Daily_Alcohol",
    "Weekend_Alcohol", "Go_Out", "health", "absences"
]

# Separate variable types
categorical_vars = ["Family_Size", "guardian", "Parent_Status"]
binary_vars = ["Family_Support"]
numeric_vars = [
    "Relationship_with_family", "Study_Time", "Daily_Alcohol",
    "Weekend_Alcohol", "Go_Out", "health", "absences"
]

# Initialize table
summary_by_success = pd.DataFrame(index=vars_to_summarize)

# Most frequent category for categorical variables
for var in categorical_vars:
    summary_by_success.loc[var, "Passed"] = students.loc[students["success"] == True, var].mode()[0]
    summary_by_success.loc[var, "Failed"] = students.loc[students["success"] == False, var].mode()[0]

# Proportion (%) of “yes” for binary Family_Support
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

# Round numeric values
summary_by_success = summary_by_success.round(2)

# Display
summary_by_success

# extra question, interestig to look at, does paid classes increase succes rate *** to changhe
# import pandas as pd

# Step 1: Create a filtered summary table
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

# Step 2: Display summary
support_summary.round(2)


# ============================================================================
# Question 6: What are the principal factors that play a key role in student failure?
# ============================================================================

print("\n" + "="*80)
print("QUESTION 6: PRINCIPAL FACTORS IN STUDENT FAILURE")
print("="*80 + "\n")

# Display Aashish's comparison table
print("Comparison of Passed vs Failed Students:")
print(summary_by_success)
print("\n")

print("Analyzing success rates by different factors...\n")

# VISUALIZATION 1: Success Rate by Absences
students['absence_group'] = 'Low (0-5)'
students.loc[students['absences'] > 5, 'absence_group'] = 'Medium (6-10)'
students.loc[students['absences'] > 10, 'absence_group'] = 'High (>10)'

absence_success = students.groupby('absence_group')['success'].apply(lambda x: (x == True).mean() * 100)

plt.figure(figsize=(8, 5))
plt.bar(range(len(absence_success)), absence_success.values, color='#8ecae6', edgecolor='black')
plt.xlabel('Number of Absences')
plt.ylabel('Success Rate (%)')
plt.title('Success Rate by Absences')
plt.xticks(range(len(absence_success)), ['High (>10)', 'Low (0-5)', 'Medium (6-10)'])
plt.tight_layout()
plt.savefig("success_by_absences.pdf", format="pdf", bbox_inches="tight")
plt.show()

# VISUALIZATION 2: Success Rate by Study Time
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

# VISUALIZATION 3: Success Rate by Going Out
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

# VISUALIZATION 4: Success Rate by Weekend Alcohol
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

# VISUALIZATION 5: Success Rate by Internet Access
internet_success = students.groupby('internet')['success'].apply(lambda x: (x == True).mean() * 100)

plt.figure(figsize=(6, 5))
plt.bar(range(len(internet_success)), internet_success.values, color=['#f4a261', '#8ecae6'], edgecolor='black')
plt.xlabel('Internet Access')
plt.ylabel('Success Rate (%)')
plt.title('Success Rate by Internet Access')
plt.xticks(range(len(internet_success)), internet_success.index)
plt.tight_layout()
plt.savefig("success_by_internet.pdf", format="pdf", bbox_inches="tight")
plt.show()

# VISUALIZATION 6: Success Rate by Family Relationships
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

# Print success rates summary
print("\n" + "="*80)
print("SUCCESS RATES SUMMARY")
print("="*80)
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

# FINAL ANSWER - Based on all visualizations and analysis
print("\n" + "="*80)
print("ANSWER: PRINCIPAL FACTORS IN STUDENT FAILURE")
print("="*80)
print("""
Based on the comprehensive analysis including correlation heatmaps, comparison tables,
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

4. ROMANTIC RELATIONSHIPS
   - From earlier analysis: students in relationships have 59% success vs 72% not in relationships
   - Represents a 13 percentage point difference

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
   - Students with bad family relationships: 53% success rate
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

KEY INSIGHTS FROM CORRELATION HEATMAP:
--------------------------------------
- Going out frequency shows strong negative correlation with success
- Alcohol consumption (both daily and weekend) negatively correlates with success
- Weekend and daily alcohol are highly correlated with each other (0.66)
- Age shows negative correlation, suggesting older students struggle more

MOST SURPRISING FINDING:
------------------------
Study time alone shows only modest impact on success (70% to 74% range).
However, from Aashish's earlier analysis combining internet access and study time,
we see that internet access amplifies the benefit of study time significantly.

FINAL CONCLUSION:
-----------------
The SINGLE MOST IMPORTANT risk factor for student failure is FREQUENT GOING OUT,
with nearly a 30 percentage point drop in success rates. The second most important
factor is HIGH ABSENCES. Together, these behavioral factors (going out and missing
class) represent the strongest predictors of failure.

Schools should focus intervention efforts on:
1. Monitoring and reducing student absences through engagement programs
2. Educating students about balancing social life with academic responsibilities
3. Providing internet access to students who lack it at home
4. Encouraging family involvement and strong family-student relationships
5. Addressing alcohol consumption through counseling and support programs

Students at highest risk are those who: go out very frequently, have high absences,
lack internet access at home, consume alcohol regularly, and have weak family support.
Early identification of these factors can help schools provide timely interventions.
""")
print("="*80)

# ============================================================================
# Question 7 (Optional): Build a simple machine learning classification model
# ============================================================================

print("QUESTION 7: MACHINE LEARNING MODEL TO PREDICT STUDENT FAILURE")

print("Building a Decision Tree Classifier to predict student success/failure...\n")

# Step 1: Prepare the data - separate features (X) and target (Y)


# Select features for the model (all except success)
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

print("Features prepared: {} features, {} samples".format(X.shape[1], X.shape[0]))
print("Target variable: success (1 = passed, 0 = failed)\n")

# Step 2: Split data into training and test sets


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.3,
    random_state=42,
    stratify=Y
)

print("Training set: {} samples".format(len(X_train)))
print("Test set: {} samples\n".format(len(X_test)))

# Step 3: Train a Decision Tree Classifier


from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=20,
    random_state=42
)

dt.fit(X_train, Y_train)


# Step 4: Make predictions


Y_pred = dt.predict(X_test)

# Step 5: Evaluate the model


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Calculate accuracy
accuracy = accuracy_score(Y_test, Y_pred)
print("="*60)
print("MODEL PERFORMANCE")
print("="*60)
print("Accuracy: {:.2f}%".format(accuracy * 100))
print()

# Confusion matrix
cm = confusion_matrix(Y_test, Y_pred)
print(cm)
print("Confusion Matrix:")
print("                 Predicted Failed  Predicted Passed")
print("Actually Failed      {}                {}".format(cm[0,0], cm[0,1]))
print("Actually Passed      {}                {}\n".format(cm[1,0], cm[1,1]))

# Classification report
print("Classification Report:")
print(classification_report(Y_test, Y_pred, target_names=['Failed', 'Passed']))

# Step 6: Feature importance
print("\n" + "="*60)
print("FEATURE IMPORTANCE")
print("="*60)

feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': dt.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 10 Most Important Features:")
print("-"*60)
for i, row in feature_importance.head(10).iterrows():
    print("{:30s}: {:.4f}".format(row['Feature'], row['Importance']))

# Visualize feature importance
plt.figure(figsize=(10, 6))
top_features = feature_importance.head(10)
plt.barh(range(len(top_features)), top_features['Importance'].values, color='#8ecae6', edgecolor='black')
plt.yticks(range(len(top_features)), top_features['Feature'].values)
plt.xlabel('Importance')
plt.title('Top 10 Most Important Features for Predicting Student Success')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("ml_feature_importance.pdf", format="pdf", bbox_inches="tight")
plt.show()

# Step 7: Visualize the decision tree (simplified version)
print("\n\nStep 6: Visualizing decision tree structure...")

from sklearn.tree import plot_tree

plt.figure(figsize=(20, 10))
plot_tree(dt,
          feature_names=feature_columns,
          class_names=['Failed', 'Passed'],
          filled=True,
          rounded=True,
          fontsize=10)
plt.title('Decision Tree for Student Success Prediction (max_depth=5)')
plt.tight_layout()
plt.savefig("ml_decision_tree.pdf", format="pdf", bbox_inches="tight")
plt.show()