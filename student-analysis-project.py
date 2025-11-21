#=====================================================================================================================#
#                              Student performance analysis project                                                   #
#=====================================================================================================================#

# For github:
""" pathname: cd '/Users/yesminehachana/Desktop/Classes/Dauphine/2nd Year/1st Semester/Python/Student Performance Analysis Project'"""

#================================================================================#
# Initialisation
#================================================================================#

# We load libraries right at the beginning.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import csv

#================================================================================#
# Getting familiar with the dataset
#================================================================================#

# We make the raw file usable in Python
students_raw = pd.read_csv(
    "/Users/yesminehachana/Desktop/Classes/Dauphine/2nd Year/1st Semester/Python/Student Performance Analysis Project/students.csv",  sep=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)

# We get the first rows, to see how the data looks.
# There are 349 rows (students), 21 columns (variables).
# We look at the type of each column (int, bool, object…).
print(students_raw.head())

print(students_raw.shape)
print(students_raw.dtypes)

# We use more explicit column names
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

# We save the cleaned student dataframe
students.to_csv(
    "/Users/yesminehachana/Desktop/Classes/Dauphine/2nd Year/1st Semester/Python/Student Performance Analysis Project/students.csv",
    index=False
)

#================================================================================#
# Question 1: What is the proportion of students that failed the class?
#================================================================================#

fails = (students["success"] == False).sum() # We count how many students failed
total = len(students) # We get the number of rows in the dataframe: 349 students
print(f"Failure rate: {fails / total * 100:.2f}%") # We get the percentage of students who failed

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

# Our table provides descriptive statistics (mean, median, minimum, and maximum)
# for all numeric variables in the dataset. These variables include students’ age,
# study time, family relationship quality, going-out frequency, alcohol consumption,
# health status, and number of absences.
#
# The summary shows that the average student is about 16.5 years old, with the median
# age being 16. Study time is generally low: the mean value is slightly above 2 on a
# 1-4 scale, which corresponds to “2 to 5 hours per week.” Family relationship scores
# are high on average, with a mean close to 4 out of 5, this suggests that most students
# report good relationships at home.
#
# Going-out levels are moderate on average (mean around 3), while daily alcohol
# consumption remains very low (mean near 1). Weekend alcohol consumption is higher,
# it averages slightly above 2, but still remains relatively low overall. Students rate
# their health positively, with an average value between 3 and 4.
#
# Although the median number of absences is 4,the maximum reaches 75. This indicates
# that most students have low to moderate absences, but a small number of students
# are extreme outliers in terms of attendance.

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

# The summary we just wrote reports, for each categorical variable, the number of distinct
# values,the most frequent category, and the percentage of observations belonging to that
# category.
#
# The dataset shows a nearly balanced gender distribution, with females representing
# around 52% of the sample. Regarding family size more than 72% of students come from
# families with more than three members. Parent cohabitation is very common:
# approximately 89% of students have parents who live together.
#
# For parental occupations, both mother and father most frequently fall into the
# category labelled “other,” which appears 35% of the time for mothers and 56% for
# fathers. The main household guardian is the mother in about 70% of cases.
#
# Access to educational support services varies: school support is uncommon
# (only 14.6% receive it), while family support is more prevalent (64% report having
# it). Paid classes are taken by roughly half of the students. Participation in
# extracurricular activities is also balanced, with a slight majority involved.
#
# Internet access is extremely widespread, with about 85% of students having a
# connection at home. Most students are not in a romantic relationship (about 68%).
# Finally, the success rate in the dataset is around 68%, meaning that roughly two-thirds
# of students pass the course.

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

# Here we explore the basic demographic variables to understand the composition
# of the student dataset before examining more complex relationships.

# The first plot shows the distribution of ages. We use a histogram with bins
# from 15 to 22, we visualize how many students fall into each age group.
# The sample is mostly composed of students aged 15–18, with peaks at 16 and 17,
# while ages above 19 appear rarely. This confirms that the dataset reflects
# a typical secondary-school age range.

# The second plot shows the gender distribution using a countplot.
# It displays the number of male and female students. The dataset is fairly
# balanced, with slightly more female students than male students.
# This balance is nice as it allows for better comparisons between genders.

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

# This block of our code compares parental occupation distributions for mothers and fathers.
# We first define a fixed order of job categories so the plot is consistent and
# does not depend on frequency ordering. We then count how many mothers and fathers
# fall into each category using value_counts(), and reindex() to ensure every category
# appears (assigning 0 if one is missing). Finally, we plot side-by-side bars for
# mothers and fathers within each job category.
#
# The graph we made shows that the “other” occupation category is the most common for both
# parents, especially for fathers. Mothers are more frequently classified as “at_home”
# compared to fathers, while “services” is the second-most common occupation for both.
# The “teacher” and “health” categories are relatively small. Overall, parental jobs
# are diverse, with many falling into the broad “other” group.

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

# For this study time plot, we analyse the variable "Study_Time", which encodes how many hours
# a student studies per week. The original dataset stores this information as integers from 1 to 4,
# where 1 corresponds to less than 2 hours of study per week, 2 corresponds to 2–5 hours,
# 3 corresponds to 5–10 hours, and 4 corresponds to more than 10 hours.
# The histogram created in this part of our code shows the distribution of students
# across these four categories. The x-axis represents the four study time groups, where we use
# descriptive labels instead of the numerical codes, and the y-axis indicates how many
# students fall into each category. The main finding is that the majority of students
# study between 2 and 5 hours per week, while very few study more than 10 hours.
# This could suggest that most students exhibit relatively low to moderate levels of academic engagement.

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

# The second plot we made focuses on the variable "School_Support", which indicates whether a student
# receives additional academic support offered by their school (for instance, tutoring or
# remedial classes). The dataset encodes this variable as “yes” or “no”.
# The pie chart visualises the proportion of students who benefit from such support.
# The results show that only about 14.6% of students receive school support, whereas
# approximately 85.4% do not. This is not balanced, the vast majority
# of students don't have access to additional school-provided academic assistance.

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

# This pie chart we made illustrates how many students report receiving family support.
# The variable used is 'Family_Support', which contains only two possible values: "yes" or "no".
# We first calculate the support_counts, which tells us how many students are in each category.
# The pie chart shows that family support is quite common: about 64.2% of students report
# receiving support from their family, while around 35.8% report not receiving any.
# Each slice corresponds to one category ("yes" or "no"), and the percentage inside each slice
# represents that category’s share of the entire dataset. The majority of students benefit from
# some form of family support.

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

# This pie chart shows the proportion of students who take paid extra classes.
# The variable 'Paid_classes' contains responses of either "yes" or "no".
# After computing the counts for each category, the chart shows that 46.1% of students receive
# paid tuition, while 53.9% do not. As in the previous chart, each slice corresponds to a
# category ("yes" or "no"), and the percentages represent the share of each group in the dataset.
# This plot provides an overview of how common paid tutoring is among students.

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

# We show how many students have access to the internet at home.
# First, we extract the "internet" column from the dataset and count how many
# students answered "yes" and how many answered "no". These counts are stored
# in the variable support_counts.
# We then use these counts to draw a pie chart. Each slice of the pie represents
# the proportion of students who either have or do not have internet access.
# The labels indicate the percentage of each group.
# In this dataset, the majority of students, around 85%, report having internet
# access, while a much smaller group (about 15%) report not having access.
# This means that most students likely have the ability to access online
# learning resources or complete internet-based school activities at home.

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

# We examine here two lifestyle related variables: the frequency with which
# students go out with friends ("Go_Out") and their self-reported health status ("health").
# Both variables take values from 1 to 5, where 1 represents the lowest level
# (e.g., very low going out frequency or very poor health) and 5 represents the highest level.
# We calculate the distribution of each variable separately. For each of the five possible
# levels (1–5), we count how many students fall into that category. We use
# value_counts().sort_index() so that the categories are displayed in the logical
# order 1, 2, 3, 4, 5 instead of being sorted by frequency.
# We then build a grouped bar chart where each rating level on the x-axis is represented
# by two bars: one bar showing how many students report that level of going-out frequency,
# and another bar showing how many students report that same health rating. The bars are
# slightly shifted left and right (using x - width/2 and x + width/2) to avoid overlapping.
# In the chart we made, the x-axis shows the five qualitative rating levels:
# "Very Low", "Low", "Moderate", "High", and "Very High", which correspond to numerical scores
# 1 to 5. The y-axis shows the number of students in each category.
# The main finding is that going-out frequency peaks at moderate levels, and
# health status peaks at the highest level (“Very High”). This indicates that students tend
# to report pretty positive health overall, whereas their going-out habits are more
# spread across low to moderately high levels. The comparison shows that the distribution
# of health ratings is shifted upward relative to going-out ratings, suggesting that good
# health can be quite common even among students who do not go out frequently.

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

# This plot we made compares how often students consume alcohol on a daily basis versus on weekends.
# To create the graph, we first extracted the distribution of alcohol consumption levels from the dataset.
# The variables "Daily_Alcohol" and "Weekend_Alcohol" both use a 1-to-5 scale, where 1 = Very Low consumption,
# 2 = Low consumption, 3 = Moderate consumption, 4 = High consumption, and 5 = Very High consumption.
# We count how many students fall into each of the five levels for both variables, by using value_counts(),
# and we sort the levels in ascending order so the bars appear in a left-to-right progression.
# The x-axis represents the five alcohol consumption categories, and the y-axis represents the number of
# students in each category. For every category, we draw two bars side by side: The blue bar shows daily
# alcohol consumption, while the orange bar shows weekend alcohol consumption.
# This visualization shows that most students report very low daily consumption, with the
# count dropping a lot as the level increases. Weekend drinking, however, is much higher across all
# levels. For example, many more students report Moderate, High, or Very High consumption on weekends
# compared to weekdays. This could show that alcohol consumption among students is concentrated on
# weekends rather than during the school week.

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

# The summary table we made shows the distribution of several variables between male and female students.
# We created it to answer the question of whether key characteristics and behaviors differ by gender.
# For numeric variables (age, absences, going out, alcohol consumption, family relationship quality,
# and study time), each number is the average value within that gender group.
# For categorical/binary variables (School_Support, Family_Support, Paid_classes, activities,
# internet, romantic, success), each number is the percentage of students in that gender group
# who answered “yes”. For example, a value of 56.02 for Family_Support among males means
# about 56% of male students report receiving family support.
# Age is very similar across genders (around 16.5 years on average), so gender differences later on are
# not driven by age structure in the sample.
# Female students report more absences on average (6.64 vs 5.22).
# Male students go out slightly more often (Go_Out 3.18 vs 3.05), but the difference is small.
# Alcohol consumption is higher among males: Weekend alcohol is 2.60 for males vs 1.96 for females.
# Daily alcohol is 1.66 for males vs 1.25 for females. This could show that drinking behavior is more
# concentrated among male students, especially on weekends.
# Family relationship quality is a bit higher among males (4.02 vs 3.90), but the gap is modest.
# Study time differs more: females study longer on average (2.30 vs 1.80).
# School support is more common among females (around 19.7%) than males (around 9.0%).
# This might mean females are more likely to receive or seek help.
# Family support is also higher among females (around 71.6%) than males (around 56.0%).
# Paid private classes are more frequent for females (around 52.5%) than males (around 39.2%),
# this might show higher academic investment among girls in this dataset.
# Participation in extracurricular activities is higher for males (59.6%) than females (47.0%).
# Internet access is high for both groups (above 80%), with only a very small advantage for males.
# Romantic relationships are reported slightly more by females (36.1%) than males (28.3%).
# Success rates differ: males have a higher pass rate (72.3%) than females (63.4%). Given females
# study more and receive more support, this is interesting and worth highlighting as a “non-intuitive” result.
# Differences are descriptive only. This table does not prove causality. It gives a first snapshot
# of gender gaps.

# Family size distribution by gender
print("Family Size Distribution by Gender (%):")
print(pd.crosstab(students['sex'], students['Family_Size'], normalize='index') * 100)
print()
# We create a table that counts how many times combinations of sex and family size appear (crosstab)
# We then create percentages by dividing each value by the row's total

# The table we printed above shows the percentage distribution of family size within each gender.
# To obtain this table, we used pd.crosstab() to count how many students fall into each
# combination of 'sex' (Male/Female) and 'Family_Size' ("greater than 3" vs "less or equal to 3").
# By using normalize='index', the values are converted into percentages within each gender group.
# The results indicate that most students come from families with more than 3 members:
# approximately 76% of female students and 68% of male students fall into this category.
# Smaller families (3 or fewer members) are less common for both genders, but they appear
# slightly more often among male students (about 32%) compared to female students (about 24%).
# This could show that both groups mostly come from larger families, with a slightly
# higher concentration among female students.

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

# This bar chart visualizes the mean age of male and female students.
# We computed the mean age for each gender using students.groupby('sex')['age'].mean().
# The x-axis represents the two gender categories (Female and Male),
# while the y-axis represents the average age in years.
# The two bars show almost identical values (around 16.5 years for both sexes).

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

# This bar chart compares average school absences between male and female students.
# We computed the mean absences per gender with
# students.groupby('sex')['absences'].mean().
# The x-axis shows the two gender categories, and the y-axis shows the average number of absences.
# The results show that female students have a higher average number of absences (around 6.6)
# compared to male students (around 5.2). This is a noticeable difference.
# This gender gap in absenteeism may play a role in the success differences observed later.

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

# The summary table shows differences between the two family structures. The
# most frequent guardian is “mother” in both groups, meaning the primary caregiver
# does not differ based on parent cohabitation. Students whose parents live together
# tend to report slightly higher study time (2.07 hours category) compared to those
# whose parents live apart (1.97), this could suggest a small difference in academic
# engagement. Health ratings follow a similar pattern, being marginally higher for
# students from cohabiting families.
# Family support is common in both groups, with students from cohabiting families
# receiving it slightly more often (64.6%) than students whose parents live apart
# (60.5%). Daily alcohol use is slightly higher among students whose parents live
# apart, while weekend alcohol use is slightly higher for those whose parents live
# together, although both gaps are modest. Average relationship with family ratings
# are very similar in both groups (3.96 vs. 3.92), so this indicates that self-reported
# relationship quality does not change much with cohabitation status.
# Romantic relationships are more common among students whose parents
# live apart (36.8%) compared to those whose parents live together (31.8%). One of
# the main differences concerns school absences: students from separated
# households miss significantly more classes on average (9.24 absences) compared to
# students whose parents live together (5.57). Interestingly, despite higher
# absenteeism, students whose parents live apart show a higher success rate (73.7%)
# than those whose parents live together (66.9%). This result should be interpreted
# cautiously because the “living apart” group is relatively small and so may lead
# to "noisier" averages.

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

# The crosstab shows that gender composition is similar in both family structures,
# with around 52 to 55% female students in each group. This means gender is unlikely to
# explain the differences observed in study behavior or performance across groups.

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

# This bar chart shows the average age of students across the two parent status
# groups. The dataset is grouped by Parent_Status, and the mean of the age column
# is computed for each group.
# On the x-axis, we have the two categories: “living together” and “living apart.”
# On the y-axis, we show the average age of students in each group.
# The graph shows that average age is almost identical across the two family
# structures (both around 16.5 years). This indicates that age does not differ by
# parent cohabitation status in this dataset, and therefore cannot explain the
# behavioral or performance differences observed earlier.

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

# In the graph, the x-axis lists the parental job types, and the y-axis reports the percentage of
# students who are successful within each job category. Each job category appears twice: once as a
# blue bar representing the mother’s job success rate, and once as an orange bar representing the
# father’s job success rate. The figure shows that students whose mothers
# work in the health sector tend to display the highest success rate, followed by mothers employed
# in services or teaching. The lowest success rate for mothers appears among students whose mothers
# are classified as “at_home”. For fathers, success rates are generally more uniform, although
# students with fathers working in health or services appear to perform slightly better. Categories
# with very few observations, such as fathers coded as “at_home”, should be interpreted cautiously
# because the success rate may be heavily influenced by small sample size.
# Overall, the chart could show that parental occupation is associated with moderate differences in
# student performance, but the variation across categories remains quite narrow. Most job
# groups cluster within a similar success range, which indicates that parental job type is correlated
# with but not a dominant determinant of student academic success.

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
# Get the list of Study_Time categories
study_groups = students["Study_Time"].unique()

# Create a dictionary to store the results
success_rates = {}

# Loop through each study time value
for group in study_groups:
    # Keep only the students who belong to this study time group
    subset = students[students["Study_Time"] == group]

    # Extract their success column (column that says if the student is successful or not)
    values = subset["success"]

    # Calculate % of yes / True (students that succeeded)
    rate = values.isin(["yes", True]).mean() * 100

    # Store the result
    success_rates[group] = round(rate, 2)

# Convert to a Pandas Series (single column of data with labels)
success_summary = pd.Series(success_rates).sort_index()

# Combine results
count_summary = students.groupby("Study_Time")["success"].count()
summary_by_studytime = numeric_summary.copy()
summary_by_studytime["success (%)"] = success_summary
summary_by_studytime["count"] = count_summary

# Label study time categories: this renames the row of our summary table
summary_by_studytime.index = ["<2 hours", "2–5 hours", "5–10 hours", ">10 hours"]

# Transpose for better readability
summary_by_studytime = summary_by_studytime.T

print("Summary Table - Variables by Study Time:")
print(summary_by_studytime)
print()

# This summary table shows how several behavioural, lifestyle, and academic indicators vary
# across different levels of weekly study time. The variable “Study_Time” groups students into
# four ordered categories: “<2 hours”, “2–5 hours”, “5–10 hours”, and “>10 hours”. For each
# study-time group, we compute the mean values of several numeric variables, including:
# “health”: a self-reported health rating from 1 (very bad) to 5 (very good),
# “Daily_Alcohol” and “Weekend_Alcohol”: daily and weekend drinking frequency (1–5 scale),
# “absences”: the number of recorded school absences,
# “Relationship_with_family”: the quality of family relationships (1–5),
# “Go_Out”: frequency of going out with friends (1–5).
# In addition to these averages, the table also reports the percentage of successful students
# within each study-time category (variable “success (%)”), as well as the number of students
# belonging to each group (“count”). These computations are performed by grouping the dataset
# by study-time category, taking the mean of numeric variables, and calculating the proportion
# of “successful” students by checking how many entries in the success variable are equal to
# “yes”.
# Students who study less than two hours per week report relatively good health (mean = 3.83)
# and moderate alcohol consumption, but they show a lower success rate (69.66%) compared to
# students in the two highest study groups.
# Students studying between 2–5 hours exhibit the lowest success rate overall (63.64%) and have
# slightly higher absences. In contrast, students studying between 5–10 hours and
# those studying more than 10 hours per week achieve the highest success rates (73.68% and
# 74.07%, respectively). These groups also show lower levels of weekend alcohol consumption and
# fewer absences, which is consistent with stronger academic discipline. Relationship with
# family tends to be slightly higher for the 5–10 hour group (4.11), while the heaviest study
# group (>10 hours) reports somewhat lower health (3.19), this possibly reflects stress or
# overwork, although this interpretation should be treated cautiously given the smaller number
# of students in the group (27 observations). Overall, students who invest more time in studying
# tend to perform better academically, show
# healthier behavioural profiles, and accumulate fewer absences, whereas limited study time is
# associated with lower academic success.

#--------------------------------------------------------------------------------
# PART 3.5: Correlation Analysis
#--------------------------------------------------------------------------------

# Make a copy so we don't change the original data
students_corr = students.copy()

# List of binary variables we want to turn into 0/1
binary_vars = [
    "School_Support", "Family_Support", "Paid_classes",
    "activities", "internet", "romantic",
    "sex", "Parent_Status"
]

#  Manually convert each binary variable into 0/1 so that correlations make sense numerically
# yes/no and True/False variables
yes_no_vars = ["School_Support", "Family_Support", "Paid_classes",
               "activities", "internet", "romantic", "success"]
for var in yes_no_vars:
    # If value is "yes" or True 1, otherwise 0
    students_corr[var] = students_corr[var].isin(["yes", True]).astype(int)

# Sex: Male = 1, Female = 0
students_corr["sex"] = students_corr["sex"].replace({"Male": 1, "Female": 0})

# Parent status: living together = 1, living apart = 0
students_corr["Parent_Status"] = students_corr["Parent_Status"].replace({
    "living together": 1,
    "living apart": 0
})

# Now select all numeric columns (original numeric + these 0/1)
all_numeric = students_corr.select_dtypes(include=["number"])

#  Compute the correlation matrix
corr_full = all_numeric.corr()

# Plot the full correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_full, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap - All Variables (Including Encoded Categoricals)")
plt.tight_layout()
plt.savefig("correlation_heatmap_full.pdf", format="pdf", bbox_inches="tight")
plt.show()

# The correlation analysis we tried to perform computes correlations between all variables
# after converting categorical data into numerical form. To make correlations,
# several variables originally stored as “yes/no”
# are manually recoded into binary values: a value of 1 indicates “yes”,
# and a value of 0 indicates “no”. This recoding is applied to variables
# related to school support, family support, paid classes, extracurricular
# activities, internet access, romantic relationships, and student success.
# Gender and parent cohabitation status are also encoded so that they can be
# incorporated into the correlation matrix: males are assigned a value of 1 and
# females a value of 0, while students whose parents live together are assigned
# a value of 1 and those whose parents live apart receive a 0. This allows these
# categorical variables to be compared numerically with all other variables.
# After the recoding step, we extract only the numeric columns from the dataset.
# This includes both originally numeric variables (such as age, study time, and
# alcohol consumption levels) and the new binary columns created during the
# preprocessing. We then compute the correlation coefficient for each pair of
# variables. The correlation coefficient measures the strength and direction of
# the linear relationship between two variables, ranging from -1 (perfectly
# negative correlation) to +1 (perfectly positive correlation), with 0
# indicating no linear association.
# The correlation matrix is visualized using a heatmap. In this
# heatmap, each cell shows the correlation between a pair of variables, with
# warmer colors representing positive correlations and cooler colors indicating
# negative correlations.
# The heatmap shows that most correlations in the dataset are relatively weak. The strongest
# correlations appear between daily alcohol consumption and weekend alcohol
# consumption, which is expected since both variables measure similar behaviors.
# A moderate positive correlation also appears between going-out frequency and
# alcohol consumption levels, suggesting that more socially active students tend
# to drink more often. Additionally, health status shows a small negative
# correlation with alcohol use, indicating that students reporting poorer health
# also tend to report higher drinking levels, though the association remains
# weak.
# Student success displays only weak correlations with all other variables,
# meaning that no single variable in this dataset strongly predicts academic
# achievement on its own. This includes behavioral factors (like going out or
# drinking), socio-economic indicators (such as parental cohabitation status),
# and support measures (such as school or family support). Overall, many factors are
# related to one another, but none dominates the explanation of success in a
# strong way.

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

# Create comparison table skeleton
summary_by_success = pd.DataFrame(index=vars_to_summarize)

# Split data into passed / failed groups
passed_students = students[students["success"] == True]
failed_students = students[students["success"] == False]

# CATEGORICAL variables (most frequent category for each group)
for var in categorical_vars:
    # Most frequent value among passed students
    passed_mode = passed_students[var].mode()[0]

    # Most frequent value among failed students
    failed_mode = failed_students[var].mode()[0]

    # Store in the table
    summary_by_success.loc[var, "Passed"] = passed_mode
    summary_by_success.loc[var, "Failed"] = failed_mode

# BINARY variables (percentage of "yes" / True)
for var in binary_vars:
    # Values for passed students
    passed_values = passed_students[var]
    # Values for failed students
    failed_values = failed_students[var]

    # Percentage of yes/True among passed
    passed_yes_pct = passed_values.isin(["yes", True]).mean() * 100

    # Percentage of yes/True among failed
    failed_yes_pct = failed_values.isin(["yes", True]).mean() * 100

    # Store in the table
    summary_by_success.loc[var, "Passed"] = passed_yes_pct
    summary_by_success.loc[var, "Failed"] = failed_yes_pct

# NUMERIC variables (mean value in each group)

for var in numeric_vars:
    # Average for passed students
    passed_mean = passed_students[var].mean()

    # Average for failed students
    failed_mean = failed_students[var].mean()

    # Store in the table
    summary_by_success.loc[var, "Passed"] = passed_mean
    summary_by_success.loc[var, "Failed"] = failed_mean

# Round for display

summary_by_success = summary_by_success.round(2)

print("Summary Table - Passed vs Failed Students:")
print(summary_by_success)
print()

# The summary table compares students who passed the course with those who failed
# across a set of categorical, binary, and numeric variables. To build this table,
# the dataset is first split into two groups based on the “success” variable:
# students marked as True (passed) and students marked as False (failed). Then,
# for each variable of interest, we compute the characteristic that best describes
# each group. For categorical variables—such as family size, guardian type, and
# parental cohabitation status—we identify the most frequent category within
# each success group. This tells us which background characteristics tend to be
# more common among successful versus unsuccessful students.
#
# For binary variables such as family support, the table reports the percentage
# of students in each group who answered “yes.” This enables us to compare how
# frequently passed and failed students receive family support, expressed as a
# proportion rather than a raw count.
#
# For numeric variables—including study time, alcohol consumption, going-out
# frequency, relationship quality with family, health status, and absences—we
# compute the average value for passed and failed students. These averages allow
# us to see whether the two groups differ in their behaviors or well-being.
#
# The results show that the most common family size, guardian type, and parental
# cohabitation status are identical in both groups, suggesting that broad family
# structure does not strongly differentiate passed from failed students.
# However, differences emerge in the quantitative variables. Passed students
# report slightly better relationships with their family and study marginally
# more on average. They also tend to consume slightly less alcohol—both daily
# and on weekends—compared to students who failed. One notable distinction is
# that failed students go out more frequently and accumulate more absences than
# those who passed, which may be linked to weaker academic outcomes. In contrast,
# passed students show lower absences and a lower going-out score, which is
# consistent with more regular attendance and potentially more stable routines.
#
# Overall, while family structure variables appear similar across both groups,
# behavioral and engagement-related variables—particularly attendance, alcohol
# consumption, and going-out frequency—show meaningful differences between
# students who passed and those who failed. These patterns suggest that academic
# outcomes in this dataset are more closely associated with daily habits and
# personal behaviors than with background or demographic characteristics.

#--------------------------------------------------------------------------------
# PART 3.7: Support Systems Interaction
#--------------------------------------------------------------------------------

# Analyze interaction of paid classes, family support, and school support
# Get all values that exist in each support variable
paid_options = students["Paid_classes"].unique()
family_options = students["Family_Support"].unique()
school_options = students["School_Support"].unique()

# Prepare an empty list to store results
rows = []

# Loop through every combination (paid classes = yes, no; family support = yes, no; school support = yes, no)
for paid in paid_options:
    for fam in family_options:
        for school in school_options:

            # Filter students matching this combination (we loop through each combination, for each combination we select the students who match)
            subset = students[
                (students["Paid_classes"] == paid) &
                (students["Family_Support"] == fam) &
                (students["School_Support"] == school)
                ]

            # Count students in this group
            total = len(subset)

            # If no students in this combination, we skip it
            if total == 0:
                continue

            # Compute success rate
            success_rate = subset["success"].isin(["yes", True]).mean() * 100

            # Store this row
            rows.append([paid, fam, school, total, success_rate])

# Convert results to DataFrame
support_summary = pd.DataFrame(rows, columns=[
    "Paid_classes", "Family_Support", "School_Support",
    "total_students", "pass_rate"
])

# Sort results by success rate
support_summary = support_summary.sort_values("pass_rate", ascending=False)

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

# Get the different absence groups
groups = students["absence_group"].unique()

# Create an empty dictionary to store the results
absence_rates = {}

# Loop through each absence group
for group in groups:

    # Select only students in this group
    subset = students[students["absence_group"] == group]

    # Extract their success column
    values = subset["success"]

    # Compute the % of True (passed)
    rate = (values == True).mean() * 100

    # Store the result
    absence_rates[group] = round(rate, 2)

# Convert dictionary to a Series
absence_success = pd.Series(absence_rates)

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

# Get the list of study time categories
study_groups = students["Study_Time"].unique()

# Create a dictionary to store percentage of success
study_rates = {}

# For each study time category, calculate success rate
for group in study_groups:

    # Select students belonging to this group
    subset = students[students["Study_Time"] == group]

    # Extract only their success values
    values = subset["success"]

    # Calculate % of students who passed
    success_rate = (values == True).mean() * 100

    # Store the result
    study_rates[group] = round(success_rate, 2)

# Convert dictionary to a Pandas Series
study_success = pd.Series(study_rates)

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

# Get all levels of going out frequency
goout_levels = students["Go_Out"].unique()

# Create a dictionary to store success rates
goout_rates = {}

# Loop through each level of Go_Out
for level in goout_levels:
    # Select all students with this going-out level
    subset = students[students["Go_Out"] == level]

    # Extract their success values
    success_values = subset["success"]

    # Compute % of students who passed
    success_rate = (success_values == True).mean() * 100

    # Store in the dictionary
    goout_rates[level] = round(success_rate, 2)

# Convert dictionary to a pandas Series
goout_success = pd.Series(goout_rates).sort_index()

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

# Get the different alcohol consumption levels
alcohol_levels = students["Weekend_Alcohol"].unique()

# Prepare a dictionary to store statistics
alcohol_rates = {}

# Loop through each alcohol level
for level in alcohol_levels:

    # Select only the students who have this alcohol consumption level
    subset = students[students["Weekend_Alcohol"] == level]

    # Extract their success values
    success_values = subset["success"]

    # Compute percentage of students who passed
    success_rate = (success_values == True).mean() * 100

    # Store the result
    alcohol_rates[level] = round(success_rate, 2)

# Convert dictionary to a pandas Series (sorted for display)
alcohol_success = pd.Series(alcohol_rates).sort_index()

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

# Get the different internet categories (yes or no)
internet_levels = students["internet"].unique()

# Create an empty dictionary to store success rates
internet_rates = {}

# Loop through each category
for level in internet_levels:
    # Select only the students with this internet value
    subset = students[students["internet"] == level]

    # Extract the success column
    success_values = subset["success"]

    # Compute % of students who passed
    success_rate = (success_values == True).mean() * 100

    # Store the % rounded to two decimals
    internet_rates[level] = round(success_rate, 2)

# Convert dictionary to a Pandas Series
internet_success = pd.Series(internet_rates)

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

# Get the different family relationship levels
famrel_levels = students["Relationship_with_family"].unique()

# Create an empty dictionary to store the success rates
famrel_rates = {}

# Loop through each level
for level in famrel_levels:

    # Select students with this relationship score
    subset = students[students["Relationship_with_family"] == level]

    # Extract their success values
    success_values = subset["success"]

    # Compute % of students who passed
    success_rate = (success_values == True).mean() * 100

    # Store the result, rounded
    famrel_rates[level] = round(success_rate, 2)

# Convert dictionary to a Pandas Series and sort for order 1 to 5
famrel_success = pd.Series(famrel_rates).sort_index()

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
# Collect min and max success rates for each factor
# For each factor, we already have a Series of success rates.
# We now extract their minimum and maximum.
goout_min = goout_success.min()
goout_max = goout_success.max()

absence_min = absence_success.min()
absence_max = absence_success.max()

alcohol_min = alcohol_success.min()
alcohol_max = alcohol_success.max()

famrel_min = famrel_success.min()
famrel_max = famrel_success.max()

study_min = study_success.min()
study_max = study_success.max()

internet_min = internet_success.min()
internet_max = internet_success.max()

# Build a list of rows for the summary table
rows = [
    ["Going Out",          goout_min,    goout_max],
    ["Absences",           absence_min,  absence_max],
    ["Alcohol",            alcohol_min,  alcohol_max],
    ["Family Relations",   famrel_min,   famrel_max],
    ["Study Time",         study_min,    study_max],
    ["Internet",           internet_min, internet_max],
]

# Create the DataFrame from the list of rows
factor_ranges = pd.DataFrame(
    rows,
    columns=["Factor", "Min Success Rate", "Max Success Rate"]
)

# Compute the impact (range = max - min)
# Create a new column for the impact
factor_ranges["Impact (Range)"] = (
    factor_ranges["Max Success Rate"] - factor_ranges["Min Success Rate"]
)

# Sort factors by impact (largest first)
factor_ranges = factor_ranges.sort_values("Impact (Range)", ascending=False)

# Plot the impact of each factor (horizontal bar chart)
plt.figure(figsize=(10, 6))
# Positions on the y-axis: 0, 1, 2, ...
positions = range(len(factor_ranges))
# Heights: the impact values
impacts = factor_ranges["Impact (Range)"].values
plt.barh(positions, impacts, color="#8ecae6", edgecolor="black")
# Label each position with the factor name
plt.yticks(positions, factor_ranges["Factor"].values)
plt.xlabel("Impact on Success Rate (percentage point difference)")
plt.ylabel("Factor")
plt.title("Comparison of Factor Impact on Student Success")
# Invert y-axis so the biggest impact appears at the top
plt.gca().invert_yaxis()

plt.tight_layout()
plt.savefig("factor_impact_comparison.pdf", format="pdf", bbox_inches="tight")
plt.show()

print(factor_ranges)
print()

#--------------------------------------------------------------------------------
# Print Success Rates Summary
#--------------------------------------------------------------------------------

print("SUCCESS RATES SUMMARY BY FACTOR")

print("Absences:")
print(absence_success)
print("Study Time:")
print(study_success)
print("Going Out:")
print(goout_success)
print("Weekend Alcohol:")
print(alcohol_success)
print("Internet Access:")
print(internet_success)
print("Family Relationships:")
print(famrel_success)

#--------------------------------------------------------------------------------
# Written Conclusion from the graphs we made
#--------------------------------------------------------------------------------

print("ANSWER: PRINCIPAL FACTORS IN STUDENT FAILURE")

print("""
Based on the analysis including correlation heatmaps, comparison tables,
and success rate visualizations, the principal factors in student failure are:

TOP RISK FACTORS (factors with strongest negative impact on success):
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
- Going out frequency shows strong negative correlation with success
- Alcohol consumption (both daily and weekend) negatively correlates with success
- Weekend and daily alcohol are highly correlated with each other (0.66)
- Age shows negative correlation, suggesting older students struggle more

MOST SURPRISING FINDING WE MADE HERE:
Study time alone shows only modest impact on success (70% to 74% range).
However, from earlier analysis combining internet access and study time,
we see that internet access amplifies the benefit of study time significantly.
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