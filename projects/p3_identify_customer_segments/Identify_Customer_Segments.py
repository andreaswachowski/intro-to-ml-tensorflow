# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Project: Identify Customer Segments
#
# In this project, you will apply unsupervised learning techniques to identify segments of the population that form the core customer base for a mail-order sales company in Germany. These segments can then be used to direct marketing campaigns towards audiences that will have the highest expected rate of returns. The data that you will use has been provided by our partners at Bertelsmann Arvato Analytics, and represents a real-life data science task.
#
# This notebook will help you complete this task by providing a framework within which you will perform your analysis steps. In each step of the project, you will see some text describing the subtask that you will perform, followed by one or more code cells for you to complete your work. **Feel free to add additional code and markdown cells as you go along so that you can explore everything in precise chunks.** The code cells provided in the base template will outline only the major tasks, and will usually not be enough to cover all of the minor tasks that comprise it.
#
# It should be noted that while there will be precise guidelines on how you should handle certain tasks in the project, there will also be places where an exact specification is not provided. **There will be times in the project where you will need to make and justify your own decisions on how to treat the data.** These are places where there may not be only one way to handle the data. In real-life tasks, there may be many valid ways to approach an analysis task. One of the most important things you can do is clearly document your approach so that other scientists can understand the decisions you've made.
#
# At the end of most sections, there will be a Markdown cell labeled **Discussion**. In these cells, you will report your findings for the completed section, as well as document the decisions that you made in your approach to each subtask. **Your project will be evaluated not just on the code used to complete the tasks outlined, but also your communication about your observations and conclusions at each stage.**

# %%
# import libraries here; add more as necessary
import json
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler

# magic word for producing visualizations in notebook
# %matplotlib inline

'''
Import note: The classroom currently uses sklearn version 0.19.
If you need to use an imputer, it is available in sklearn.preprocessing.Imputer,
instead of sklearn.impute as in newer versions of sklearn.
'''

# %% [markdown]
# ### Step 0: Load the Data
#
# There are four files associated with this project (not including this one):
#
# - `Udacity_AZDIAS_Subset.csv`: Demographics data for the general population of Germany; 891211 persons (rows) x 85 features (columns).
# - `Udacity_CUSTOMERS_Subset.csv`: Demographics data for customers of a mail-order company; 191652 persons (rows) x 85 features (columns).
# - `Data_Dictionary.md`: Detailed information file about the features in the provided datasets.
# - `AZDIAS_Feature_Summary.csv`: Summary of feature attributes for demographics data; 85 features (rows) x 4 columns
#
# Each row of the demographics files represents a single person, but also includes information outside of individuals, including information about their household, building, and neighborhood. You will use this information to cluster the general population into groups with similar demographic properties. Then, you will see how the people in the customers dataset fit into those created clusters. The hope here is that certain clusters are over-represented in the customers data, as compared to the general population; those over-represented clusters will be assumed to be part of the core userbase. This information can then be used for further applications, such as targeting for a marketing campaign.
#
# To start off with, load in the demographics data for the general population into a pandas DataFrame, and do the same for the feature attributes summary. Note for all of the `.csv` data files in this project: they're semicolon (`;`) delimited, so you'll need an additional argument in your [`read_csv()`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html) call to read in the data properly. Also, considering the size of the main dataset, it may take some time for it to load completely.
#
# Once the dataset is loaded, it's recommended that you take a little bit of time just browsing the general structure of the dataset and feature summary file. You'll be getting deep into the innards of the cleaning in the first major step of the project, so gaining some general familiarity can help you get your bearings.

# %%
# Load in the general demographics data.
azdias = pd.read_csv('Udacity_AZDIAS_Subset.csv', sep=';')

# Load in the feature summary file.
feat_info = pd.read_csv('AZDIAS_Feature_Summary.csv', sep=';')

# %%
# Check the structure of the data after it's loaded (e.g. print the number of
# rows and columns, print the first few rows).

print("Shape of azdias: ", azdias.shape)
print("Shape of feat_info: ", feat_info.shape)

# %% [markdown]
# The shapes are as explained in the description above, so that gives confidence we imported the data correctly. Let's have a quick look anyway:

# %%
azdias.head()

# %%
feat_info.head(5)

# %%
feat_info.groupby('type')['attribute'].count()

# %%
feat_info.groupby('information_level')['attribute'].count()

# %% [markdown]
# > **Tip**: Add additional cells to keep everything in reasonably-sized chunks! Keyboard shortcut `esc --> a` (press escape to enter command mode, then press the 'A' key) adds a new cell before the active cell, and `esc --> b` adds a new cell after the active cell. If you need to convert an active cell to a markdown cell, use `esc --> m` and to convert to a code cell, use `esc --> y`. 
#
# ## Step 1: Preprocessing
#
# ### Step 1.1: Assess Missing Data
#
# The feature summary file contains a summary of properties for each demographics data column. You will use this file to help you make cleaning decisions during this stage of the project. First of all, you should assess the demographics data in terms of missing data. Pay attention to the following points as you perform your analysis, and take notes on what you observe. Make sure that you fill in the **Discussion** cell with your findings and decisions at the end of each step that has one!
#
# #### Step 1.1.1: Convert Missing Value Codes to NaNs
# The fourth column of the feature attributes summary (loaded in above as `feat_info`) documents the codes from the data dictionary that indicate missing or unknown data. While the file encodes this as a list (e.g. `[-1,0]`), this will get read in as a string object. You'll need to do a little bit of parsing to make use of it to identify and clean the data. Convert data that matches a 'missing' or 'unknown' value code into a numpy NaN value. You might want to see how much data takes on a 'missing' or 'unknown' code, and how much data is naturally missing, as a point of interest.
#
# **As one more reminder, you are encouraged to add additional cells to break up your analysis into manageable chunks.**

# %%
# Most missing_or_unknown values could be easily parsed with json.loads, but
# a few entries contains non-JSON strings:

feat_info[feat_info.missing_or_unknown.str.contains('X')]

# %% [markdown]
# Let's verify the DTypes for the AZDIAS data set:

# %%
print(azdias.dtypes.unique())
print(azdias.dtypes[azdias.dtypes == 'object'])

# %% [markdown]
# Check those columns in detail:

# %%
print(azdias['CAMEO_DEUG_2015'].unique())
print(azdias['CAMEO_DEU_2015'].unique())
print(azdias['CAMEO_INTL_2015'].unique())
print(azdias['OST_WEST_KZ'].unique())

# %% [markdown]
# In conclusion, we have 4 columns that are _not_ of numeric type. Apparently, they would be strings if it were not for the NaN values.
#
# Speaking of NaN values, what happens if we were to drop rows with missing values?

# %%
count_after_dropping_nas = len(azdias.dropna())
print(count_after_dropping_nas)

# %% [markdown]
# Without even converting the missing value codes, this would dramatically reduce the size of the dataset, and thereby preclude us from valuable data:

# %%
print("Percentage left after dropping all missing values "
      "(before converting the missing value codes): {:.1%}"
      .format(count_after_dropping_nas/len(azdias)))

# %% [markdown]
# Back to the value conversion: We can convert the `missing_or_unknown` string from `feat_info` based on the `dtype` of the corresponding `azdias` column: If the column type is numeric, then the `missing_or_unknown` string must by definition contain only numbers. If it is of type `object`, we can split the string manually.
#
# Note that this works just as well for `OST_WEST_KZ`: The relevant missing value is declared as `-1`, but it is in fact not even present in the dataset (So we don't have to worry about whether the `-1` is stored as a string or a number in the `azdias['OST_WEST_KZ']` column - if storing such mixed data types is possible):

# %%
print(feat_info[feat_info['attribute'] == 'OST_WEST_KZ'])
print(azdias['OST_WEST_KZ'].unique())


# %% [markdown]
# Or we just attempt to use `json.loads`, and we fall back to a manual parse when that fails:

# %%
def parse_missing_or_unknown(str):
    try:
        return json.loads(str)
    except ValueError:
        return str[1:-1].split(',')

# Test the function:
print(parse_missing_or_unknown('[-1,0]') == [-1, 0])
print(parse_missing_or_unknown('[-1,X]') == ['-1', 'X'])
print(parse_missing_or_unknown('[-1,X,XX]') == ['-1', 'X', 'XX'])


# %%
# Identify missing or unknown data values and convert them to NaNs.
def convert_to_nans(df):
    # I am not too happy that I have not yet found a way to vectorize
    # this call, instead having to iterate over the feat_info rows. But
    # it is not overly slow, so acceptable for now. Investigate further
    # at the 2nd answer to
    # https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas
    for index, row in feat_info.iterrows():
        # print("Converting ", row['attribute'], '...', end = "")
        missing_or_unknown = parse_missing_or_unknown(row['missing_or_unknown'])
        # print(row['missing_or_unknown'])

        df[row['attribute']].where(~df[row['attribute']].isin(missing_or_unknown), inplace=True)

convert_to_nans(azdias)

# %%
count_after_dropping_nas = len(azdias.dropna())
print(count_after_dropping_nas)

# %% [markdown]
# Once we have converted the missing or unknown values to NaNs, we can see that there are only 143 fully complete columns, so we have nothing left to work with, certainly nothing representative.

# %% [markdown]
# #### Step 1.1.2: Assess Missing Data in Each Column
#
# How much missing data is present in each column? There are a few columns that are outliers in terms of the proportion of values that are missing. You will want to use matplotlib's [`hist()`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.hist.html) function to visualize the distribution of missing value counts to find these columns. Identify and document these columns. While some of these columns might have justifications for keeping or re-encoding the data, for this project you should just remove them from the dataframe. (Feel free to make remarks about these outlier columns in the discussion, however!)
#
# For the remaining features, are there any patterns in which columns have, or share, missing data?

# %%
# Perform an assessment of how much missing data there is in each column of the
# dataset.
nan_distribution = azdias.isnull().sum()
nan_distribution.sort_values(ascending=False, inplace=True)
nan_distribution.describe()

# %% [markdown]
# If we define an outlier as being one-and-a-half times the interquartile range above the 75% quartile, it would be those columns that have more than 116515 + 1.5 * (116515 - 0) = 2.5*116515 = 291288 NaN values.

# %%
outlier_columns = nan_distribution.where(nan_distribution > 291288).dropna().index.get_values()
print(outlier_columns)

# %%
print(nan_distribution.head(10))

# %% [markdown]
# The first six columns shown above match our outlier definition. The first one even has almost exclusively NaN. Let's see the proportions:

# %%
print((nan_distribution/len(azdias)*100).head(10))

# %% [markdown]
# I tried using a histogram, but it doesn't seem to be too helpful. A bar chart seems to be more intuitive:

# %%
nan_distribution.sort_values().plot.barh(figsize=(7,22))


# %% [markdown]
# > **Investigate patterns in the amount of missing data in each column.**
#
# I am not sure I understand this question. Looking at the chart, a lot of proportions are nearly equal. To me this begs the question whether some columns are correlated in regard to their missing values. For example, if the INNENSTADT value is missing, would this usually imply that, say BALLRAUM is also missing? To find this out, however, we have to look at the rows, and we are not there yet.

# %%
# Remove the outlier columns from the dataset. (You'll perform other data
# engineering tasks such as re-encoding and imputation later.)

def drop_outlier_columns(df):
    df.drop(outlier_columns, axis='columns', inplace=True)

drop_outlier_columns(azdias)

# %%
# Drop the corresponding feature info so we can easily access all categorical values in
# step 1.2.1 Re-Encode Categorical Features
feat_info = feat_info[~feat_info.attribute.isin(outlier_columns)]

# %%
print(azdias.shape)
print(feat_info.shape)

# %%
nan_dist_without_outliers = azdias.isnull().sum()
nan_dist_without_outliers.sort_values(ascending=False, inplace=True)
print(nan_dist_without_outliers.head(10))

# %%
nan_dist_without_outliers.sort_values().plot.barh(figsize=(7,22))

# %% [markdown]
# #### Discussion 1.1.2: Assess Missing Data in Each Column
#
# The outlier columns that we removed are
#
# * TITEL_KZ
# * AGER_TYP
# * KK_KUNDENTYP
# * KBA05_BAUMAX
# * GEBURTSJAHR
# * ALTER_HH

# %% [markdown]
# Judging from the bar chart and the `head` output above, it looks like the missing values come in groups. For example, the top columns `KKK` and `REGIOTYP` have the exact same number of missing values, as have the six columns that follow (namely, `MOBI_REGIO` and the `KBA05_*` columns). Perhaps, therefore, those missing values affect the same rows.

# %% [markdown]
# #### Step 1.1.3: Assess Missing Data in Each Row
#
# Now, you'll perform a similar assessment for the rows of the dataset. How much data is missing in each row? As with the columns, you should see some groups of points that have a very different numbers of missing values. Divide the data into two subsets: one for data points that are above some threshold for missing values, and a second subset for points below that threshold.
#
# In order to know what to do with the outlier rows, we should see if the distribution of data values on columns that are not missing data (or are missing very little data) are similar or different between the two groups. Select at least five of these columns and compare the distribution of values.
# - You can use seaborn's [`countplot()`](https://seaborn.pydata.org/generated/seaborn.countplot.html) function to create a bar chart of code frequencies and matplotlib's [`subplot()`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplot.html) function to put bar charts for the two subplots side by side.
# - To reduce repeated code, you might want to write a function that can perform this comparison, taking as one of its arguments a column to be compared.
#
# Depending on what you observe in your comparison, this will have implications on how you approach your conclusions later in the analysis. If the distributions of non-missing features look similar between the data with many missing values and the data with few or no missing values, then we could argue that simply dropping those points from the analysis won't present a major issue. On the other hand, if the data with many missing values looks very different from the data with few or no missing values, then we should make a note on those data as special. We'll revisit these data later on. **Either way, you should continue your analysis for now using just the subset of the data with few or no missing values.**

# %%
# How much data is missing in each row of the dataset?

nan_by_row = azdias.isnull().sum(axis='columns')
nan_by_row.head()

# %%
nan_by_row.describe()

# %%
nan_by_row.hist(bins=80)

# %% [markdown]
# The vast majority of rows (> 600000) seems complete. As for the outliers, we notice there are comparatively many rows that miss 40 to 50 values. (There are a lot that miss at least one attribute, to be sure, as well.)

# %%
nan_by_row.hist(bins=80, log=True)

# %% [markdown]
# The logscale reveals that while the majority is complete, there are still thousands of rows that miss one or more values. I am unsure whether it makes more sense to get rid of all rows with more than 30 missing values or more than 40. Or of all rows that miss at least 1 value?
#
# The standard outlier definition of 1.5 times the IQR above the 3rd quartile would imply that any row missing seven or more attributes (1.5*3 + 3 = 7.5) is classified as an outlier. How much rows would that be?

# %%
num_complete_rows = len(nan_by_row[nan_by_row == 0])
print("# of complete rows: {} ({:.1%})"
      .format(num_complete_rows, num_complete_rows/len(nan_by_row)))

max_attributes = [0, 6, 7, 8, 9, 10, 20, 30, 40]
for max_attr in max_attributes:
    rows_with_too_many_missing_attributes = len(nan_by_row[nan_by_row > max_attr])
    percentage = rows_with_too_many_missing_attributes/len(nan_by_row)
    print("# of rows with >{} missing attributes: {} ({:.1%})"
          .format(max_attr, rows_with_too_many_missing_attributes, percentage))

# %% [markdown]
# We can see that we don't gain much by allowing rows with a lot of missing attributes. For example, it makes virtually no difference to limit ourselves to rows that have more than 20 or more than 30 missing attributes. In other words, rows that have more than 20 missing attributes will with high probability have in fact more than 30 missing attributes.
#
# We lose another 2 percentage points by tightening the criterion to only include rows with at most 10 missing attributes, so that seems a good cut-off:

# %%
MAX_NAN_VALUES_PER_ROW = 10


# %%
# Write code to divide the data into two subsets based on the number of missing
# values in each row.
def split_df_by_max_nan(df, MAX_NAN_VALUES_PER_ROW):
    nan_by_row = df.isnull().sum(axis='columns')
    return df[nan_by_row <= MAX_NAN_VALUES_PER_ROW], df[nan_by_row > MAX_NAN_VALUES_PER_ROW]

azdias_few_missing, azdias_many_missing = split_df_by_max_nan(azdias, MAX_NAN_VALUES_PER_ROW)

# %%
print(len(azdias_few_missing), len(azdias_many_missing), len(azdias_few_missing)+len(azdias_many_missing))


# %%
# Compare the distribution of values for at least five columns where there are
# no or few missing values, between the two subsets.

def compare_subsets(col_name):
    plt.figure(figsize=(12, 4))
    plt.subplot(1,2,1)
    plt.title('Few missing values')
    sns.countplot(x=col_name, data=azdias_few_missing)
    plt.subplot(1,2,2)
    plt.title('Many missing values')
    sns.countplot(x=col_name, data=azdias_many_missing)
    plt.show()

# The following columns are chosen for no particular reason:
for col in ['GEBAEUDETYP','FINANZ_SPARER', 'NATIONALITAET_KZ', 'SEMIO_SOZ', 'ALTERSKATEGORIE_GROB']:
    compare_subsets(col)

# %% [markdown]
# #### Discussion 1.1.3: Assess Missing Data in Each Row
#
# The comparison reveals major differences. For the chosen columns, `NATIONALITAET_KZ` is the one that looks essentially identical, and `ALTERSKATEGORIE_GROB` and `GEBAEUDETYP` might be very loosely seen as somewhat similar. But `FINANZ_SPARER`, `SEMIO_SOZ` compared across the datasets show no similarity whatsoever.

# %% [markdown]
# Continue with the data that misses only few values:

# %%
azdias = azdias_few_missing

# %% [markdown]
# ### Step 1.2: Select and Re-Encode Features
#
# Checking for missing data isn't the only way in which you can prepare a dataset for analysis. Since the unsupervised learning techniques to be used will only work on data that is encoded numerically, you need to make a few encoding changes or additional assumptions to be able to make progress. In addition, while almost all of the values in the dataset are encoded using numbers, not all of them represent numeric values. Check the third column of the feature summary (`feat_info`) for a summary of types of measurement.
# - For numeric and interval data, these features can be kept without changes.
# - Most of the variables in the dataset are ordinal in nature. While ordinal values may technically be non-linear in spacing, make the simplifying assumption that the ordinal variables can be treated as being interval in nature (that is, kept without any changes).
# - Special handling may be necessary for the remaining two variable types: categorical, and 'mixed'.
#
# In the first two parts of this sub-step, you will perform an investigation of the categorical and mixed-type features and make a decision on each of them, whether you will keep, drop, or re-encode each. Then, in the last part, you will create a new data frame with only the selected and engineered columns.
#
# Data wrangling is often the trickiest part of the data analysis process, and there's a lot of it to be done here. But stick with it: once you're done with this step, you'll be ready to get to the machine learning parts of the project!

# %%
# How many features are there of each data type?

feat_info.groupby('type').count()['attribute']

# %% [markdown]
# #### Step 1.2.1: Re-Encode Categorical Features
#
# For categorical data, you would ordinarily need to encode the levels as dummy variables. Depending on the number of categories, perform one of the following:
# - For binary (two-level) categoricals that take numeric values, you can keep them without needing to do anything.
# - There is one binary variable that takes on non-numeric values. For this one, you need to re-encode the values as numbers or create a dummy variable.
# - For multi-level categoricals (three or more values), you can choose to encode the values using multiple dummy variables (e.g. via [OneHotEncoder](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)), or (to keep things straightforward) just drop them from the analysis. As always, document your choices in the Discussion section.

# %%
# Assess categorical variables: which are binary, which are multi-level, and
# which one needs to be re-encoded?

categorical_cols = feat_info[feat_info.type =='categorical']['attribute']

binary_data = []
binary_nonnumeric_data = []
multilevel_data = []

for col in categorical_cols:
    unique_values = azdias[col].unique()
    if len(unique_values) == 2:
        try:
            pd.to_numeric(unique_values)
            binary_data.append(col)
        except ValueError:
            binary_nonnumeric_data.append((col, unique_values))
    elif len(unique_values) >= 3:
        multilevel_data.append((col, unique_values))
    else:
        print(col, "Say what?")
        
print("Binary data:\n ", '\n  '.join(binary_data))
print("\nBinary non-numeric data:\n ", "\n  ".join(["{} ({})".format(attr[0], attr[1]) for attr in binary_nonnumeric_data]))
print("\nMulti-level data:\n ", "\n  ".join(["{} ({} values)".format(attr[0], len(attr[1])) for attr in multilevel_data]))


# %%
# Re-encode categorical variable(s) to be kept in the analysis.

def reencode_ost_west_kz(df):
    df['OST_WEST_KZ'] = df['OST_WEST_KZ'].astype('category').cat.codes

reencode_ost_west_kz(azdias)

# %%
azdias.drop('NATIONALITAET_KZ', axis='columns', inplace=True)

# %%
multilevel_columns = [attr[0] for attr in multilevel_data if attr[0] != 'NATIONALITAET_KZ']


# %%
def one_hot_encode_multilevel_columns(df):
    return pd.get_dummies(df,columns=multilevel_columns)


# %%
azdias = one_hot_encode_multilevel_columns(azdias)

# %%
# GEBAEUDETYP_5.0 needs to be dropped because later, it is not present in the customer data
# after calling clean_data.
azdias.drop('GEBAEUDETYP_5.0', axis='columns', inplace=True)

# %%
azdias.head()

# %% [markdown]
# #### Discussion 1.2.1: Re-Encode Categorical Features
#
# `OST_WEST_KZ` could be changed to a numeric binary categorical variable with `cat.codes`.
#
# From a brief look at the data dictionary, the multi-level values seem altogether rather informative, so I didn't want to remove many of those.
#
# I have dropped `NATIONALITAET_KZ` since it seems overly simplified. According to the data dictionary, nationality is defined based on name analysis: 'German-sounding' or 'foreign-sounding'. Not being a marketing guy but a German, I wouldn't want to draw inferences based on that. I find it hard to remain non-judgemental on that kind of data.
#
# If one looks closely at the multi-level data, note that one of the levels in quite a few variables is `nan`. In particular, `VERS_TYPE`, consisting of 3 levels, has `1.`, `2.`, and `nan` as possible values (the dot signifies those are floating-point numbers). If we truncated the rows more aggressively, this variable might have become a binary category variable.
#
# I am not sure whether it is helpful to keep those `nan` values, but as we saw above, straight removing them would essentially wipe out the dataset, so let's see what happens if we keep them. 

# %% [markdown]
# #### Step 1.2.2: Engineer Mixed-Type Features
#
# There are a handful of features that are marked as "mixed" in the feature summary that require special treatment in order to be included in the analysis. There are two in particular that deserve attention; the handling of the rest are up to your own choices:
# - "PRAEGENDE_JUGENDJAHRE" combines information on three dimensions: generation by decade, movement (mainstream vs. avantgarde), and nation (east vs. west). While there aren't enough levels to disentangle east from west, you should create two new variables to capture the other two dimensions: an interval-type variable for decade, and a binary variable for movement.
# - "CAMEO_INTL_2015" combines information on two axes: wealth and life stage. Break up the two-digit codes by their 'tens'-place and 'ones'-place digits into two new ordinal variables (which, for the purposes of this project, is equivalent to just treating them as their raw numeric values).
# - If you decide to keep or engineer new features around the other mixed-type features, make sure you note your steps in the Discussion section.
#
# Be sure to check `Data_Dictionary.md` for the details needed to finish these tasks.

# %% [markdown]
# **1.2.2.1 Investigate "PRAEGENDE_JUGENDJAHRE" and engineer two new variables.**

# %%
azdias['PRAEGENDE_JUGENDJAHRE'].unique()


# %% [markdown]
# The data dictionary reveals how those values can be categorized into mainstream and avantgarde.
#
# We will create a binary variable `MOVEMENT` with `0` representing "Mainstream" and `1` for "Avantgarde". Note there's a `nan`, so some entries may be `nan`, too.

# %%
def engineer_movement(df):
    mainstream = [1., 3., 5., 8., 10., 12., 14.]
    avantgarde = [2., 4., 6., 7., 9., 11., 13., 15.]
    df.loc[ df['PRAEGENDE_JUGENDJAHRE'].isin(mainstream), 'MOVEMENT' ] = 0
    df.loc[ df['PRAEGENDE_JUGENDJAHRE'].isin(avantgarde), 'MOVEMENT' ] = 1


# %%
engineer_movement(azdias)
azdias[['PRAEGENDE_JUGENDJAHRE','MOVEMENT',]].head(10)


# %% [markdown]
# Now for the decade, I first used a Pandas Interval, but that won't work with the Imputer.
#
# The interval-type variable will contain `40` to represent the 40s, `50` for the 50s up to `90` for the 90s. We will create a mapper mapping from an individual value between 1 and 14 to that number, then use df.map() to create the `DECADE` column:

# %%
def create_praegende_jugendjahre_to_decade_mapper():
    decades = [
        (40, [1, 2]),
        (50, [3, 4]),
        (60, [5, 6, 7]),
        (70, [8, 9]),
        (80, [10, 11, 12, 13]),
        (90, [14, 15])
    ]
    original_value_to_decade_mapper = {original_value: decade for decade, original_values in decades for original_value in original_values}
    original_value_to_decade_mapper[float("nan")] = float("nan")
    return original_value_to_decade_mapper


# %%
create_praegende_jugendjahre_to_decade_mapper()


# %%
def engineer_decade(df):
    df['DECADE'] = df['PRAEGENDE_JUGENDJAHRE'].map(create_praegende_jugendjahre_to_decade_mapper())

engineer_decade(azdias)

# %%
azdias[['PRAEGENDE_JUGENDJAHRE','DECADE',]].head(10)

# %%
azdias.drop(columns=['PRAEGENDE_JUGENDJAHRE'], axis='columns', inplace=True)


# %% [markdown]
# For reuse (in particular, the cleanup function further down) it will be more convenient to group those things together:

# %%
def reengineer_praegende_jugendjahre(df):
    engineer_movement(df)
    engineer_decade(df)
    df.drop(columns=['PRAEGENDE_JUGENDJAHRE'], axis='columns', inplace=True)


# %% [markdown]
# **1.2.2.2 Investigate "CAMEO_INTL_2015" and engineer two new variables.**

# %%
azdias['CAMEO_INTL_2015'].unique()


# %%
def engineer_wealth(df):
    df['WEALTH'] = df['CAMEO_INTL_2015'].astype(float) // 10

def engineer_life_stage(df):
    df['LIFE_STAGE'] = df['CAMEO_INTL_2015'].astype(float) % 10

engineer_wealth(azdias)
engineer_life_stage(azdias)

# %%
azdias[['CAMEO_INTL_2015','WEALTH','LIFE_STAGE']].head()

# %%
azdias.drop(['CAMEO_INTL_2015'], axis='columns', inplace=True)


# %%
def reengineer_cameo_intl_2015(df):
    engineer_wealth(df)
    engineer_life_stage(df)
    df.drop(['CAMEO_INTL_2015'], axis='columns', inplace=True)


# %% [markdown]
# **1.2.2.3 Investigate the other mixed-type features**

# %%
feat_info[feat_info.type == 'mixed']

# %%
azdias['WOHNLAGE'].hist()


# %%
def engineer_rural_neighborhood(df):
    df['RURAL_NEIGHBORHOOD'] = df['WOHNLAGE'].isin([7,8])


# %%
def engineer_neighborhood_quality(df):
    df['NEIGHBORHOOD_QUALITY'] = df['WOHNLAGE']
    df[df['NEIGHBORHOOD_QUALITY'].isin([-1,0,7,8])] = float('nan')


# %%
def reengineer_wohnlage(df):
    engineer_rural_neighborhood(df)
    engineer_neighborhood_quality(df)
    df.drop(['WOHNLAGE'], axis='columns', inplace=True)


# %%
def reengineer_other_mixed_attributes(df):
    reengineer_wohnlage(df)
    df.drop(['LP_LEBENSPHASE_FEIN', 'LP_LEBENSPHASE_GROB','PLZ8_BAUMAX'], axis='columns', inplace=True)


# %%
reengineer_other_mixed_attributes(azdias)

# %% [markdown]
# #### Discussion 1.2.2: Engineer Mixed-Type Features
#
# Regarding `PRAEGENDE_JUGENDJAHRE` and `CAMEO_INTL_2015`, the conversion itself should be self-explanatory. For the others, I dropped
# * PLZ8_BAUMAX: is only relevant for the PLZ8 region
# * LP_LEBENSPHASE_FEIN and LP_LEBENSPHASE_GROB: they are in the same spirit as `CAMEO_INTL_2015`, but that one should be enough.
#
# `WOHNLAGE` (neighborhood quality) is an interesting one. I split off `RURAL_NEIGHBORHOOD` as a boolean flag for values 7 and 8, and created another column `NEIGHBORHOOD_QUALITY` to keep the values 1 to 5 (and for all other values, `NEIGHBORHOOD_QUALITY` defaults to `nan`). I then dropped `WOHNLAGE` as well.

# %% [markdown]
# #### Step 1.2.3: Complete Feature Selection
#
# In order to finish this step up, you need to make sure that your data frame now only has the columns that you want to keep. To summarize, the dataframe should consist of the following:
# - All numeric, interval, and ordinal type columns from the original dataset.
# - Binary categorical features (all numerically-encoded).
# - Engineered features from other multi-level categorical features and mixed features.
#
# Make sure that for any new columns that you have engineered, that you've excluded the original columns from the final dataset. Otherwise, their values will interfere with the analysis later on the project. For example, you should not keep "PRAEGENDE_JUGENDJAHRE", since its values won't be useful for the algorithm: only the values derived from it in the engineered features you created should be retained. As a reminder, your data should only be from **the subset with few or no missing values**.

# %%
# If there are other re-engineering tasks you need to perform, make sure you
# take care of them here. (Dealing with missing data will come in step 2.1.)

# %%
# Do whatever you need to in order to ensure that the dataframe only contains
# the columns that should be passed to the algorithm functions.

feat_info.groupby('type').count()['attribute']

# %%
cat_cols_in_azdias = list(set(feat_info[feat_info.type == 'categorical']['attribute']) & set(azdias.columns))
for cat_col in cat_cols_in_azdias:
    if azdias[cat_col].dtype != 'float64':
        print(cat_col, 'is categorical and has not been converted to numerical values')


# %% [markdown]
# ### Step 1.3: Create a Cleaning Function
#
# Even though you've finished cleaning up the general population demographics data, it's important to look ahead to the future and realize that you'll need to perform the same cleaning steps on the customer demographics data. In this substep, complete the function below to execute the main feature selection, encoding, and re-engineering steps you performed above. Then, when it comes to looking at the customer data in Step 3, you can just run this function on that DataFrame to get the trimmed dataset in a single step.

# %%
def clean_data(df):
    """
    Perform feature trimming, re-encoding, and engineering for demographics
    data
        
    INPUT: Demographics DataFrame
    OUTPUT: Trimmed and cleaned demographics DataFrame
    """
    
    df_copy = df.copy()
    
    # Put in code here to execute all main cleaning steps:
    # convert missing value codes into NaNs, ...
    convert_to_nans(df_copy)
    
    # remove selected columns and rows, ...
    drop_outlier_columns(df_copy)

    df_few_missing, df_many_missing = split_df_by_max_nan(df_copy, MAX_NAN_VALUES_PER_ROW)
    df_copy = df_few_missing

    # select, re-encode, and engineer column values.
    reencode_ost_west_kz(df_copy)
    df_copy.drop('NATIONALITAET_KZ', axis='columns', inplace=True)
    df_copy = one_hot_encode_multilevel_columns(df_copy)
   
    reengineer_praegende_jugendjahre(df_copy)
    reengineer_cameo_intl_2015(df_copy)
    reengineer_other_mixed_attributes(df_copy)

    # Return the cleaned dataframe and the discarded data
    return df_copy, df_many_missing


# %% [markdown]
# ## Step 2: Feature Transformation
#
# ### Step 2.1: Apply Feature Scaling
#
# Before we apply dimensionality reduction techniques to the data, we need to perform feature scaling so that the principal component vectors are not influenced by the natural differences in scale for features. Starting from this part of the project, you'll want to keep an eye on the [API reference page for sklearn](http://scikit-learn.org/stable/modules/classes.html) to help you navigate to all of the classes and functions that you'll need. In this substep, you'll need to check the following:
#
# - sklearn requires that data not have missing values in order for its estimators to work properly. So, before applying the scaler to your data, make sure that you've cleaned the DataFrame of the remaining missing values. This can be as simple as just removing all data points with missing data, or applying an [Imputer](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Imputer.html) to replace all missing values. You might also try a more complicated procedure where you temporarily remove missing values in order to compute the scaling parameters before re-introducing those missing values and applying imputation. Think about how much missing data you have and what possible effects each approach might have on your analysis, and justify your decision in the discussion section below.
# - For the actual scaling function, a [StandardScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) instance is suggested, scaling each feature to mean 0 and standard deviation 1.
# - For these classes, you can make use of the `.fit_transform()` method to both fit a procedure to the data as well as apply the transformation to the data at the same time. Don't forget to keep the fit sklearn objects handy, since you'll be applying them to the customer demographics data towards the end of the project.

# %%
# If you've not yet cleaned the dataset of all NaN values, then investigate and
# do that now.

nan_distribution = azdias.isnull().sum()
nan_distribution.sort_values(ascending=False, inplace=True)
nan_distribution.describe()

# %%
azdias.shape

# %%
# I use, for no particular reason, the most_frequent_strategy.
# I also tried mean and median but the PCA results are the same.
imp = Imputer(missing_values=np.nan, strategy='most_frequent')
azdias_imputed = imp.fit_transform(azdias)

# Re-create dataframe
azdias_imputed = pd.DataFrame(azdias_imputed)
azdias_imputed.columns = azdias.columns
azdias_imputed.index = azdias.index

# %%
# Apply feature scaling to the general population demographics data.

scaler = StandardScaler()
azdias_scaled = scaler.fit_transform(azdias_imputed)
print(azdias_scaled.shape)

# %% [markdown]
# ### Discussion 2.1: Apply Feature Scaling
#
# The investigation of nan values shows a large percentage of values across all columns is still not initialized. At minimum, a column is missing around 20% of data (= 185102/891221).
#
# Removing all rows with nan values could leave us with too little data or skew the distributions. In general, imputation is a complicated topic, of which I know too little at the moment. To advance this project, I will apply a simple mean imputation for now, but from a quick Google search (see references), there's a lot more to learn.
#
# * https://www.theanalysisfactor.com/seven-ways-to-make-up-data-common-methods-to-imputing-missing-data/
# * https://www.theanalysisfactor.com/missing-data-two-recommended-solutions/
# * https://github.com/kearnz/autoimpute

# %% [markdown]
# ### Step 2.2: Perform Dimensionality Reduction
#
# On your scaled data, you are now ready to apply dimensionality reduction techniques.
#
# - Use sklearn's [PCA](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) class to apply principal component analysis on the data, thus finding the vectors of maximal variance in the data. To start, you should not set any parameters (so all components are computed) or set a number of components that is at least half the number of features (so there's enough features to see the general trend in variability).
# - Check out the ratio of variance explained by each principal component as well as the cumulative variance explained. Try plotting the cumulative or sequential values using matplotlib's [`plot()`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html) function. Based on what you find, select a value for the number of transformed features you'll retain for the clustering part of the project.
# - Once you've made a choice for the number of components to keep, make sure you re-fit a PCA instance to perform the decided-on transformation.

# %%
# Apply PCA to the data.
pca = PCA(random_state = 123)
azdias_pca = pca.fit_transform(azdias_scaled)

# Re-create dataframe
azdias_scaled = pd.DataFrame(azdias_scaled)
azdias_scaled.columns = azdias.columns
azdias_scaled.index = azdias.index


# %%
# Investigate the variance accounted for by each principal component.

# scree_plot is part of the helper functions of this Udacity course,
# I am reusing it here.
def scree_plot(pca):
    '''
    Creates a scree plot associated with the principal components 
    
    INPUT: pca - the result of instantian of PCA in scikit learn
            
    OUTPUT:
            None
    '''
    num_components=len(pca.explained_variance_ratio_)
    ind = np.arange(num_components)
    vals = pca.explained_variance_ratio_
 
    plt.figure(figsize=(10, 6))
    ax = plt.subplot(111)
    cumvals = np.cumsum(vals)
    ax.bar(ind, vals)
    ax.plot(ind, cumvals)
    #for i in range(num_components):
    #    ax.annotate(r"%s%%" % ((str(vals[i]*100)[:4])), (ind[i]+0.2, vals[i]), va="bottom", ha="center", fontsize=12)
 
    ax.xaxis.set_tick_params(width=0)
    ax.yaxis.set_tick_params(width=2, length=12)
 
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    plt.title('Explained Variance Per Principal Component')
    

scree_plot(pca)

# %% [markdown]
# The graph shows that the first 5 or so values account for more than 20% of the variance explained. The accumulated curve continuous to flatten afterwards. Even around 100 components account only for approximately 90% of the variance explained. Let's try to find the number of components necessary to reach those 90% of variance explained:

# %%
# Re-apply PCA to the data while selecting for number of components to retain.

# Initially, I started with a lower value (90 from looking at the graph), but
# to accelerate future runs, I adjusted the value to be near the expected result:
n_comps = 103 
explained_variance = 0
while explained_variance <0.9:
    pca = PCA(n_components=n_comps, random_state = 123)
    pca = pca.fit(azdias_scaled)
    explained_variance = pca.explained_variance_ratio_.sum()
    print(n_comps, explained_variance)
    n_comps += 1
num_comps = n_comps-1

# %%
azdias_pca = pca.transform(azdias_scaled)
azdias_pca.shape


# %% [markdown]
# ### Discussion 2.2: Perform Dimensionality Reduction
#
# As mentioned in the graph discussion (below the graph), the idea is to have as many components as are necessary to account for 90% of the variance explained. It turned out that 103 of the 192 components are necessary for this.

# %% [markdown]
# ### Step 2.3: Interpret Principal Components
#
# Now that we have our transformed principal components, it's a nice idea to check out the weight of each variable on the first few components to see if they can be interpreted in some fashion.
#
# As a reminder, each principal component is a unit vector that points in the direction of highest variance (after accounting for the variance captured by earlier principal components). The further a weight is from zero, the more the principal component is in the direction of the corresponding feature. If two features have large weights of the same sign (both positive or both negative), then increases in one tend expect to be associated with increases in the other. To contrast, features with different signs can be expected to show a negative correlation: increases in one variable should result in a decrease in the other.
#
# - To investigate the features, you should map each weight to their corresponding feature name, then sort the features according to weight. The most interesting features for each principal component, then, will be those at the beginning and end of the sorted list. Use the data dictionary document to help you understand these most prominent features, their relationships, and what a positive or negative value on the principal component might indicate.
# - You should investigate and interpret feature associations from the first three principal components in this substep. To help facilitate this, you should write a function that you can call at any time to print the sorted list of feature weights, for the *i*-th principal component. This might come in handy in the next step of the project, when you interpret the tendencies of the discovered clusters.

# %%
# Map weights for the first principal component to corresponding feature names
# and then print the linked values, sorted by weight.
# HINT: Try defining a function here or in a new cell that you can reuse in the
# other cells.

def print_pca_component(pca_component, top_n = None, ascending = False):
    """
    For the given PCA model, print the top n features by weight
    for the component with index component_index.
    
    If n == none, print all weights.
    If ascending = True, print the largest _negative_ values. 
    """
    
    pca_series = pd.Series(pca_component, index = azdias_imputed.keys())

    weights_sorted = pca_series.sort_values(ascending=ascending)
    if top_n is None:
        print(weights_sorted)
    else:
        print(weights_sorted[:top_n])


# %% [markdown]
# For the interpretation, we will look at the top 5 positive and negative attributes of each component:

# %%
print_pca_component(pca.components_[0], 5)
print_pca_component(pca.components_[0], 5, True)

# %%
# Map weights for the second principal component to corresponding feature names
# and then print the linked values, sorted by weight.

print_pca_component(pca.components_[1], 5)
print_pca_component(pca.components_[1], 5, True)

# %%
# Map weights for the third principal component to corresponding feature names
# and then print the linked values, sorted by weight.

print_pca_component(pca.components_[2], 5)
print_pca_component(pca.components_[2], 5, True)

# %% [markdown]
# ### Discussion 2.3: Interpret Principal Components
#
# (Double-click this cell and replace this text with your own text, reporting your observations from detailed investigation of the first few principal components generated. Can we interpret positive and negative values from them in a meaningful way?)
#
# #### First component
#
# The most relevant positively correlated features are `MOBI_REGIO` and `KBA05_ANTG1`. The higher the `MOBIO_REGIO` value, the lower the movement, and a high `KBA05_ANTG1` value signifies a high fraction of home owners - those clearly go hand in hand. Likewise, this explains all of the five negatively correlated features. The principal component identifies high proportion of home builders (i.e. a low value for `FINANZ_HAUSBAUER`), and _not_ low-income earners (i.e. an inverse correlation to the original `LP_STATUS_GROB` value being 1). Most significant is a low value of `HH_EINKOMMEN_SCORE`, that is, the household income which tends to be the highest. As expected, this is correlated with `WEALTH`, the re-engineered value from `CAMEO_INTL_2015`. A low value for `WEALTH` also represents wealthy households.
#
# In conclusion, the combination of those attributes and their values seem indicative for an upper-middle class or upper-class lifestyle.
#
# #### Second component
#
# In a nutshell, the second component correlates with a hedonistic lifestyle that is unconcerned with financial matters: The lower a person exhibits a saver, investor, or at least low-key financial mind (`FINANZ_SPARER`, `FINANZ_ANLEGER`, `FINANZ_UNAUFFAELLIGER`), the more it correlates with this component. Consequently, we find `FINANZTYP_1`, "low financial interest". We also find positive values for `SEMIO_REL`, thus a low affinity to religion, underscoring the hedonism.
#
# `ALTERSKATEGORIE_GROB` is inversely correlated, so this component identifies preferrably younger generations (cf. the third component where it's the opposite). I do not understand why `FINANZ_VORSORGER` (be preapred) should be at the opposite end to `FINANZ_SPARER` (money saver), it's hard to judge from the data dictionary alone. A high affinity to being sensual-minded (i.e. a low value for `SEMIO_LUST`) and a propensity for being a heavy shopper (low values for `RETOURTYP_BK_S`) affirms, again, the hedonistic outlook.
#  
#
# #### Third component
#
# The attributes and their correlations point to traditional minded (`SEMIO_TRADV`), rational (`SEMIO_TRADV`) people, close to urban centers (`BALLRAUM`), and low-income earners (`LP_STATUS_FEIN_1.0`), older generations (negative correlation to `DECADE`, i.e. likely their formative years were the sparse post-war years in the 1950s). It sounds like this component correlates well with working-class or lower-middle-class people.
#
#
# #### References
#
# * https://online.stat.psu.edu/stat505/lesson/11/11.4 provides an example for interpreting principal components

# %% [markdown]
# ## Step 3: Clustering
#
# ### Step 3.1: Apply Clustering to General Population
#
# You've assessed and cleaned the demographics data, then scaled and transformed them. Now, it's time to see how the data clusters in the principal components space. In this substep, you will apply k-means clustering to the dataset and use the average within-cluster distances from each point to their assigned cluster's centroid to decide on a number of clusters to keep.
#
# - Use sklearn's [KMeans](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans) class to perform k-means clustering on the PCA-transformed data.
# - Then, compute the average difference from each point to its assigned cluster's center. **Hint**: The KMeans object's `.score()` method might be useful here, but note that in sklearn, scores tend to be defined so that larger is better. Try applying it to a small, toy dataset, or use an internet search to help your understanding.
# - Perform the above two steps for a number of different cluster counts. You can then see how the average distance decreases with an increasing number of clusters. However, each additional cluster provides a smaller net benefit. Use this fact to select a final number of clusters in which to group the data. **Warning**: because of the large size of the dataset, it can take a long time for the algorithm to resolve. The more clusters to fit, the longer the algorithm will take. You should test for cluster counts through at least 10 clusters to get the full picture, but you shouldn't need to test for a number of clusters above about 30.
# - Once you've selected a final number of clusters to use, re-fit a KMeans instance to perform the clustering operation. Make sure that you also obtain the cluster assignments for the general demographics data, since you'll be using them in the final Step 3.3.

# %%
# Over a number of different cluster counts...

centers = list(range(2,20,2))

# run k-means clustering on the data and...
# compute the average within-cluster distances.

# This function is adapted from the Changing K exercise in the
# clustering lesson.
def get_kmeans_score(data, center):
    '''
    returns the kmeans score regarding SSE for points to centers
    INPUT:
        data - the dataset you want to fit kmeans to
        center - the number of centers you want (the k value)
    OUTPUT:
        score - the SSE score for the kmeans model fit to the data
    '''
    kmeans = KMeans(n_clusters=center, random_state = 123)
    model = kmeans.fit(data)
    score = np.abs(model.score(data))  
    return score

scores = []

for center in centers:
    print(datetime.now().strftime("%H:%M:%S"), "Calculating score for", center, "clusters ...", end = " ")
    time1 = time.time()
    scores.append(get_kmeans_score(azdias_pca, center))
    time2 = time.time()
    print('%s (took %0.3f s)' % (scores[-1], (time2-time1)))

# %%
# Investigate the change in within-cluster distance across number of clusters.
# HINT: Use matplotlib's plot function to visualize this relationship.

# We are plotting a scree plot as done in the Changing K exercise:
plt.plot(centers, scores, linestyle='--', marker='o', color='b');
plt.xlabel('K');
plt.ylabel('SSE');
plt.title('SSE vs. K');

# %%
# Re-fit the k-means model with the selected number of clusters and obtain
# cluster predictions for the general population demographics data.

num_clusters = 10

# %%
kmeans = KMeans(n_clusters=num_clusters, random_state = 123)
kmeans = kmeans.fit(azdias_pca)

# %%
azdias_labels = kmeans.predict(azdias_pca)

# %%
plt.hist(azdias_labels)

# %% [markdown]
# ### Discussion 3.1: Apply Clustering to General Population
#
# I initially tried 1, 2, 3, ... 14 clusters but there was still a noticeable slope at the end of the score. I then tried `range(2,40,4)` with a similar result, so I then concentrated on the first 20 and investigated just the change in slope. Anything between 5 to 10 clusters seems to be a judgment call. Having said that, the slope definitely flattens: A result from the 2nd analysis was that, whereas a decrease in SSE of 0.2 requires a cluster increase to 10 initially, it takes up to cluster 38 to reach another 0.2 SSE decrease.
#
# Because the curve's slope remains fairly constant after 10 clusters, that is the number that will be used for the remaining analysis.

# %% [markdown]
# ### Step 3.2: Apply All Steps to the Customer Data
#
# Now that you have clusters and cluster centers for the general population, it's time to see how the customer data maps on to those clusters. Take care to not confuse this for re-fitting all of the models to the customer data. Instead, you're going to use the fits from the general population to clean, transform, and cluster the customer data. In the last step of the project, you will interpret how the general population fits apply to the customer data.
#
# - Don't forget when loading in the customers data, that it is semicolon (`;`) delimited.
# - Apply the same feature wrangling, selection, and engineering steps to the customer demographics using the `clean_data()` function you created earlier. (You can assume that the customer demographics data has similar meaning behind missing data patterns as the general demographics data.)
# - Use the sklearn objects from the general demographics data, and apply their transformations to the customers data. That is, you should not be using a `.fit()` or `.fit_transform()` method to re-fit the old objects, nor should you be creating new sklearn objects! Carry the data through the feature scaling, PCA, and clustering steps, obtaining cluster assignments for all of the data in the customer demographics data.

# %%
# Load in the customer demographics data.
customers = pd.read_csv('Udacity_CUSTOMERS_Subset.csv', sep=';')
print(customers.shape)

# %%
# Apply preprocessing, feature transformation, and clustering from the general
# demographics onto the customer data, obtaining cluster predictions for the
# customer demographics data.

customers_cleaned, customers_many_missing = clean_data(customers)
print(customers.shape)

# %%
customers_imputed = imp.transform(customers_cleaned)

customers_imputed = pd.DataFrame(customers_imputed)
customers_imputed.columns = customers_cleaned.columns
customers_imputed.index = customers_cleaned.index

# %%
customers_scaled = scaler.transform(customers_imputed)
print(customers_scaled.shape)

# %%
customers_pca = pca.transform(customers_scaled)

# %%
customers_labels = kmeans.predict(customers_pca)

# %%
plt.hist(customers_labels)


# %% [markdown]
# ### Step 3.3: Compare Customer Data to Demographics Data
#
# At this point, you have clustered data based on demographics of the general population of Germany, and seen how the customer data for a mail-order sales company maps onto those demographic clusters. In this final substep, you will compare the two cluster distributions to see where the strongest customer base for the company is.
#
# Consider the proportion of persons in each cluster for the general population, and the proportions for the customers. If we think the company's customer base to be universal, then the cluster assignment proportions should be fairly similar between the two. If there are only particular segments of the population that are interested in the company's products, then we should see a mismatch from one to the other. If there is a higher proportion of persons in a cluster for the customer data compared to the general population (e.g. 5% of persons are assigned to a cluster for the general population, but 15% of the customer data is closest to that cluster's centroid) then that suggests the people in that cluster to be a target audience for the company. On the other hand, the proportion of the data in a cluster being larger in the general population than the customer data (e.g. only 2% of customers closest to a population centroid that captures 6% of the data) suggests that group of persons to be outside of the target demographics.
#
# Take a look at the following points in this step:
#
# - Compute the proportion of data points in each cluster for the general population and the customer data. Visualizations will be useful here: both for the individual dataset proportions, but also to visualize the ratios in cluster representation between groups. Seaborn's [`countplot()`](https://seaborn.pydata.org/generated/seaborn.countplot.html) or [`barplot()`](https://seaborn.pydata.org/generated/seaborn.barplot.html) function could be handy.
#   - Recall the analysis you performed in step 1.1.3 of the project, where you separated out certain data points from the dataset if they had more than a specified threshold of missing values. If you found that this group was qualitatively different from the main bulk of the data, you should treat this as an additional data cluster in this analysis. Make sure that you account for the number of data points in this subset, for both the general population and customer datasets, when making your computations!
# - Which cluster or clusters are overrepresented in the customer dataset compared to the general population? Select at least one such cluster and infer what kind of people might be represented by that cluster. Use the principal component interpretations from step 2.3 or look at additional components to help you make this inference. Alternatively, you can use the `.inverse_transform()` method of the PCA and StandardScaler objects to transform centroids back to the original data space and interpret the retrieved values directly.
# - Perform a similar investigation for the underrepresented clusters. Which cluster or clusters are underrepresented in the customer dataset compared to the general population, and what kinds of people are typified by these clusters?

# %%
# Compare the proportion of data in each cluster for the customer data to the
# proportion of data in each cluster for the general population.

def create_proportions(labels, df_many_missing):
    # Add the quantity of missing data at the beginning, as a separate cluster
    weights = np.append([len(df_many_missing)],
                       np.histogram(labels, bins=num_clusters)[0])
    print(weights.sum()) # Visual check to see whether we still have the original amount of rows
    return weights / weights.sum()

azdias_proportions = create_proportions(azdias_labels, azdias_many_missing)
customers_proportions = create_proportions(customers_labels, customers_many_missing)

proportions = pd.DataFrame({'azdias': azdias_proportions,
                            'customers': customers_proportions},
                           columns=['azdias', 'customers'], index=range(-1,10))
proportions


# %%
def create_barplot(ax, col, title, annotate = False):
    ax.set_title(title)
    ax.set(ylim=(0, 0.3), xlabel = "cluster", ylabel = "proportion")
    sns.barplot(data=proportions[[col]].transpose(), ax=ax)

figure, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,4))
create_barplot(ax1, 'azdias', 'General Population')
ax1.annotate("cluster -1 represents discarded values due to too many missing attributes",
            xy = (0,0), xytext=(70,2), xycoords='figure pixels')
create_barplot(ax2, 'customers', 'Customer Data')

# %%
# What kinds of people are part of a cluster that is overrepresented in the
# customer data compared to the general population?

investigated_cluster = 8
cluster_centroid_ = pca.inverse_transform(kmeans.cluster_centers_[investigated_cluster])

# %%
print_pca_component(cluster_centroid_, 5)
print_pca_component(cluster_centroid_, 5, True)

# %%
# What kinds of people are part of a cluster that is underrepresented in the
# customer data compared to the general population?

investigated_cluster = 9 # 0, 1
cluster_centroid_ = pca.inverse_transform(kmeans.cluster_centers_[investigated_cluster])

print_pca_component(cluster_centroid_, 5)
print_pca_component(cluster_centroid_, 5, True)

# %% [markdown]
# ### Discussion 3.3: Compare Customer Data to Demographics Data
#
# The bar charts for the cluster proportions show clear differences between the general population and the target audience. For example, while cluster 0, 1, and 9 account in total for approx. 24% of the general population, just 2% of the target audience is found here. On the other hand, cluster 8 contains approx. a tenth of the general population, but around a quarter of the customer group. Cluster 3 is noteworthy since it is, with ca. 21%, the largest cluster of the general population, with a slightly smaller value (roughly 18%) in the customer group.
#
# As depicted with cluster -1, we had to discard 27% of the customer data due to too many missing values. In the general population, we removed 12% of values for the same reason.

# %% [markdown]
# > Congratulations on making it this far in the project! Before you finish, make sure to check through the entire notebook from top to bottom to make sure that your analysis follows a logical flow and all of your findings are documented in **Discussion** cells. Once you've checked over all of your work, you should export the notebook as an HTML document to submit for evaluation. You can do this from the menu, navigating to **File -> Download as -> HTML (.html)**. You will submit both that document and this notebook for your project submission.

# %%
