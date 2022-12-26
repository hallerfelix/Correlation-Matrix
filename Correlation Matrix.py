#!/usr/bin/env python
# coding: utf-8

# In[1]:


#impot libraries
import pandas as pd
import seaborn as sns
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import scikit_posthocs as sp
import os
get_ipython().run_line_magic('matplotlib', 'inline')

## Create a random dataframe

# Generates a random NumPy array with 30 rows and 16 columns and stores it in the variable data.
data = np.random.rand(30, 16)

# Defines a list of column names
columns = ['Normalized.Richness',
           'Shannon.Index',
           'Weight',
           'Liver-to-body weight',
           'Spleen-to-body weight',
           'Tumor number',
           'Tumor burden',
           'MTS',
           'Adenoma number',
           'Carcinoma number',
           'Escherichia-Shigella',
           'Lachnospiraceae NK4A136 group',
           'Lachnospiraceae UCG-006',
           'Roseburia',
           'Erysipelotrichaceae',
           'Eubacterium coprostanoligenes group']

# Uses the list of column names to create a Pandas DataFrame df with the random data.
df = pd.DataFrame(data, columns=columns)

# Calculates the correlation matrix for the data using the np.corrcoef function and stores it in the variable corr_matrix.
corr_matrix = np.corrcoef(data, rowvar=False)

# Performs a matrix multiplication between data and corr_matrix and stores the result back in data.
data = data @ corr_matrix

# Creates a new DataFrame df using the modified data and the same list of column names.
df = pd.DataFrame(data, columns=columns)

###########################################################################################################################
###########################################################################################################################

# "import" dataframe
df = df
# startcolumn of microbiome
# microbiome data will be centered log-transformed
startcolumn_microbiome = 11

###########################################################################################################################
###########################################################################################################################

# centre log transform the dataframe
def centered_log_transform(df, col):
    
    # Take the natural logarithm of the values in the column
    logged_col = np.log(df[col])

    # Subtract the mean of the logged values from each value to center the data around zero
    centered_log_col = logged_col - np.mean(logged_col)

    return centered_log_col

###########################################################################################################################
###########################################################################################################################

# create new folder to save plots
if not os.path.isdir("/Plots"):
    # if path folder in working directory is 
    # not present then create it.
    try: 
        os.mkdir("Plots") 
    except OSError as error: 
        print("The path ~Plots/ already exists")

# make metadata and microbiome dataframe
df_meta = df.iloc[:, 0:startcolumn_microbiome-1]
df_biom = df.iloc[:, startcolumn_microbiome-1:16]

# replace all 0 values of the microbiome table with NaN
df_biom = df_biom.replace(0, np.nan)

# get columns biom
col_biom = df_biom.columns.tolist()

# apply centered_log_transform function on col_biom
for col in col_biom:
    df_biom[col] = centered_log_transform(df_biom, col)

# join both dataframes
df = df_meta.join(df_biom)

# make columns list
columns = df.columns.tolist()

# center and scale the dataframe    
scaler = StandardScaler()
df[df.columns] = scaler.fit_transform(df[df.columns])

# make new dataframe for R^2
df_stat = pd.DataFrame(np.zeros([len(columns), 0])*np.nan)
df_stat = df_stat.set_axis(columns)

# calculate Pearson R^2
df_stat = df.corr(method="pearson")

# calculate Pearson p-value
df_p = df.corr(method=lambda x, y: stats.pearsonr(x, y)[1]) - np.eye(*df.corr().shape)

# make new R^2 dataframe that only includes significant correlations
# Iterate over the columns of the DataFrame
for col in df_stat.columns.tolist():
    # Iterate over the indices of the DataFrame
    for row in df_stat.index.tolist():
        # If the value in df_p at (row,col) is greater than 0.05
        if df_p[row][col] > 0.05:
            # Set the value in df_stat at (row,col) to np.nan
            df_stat[row][col] = np.nan

# make nice seaborn correlation matrix
# Create a mask for the heatmap
mask = np.triu(np.ones_like(df_stat, dtype=bool))

# Create a figure and axes for the heatmap
f, ax = plt.subplots(figsize=(11, 9))

# Create a diverging color palette
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Create the heatmap
sns.heatmap(df_stat, mask=mask, vmax=1, vmin=-1, cmap=cmap, center=0, annot=True,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot_kws={"color":"k"})
#show the plot
plt.show()

# save the plot
plt.savefig(f"Plots/Correlation_original.png", format="png",bbox_inches="tight")

# make * for p-values
p = df_p.applymap(lambda x: ''.join(['*' for t in [.05, .01, .001] if x<=t]))

# new dataframe with R² and * for p-values
output = df.corr().round(2).astype(str) + p
output

