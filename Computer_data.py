import pandas as pd
import scipy 
from scipy import stats
import statsmodels.formula.api as sm
from statsmodels.formula.api import ols
import numpy as np

# Import Dataset

Computer_data = pd.read_csv("C:/Users/SHAJIUDDIN MOHAMMED/Desktop/Data sets/Computer_Data.csv")

# Removing Unnecessary Columns

Computer_data.columns = "Serial.No","price","speed","hd","ram","screen","cd","multi","premium","ads","trend"

Computer_data1 = Computer_data.drop(columns = "Serial.No")
Computer_data1


# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

Computer_data.describe()

# Graphical Representation

import matplotlib.pyplot as plt

plt.bar(height = Computer_data1.speed , x = np.arange(1,6260,1))
plt.hist(Computer_data.speed)
plt.boxplot(Computer_data.speed)

# Joint Plots 

import seaborn as sns
sns.jointplot(x = Computer_data1.speed ,y = Computer_data1.price)
sns.jointplot(x = Computer_data1.hd ,y = Computer_data1.price)
sns.jointplot(x = Computer_data1.ram ,y = Computer_data1.price)

# Box Plot Representation

Computer_data1.boxplot('price','cd') 
# From the graph we can clearly see there are many outliers are present

Computer_data1.boxplot('price','multi') 
# From the graph we can clearly see there are many outliers are present

Computer_data1.boxplot('price','premium') 
# From the graph we can clearly see there are many outliers are present

Computer_data1.info()

# Checking the NA values and Count of the variables

cat_Compudata1 = Computer_data.select_dtypes(include = ['object']).copy()
cat_Compudata1
print(cat_Compudata1.isnull().values.sum()) 

print(cat_Compudata1['cd'].value_counts())
print(cat_Compudata1['multi'].value_counts())
print(cat_Compudata1['premium'].value_counts())

# Plot Representation of the String Variables

cd_count = cat_Compudata1.cd.value_counts()
sns.set(style = "darkgrid")
sns.barplot(cd_count.index,cd_count.values,alpha = 0.9)
plt.show()

# Multi Variable
multi_count = cat_Compudata1.multi.value_counts()
sns.set(style = "dark")
sns.barplot(multi_count.index,multi_count.values,alpha = 0.9)
plt.show()

# Premium Varibale
premium_count = cat_Compudata1.premium.value_counts()
sns.set(style = "dark")
sns.barplot(premium_count.index,premium_count.values,alpha = 0.9)
plt.show()


# Creation of Dummy Variabels

cat_Compudata1_onehot = cat_Compudata1
cat_Compudata1_onehot = pd.get_dummies(cat_Compudata1_onehot, columns=['cd','multi','premium'], prefix = ['cd','multi','premium'])
print(cat_Compudata1_onehot.head())


# Concatenation of the Dummy variables to data sheet and drop of original columns

Compudata_df = pd.concat([Computer_data1, cat_Compudata1_onehot], axis=1)
Compudata_df
Compudata_df = Compudata_df.drop(['cd','multi','premium'], axis=1)
Compudata_df


# Scatter plot 

sns.pairplot(Compudata_df.iloc[:, :])
                             
# Correlation matrix 
Compudata_df.corr()

# As per above analysis there is no collinearty problem exists

# As there are multiple input variable , creating object for input and output variables
y = Compudata_df.iloc[:,0]
x = Compudata_df.iloc[: , 1 :]

# MODEL BUILDING

model = sm.ols('y ~ x', data = Compudata_df).fit()
model.summary()

# Checking the influence values
# Influence Index Plots
import statsmodels.api as smf

smf.graphics.influence_plot(model)

# Studentized Residuals = Residual/standard deviation of residuals
# index 1440 & 1700 is showing high influence so we can exclude that entire row

Compudata_new = Compudata_df.drop(Compudata_df.index[[1440,1700]])

x1 = Compudata_new.iloc[: , 1 :]
x1
y1 = Compudata_new.iloc[:,0]
y1

# Model after removing the Outliers

model1 = sm.ols('np.sqrt(y1) ~ x1', data = Compudata_new).fit()
model1.summary() 

# R^2 value = 0.79 and all the other varibales are significant 

# Prediction
pred = model1.predict(Compudata_new)

# Q-Q plot
res = model1.resid
smf.qqplot(res)
plt.show()

# Q-Q plot
from scipy import stats
import pylab
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

# Residuals vs Fitted plot
sns.residplot(x = pred, y = np.sqrt(y1), lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

smf.graphics.influence_plot(model1)

### Splitting the data into train and test data
 
from sklearn.model_selection import train_test_split
Compudata_df_train, Compudata_df_test = train_test_split(Compudata_df, test_size = 0.2) # 20% test data
Compudata_df_test
Compudata_df_train

# preparing the model on train data 
model_train = sm.ols('np.sqrt(price) ~ speed + hd + ram + screen + ads + trend + cd_no + cd_yes + multi_no + multi_yes + premium_no + premium_yes', data = Compudata_df).fit()
model_train.summary()


# prediction on test data set 

test_pred = model_train.predict(Compudata_df_test)
test_pred

# test residual values 
test_resid = test_pred - Compudata_df_test.price 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse 

# The results of test Rmse = 2245.4081092565443

# train_data prediction
train_pred = model_train.predict(Compudata_df_train)
train_resid  = train_pred - Compudata_df_train.price
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse 

# The results of train Rmse = 2248.5407893111137

# As there is only slight variation in the test and train RMSE,the model is the best fit model

