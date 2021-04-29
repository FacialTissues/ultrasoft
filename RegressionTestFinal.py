import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab
import math
import matplotlib.ticker as mtick

from scipy import stats
from scipy.stats import kurtosis, skew

import statsmodels.api as sm
from statsmodels.stats import diagnostic as diag
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

dataset = pd.read_excel('C:/Users/CAAMCO 14May2018/Downloads/Regression.xlsx')
dataset.columns = dataset.columns.str.replace(' ', '')
pd.set_option('max_columns', None)

# set the index equal to the year column
dataset.index = dataset['Date']
dataset = dataset.drop('Date', axis = 1)

#set the data type of the data frame
dataset = dataset.astype(float)
dataset = dataset.loc['1993-01-25':'2020-09-28']
print(dataset)

y = dataset['PerformanceofSPYafter26weeks']
x = dataset['5-30USYieldCurve']
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x,y,'o',label= 'Relationship')
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_title('Performance of SPY after 26 weeks against 5-30 US Yield Curve')
ax.set_xlabel('5-30 US Yield Curve')
ax.set_ylabel('% change after 26 weeks (SPY)')
ax.legend()
plt.show()

print(dataset.corr())
print(dataset.describe())

Y = dataset[['PerformanceofSPYafter26weeks']]
X = dataset[['5-30USYieldCurve']]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)


regressor = LinearRegression()  
regressor.fit(X_train, y_train) #training the algorithm

#To retrieve the intercept:
intercept = regressor.intercept_[0]
#For retrieving the slope:
coefficient = regressor.coef_[0][0]

print("The coefficient for our model is {:.3}".format(coefficient))
print("The intercept for our mode is {:.4}".format(intercept))

prediction = regressor.predict([[1.5]])
predicted_value = prediction [0][0]
print("The predicted value is {:.4}".format(predicted_value))

y_predict = regressor.predict(X_test)

print(y_predict[:5])

#Evaluating the Model
X2 = sm.add_constant(X)
#create a OLS model
model = sm.OLS(Y, X2)
#fit the data
est = model.fit()

#make some confidence intercals, 95% by default.
print(est.conf_int())

#Hypothesis testing
#estimate the p-values
print(est.pvalues)

model_mse = mean_squared_error(y_test, y_predict)
model_mae = mean_absolute_error(y_test, y_predict)
model_rmse = math.sqrt(model_mse)

print("MSE {:.3}".format(model_mse))
print("MAE {:.3}".format(model_mae))
print("RMSE {:.3}".format(model_rmse))

#R-Squared
model_r2 = r2_score(y_test, y_predict)
print("R2 {:.2}".format(model_r2))

#print out a summary
print(est.summary())

#plot residuals
(y_test - y_predict).hist(grid = False )
plt.title("Model Residuals")
plt.show()

#Plotting line
plt.scatter(X_test, y_test, label = 'y')
plt.plot(X_test, y_predict, linewidth = 5, linestyle = '-', label = 'Predicted y')
plt.title(" Performance of SPY after 26weeks Vs. 5-30 US Yield Curve")
plt.xlabel("5-30 US Yield Curve")
plt.ylabel("Performance of SPY after 26 weeks")
plt.legend()
plt.show()

#Save the model for future use
import pickle
with open ('my_linear_regression.sav', 'wb') as f:
    pickle.dump(regressor,f)

with open ('my_linear_regression.sav', 'rb') as f:
    regressor_2 = pickle.load(f)

print(regressor_2.predict([[1.5]]))