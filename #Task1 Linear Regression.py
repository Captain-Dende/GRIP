# Importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Importing the Dataset
dataset = pd.read_csv('student_scores.csv')

dataset.head(10)

reg_plot = sns.lmplot('Hours','Scores',dataset, scatter_kws={'marker':'o','color':'green'}, line_kws={'color':'red'})
reg_plot.savefig('img1.png')
plt.suptitle('Regression Graph')


# splitting the data into dependent variable (Y) and independent variable (X)
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,1:].values   # converting to 2D arrays


# splitting the datasets into train and and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=0)

# fitting simple linear regression on training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor = regressor.fit(X_train,Y_train)


# Visualising the result
plt.scatter(X_train, Y_train)
plt.plot(X_train, regressor.predict(X_train), color='red')
plt.xlabel('Hours studied')
plt.ylabel('Percentage Scores')
plt.show()

# Predicting test set result
X_test = X_test.reshape(-1,1)
Y_pred = regressor.predict(X_test)

# predicting with a specified hours (9.25hrs/day)
hours = 9.25
hours = np.reshape(hours,(-1,1))
my_prediction = regressor.predict(hours)
print('No of hours = ',hours)
print('Predicted score = ',my_prediction)

# Evaluating the Accuracy of the prediction
from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, Y_pred)) 