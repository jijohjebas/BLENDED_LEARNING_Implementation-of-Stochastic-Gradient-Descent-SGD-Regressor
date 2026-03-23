# BLENDED_LEARNING
# Implementation-of-Stochastic-Gradient-Descent-SGD-Regressor

## AIM:
To write a program to implement Stochastic Gradient Descent (SGD) Regressor for linear regression and evaluate its performance.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import Libraries: Bring in essential libraries such as pandas, numpy, matplotlib, and sklearn.

2.Load Dataset: Import the dataset containing car prices along with relevant features.

3.Data Preprocessing: Manage missing data and select key features for the model, if required.

4.Split Data: Divide the dataset into training and testing subsets.

5.Train Model: Build a linear regression model and train it using the training data.

6.Make Predictions: Apply the model to predict outcomes for the test set.

7.Evaluate Model: Measure the model's performance using metrics like R² score, Mean Absolute Error (MAE), etc.

8.Check Assumptions: Plot residuals to verify assumptions like homoscedasticity, normality, and linearity.

9.Output Results: Present the predictions and evaluation metrics.

## Program:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

#Load Dataset
data=pd.read_csv('CarPrice_Assignment.csv')
print(data.head())
print(data.info())

#Data Preprocessing
#Dropping unnecessary columns and handling categorial variables
data=data.drop(['CarName','car_ID'],axis=1)
data=pd.get_dummies(data,drop_first=True)


#Splitting the data into features and target variable
x=data.drop('price',axis=1)
y=data['price']

#Standardizing the data
scaler=StandardScaler()
x=scaler.fit_transform(x)
y=scaler.fit_transform(np.array(y).reshape(-1,1))

#Splitting the dataset into training and testing sets
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#Creating the SGD Regressor model
sgd_model=SGDRegressor(max_iter=1000,tol=1e-3)

#Fitting the model on the training data
sgd_model.fit(x_train,y_train)

#Making predictions
y_pred=sgd_model.predict(x_test)

#Evaluating model performance
mse=mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)


#Print Evaluation Metrics
print("Name:Jijo.H.Jebas")
print("Reg no:212225040156")
print("Mean Squared Error:",mse)
print("Mean Absolute Error:",mae)
print("R-squared Score:",r2)


#Print model coefficients
print("\nModel Coefficients")
print("Coefficients:",sgd_model.coef_)
print("Intercept:",sgd_model.intercept_)


#Visualizing actual vs predicted prices
plt.scatter(y_test,y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices using SGD Regressor")
plt.plot([min(y_test),max(y_test)],[min(y_test),max(y_test)],color='red')
plt.show()

print(y.shape)

```

## Output:
<img width="628" height="419" alt="image" src="https://github.com/user-attachments/assets/02ff93ed-aa41-41e3-b7b0-1e1bd3a07fb3" />
<img width="647" height="619" alt="image" src="https://github.com/user-attachments/assets/da3fd741-8b81-4ce3-b71d-93380eb9aa6d" />
<img width="633" height="203" alt="image" src="https://github.com/user-attachments/assets/a84ae308-ef99-43b5-8023-7936326be92b" />
<img width="677" height="433" alt="image" src="https://github.com/user-attachments/assets/07b3a99b-c28f-468f-a03d-587da60ec9a3" />


## Result:
Thus, the implementation of Stochastic Gradient Descent (SGD) Regressor for linear regression has been successfully demonstrated and verified using Python programming.
