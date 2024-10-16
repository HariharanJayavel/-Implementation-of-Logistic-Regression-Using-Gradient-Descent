# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Load the dataset.
3. Define X and Y array.
4. Define a function for costFunction,cost and gradient.
5. Define a function to plot the decision boundary. 6.Define a function to predict the 
   Regression value.

## Program and Output:
```

Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: HARIHARAN J
RegisterNumber: 212223240047
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv('/content/Placement_Data_Full_Class (1).csv')
dataset.info() 
```
![image](https://github.com/user-attachments/assets/a3cdf589-37aa-4bfe-9b3d-e674f9dbfff1)
```
dataset.drop('sl_no',axis=1)
dataset = dataset.drop('sl_no',axis=1)
dataset.info()
```
![image](https://github.com/user-attachments/assets/447c8804-9770-42ce-9607-1c08ba513b6f)
```
dataset["gender"] = dataset["gender"].astype('category')
dataset["ssc_b"] = dataset["ssc_b"].astype('category')
dataset["hsc_b"] = dataset["hsc_b"].astype('category')
dataset["hsc_s"] = dataset["hsc_s"].astype('category')
dataset["degree_t"] = dataset["degree_t"].astype('category')
dataset["workex"] = dataset["workex"].astype('category')
dataset["specialisation"] = dataset["specialisation"].astype('category')
dataset["status"] = dataset["status"].astype('category')
dataset.dtypes
```
![image](https://github.com/user-attachments/assets/5d826655-2200-43a7-b3fd-b7ad09f4713a)
```
dataset["gender"] = dataset["gender"].cat.codes
dataset["ssc_b"] = dataset["ssc_b"].cat.codes
dataset["hsc_b"] = dataset["hsc_b"].cat.codes
dataset["hsc_s"] = dataset["hsc_s"].cat.codes
dataset["degree_t"] = dataset["degree_t"].cat.codes
dataset["workex"] = dataset["workex"].cat.codes
dataset["specialisation"] = dataset["specialisation"].cat.codes
dataset["status"] = dataset["status"].cat.codes
dataset
```
![image](https://github.com/user-attachments/assets/2e1efbf1-2c19-41c5-872e-347caf80f1d1)
```
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
y
```
![image](https://github.com/user-attachments/assets/e6964b50-76e3-4825-83d4-5761c0bb4b65)
```
theta = np.random.randn(x.shape[1])
Y=y
def sigmoid(z):
  return 1/(1+np.exp(-z))

def loss(theta,x,Y):
  h=sigmoid(x.dot(theta))
  return -np.sum(Y*np.log(h)+(1-Y)*np.log(1-h))
def gradient_descent(theta,x,Y,alpha,num_iterations):
  m=len(Y)
  for i in range(num_iterations):
    h=sigmoid(x.dot(theta))
    gradient=x.T.dot(h-Y)/m
    theta-=alpha*gradient
  return theta
theta=gradient_descent(theta,x,y,alpha=0.01,num_iterations=1000)
def predict(theta,x):
  h=sigmoid(x.dot(theta))
  y_pred=np.where(h>=0.5,1,0)
  return y_pred

y_pred=predict(theta,x)
accuracy=np.mean(y_pred.flatten() == y)
print(accuracy)
```
![image](https://github.com/user-attachments/assets/619fced8-3580-4d48-a08c-d886e8438836)

```
print(theta)
```
![image](https://github.com/user-attachments/assets/ddf619de-f36d-478b-b869-111a3cf603ee)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

