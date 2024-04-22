#Importing library
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
dataset=pd.read_csv("diabetes.csv")
dataset.info()
dataset.isnull().sum()
dataset.describe()
#Exploring Glucose and target variables
plt.figure(figsize =(10,8))
sns.violinplot(data=dataset,x="Outcome",y="Glucose",split=True,linewidth=2,inner="quart")
#Exploring Glucose and target variables
plt.figure(figsize =(10,8))
#Plotting Density function graph of the glucose and target variable
kde=sns.kdeplot(dataset["Glucose"][dataset["Outcome"]==1],color="Red",fill=True)
kde=sns.kdeplot(dataset["Glucose"][dataset["Outcome"]==0],color="Blue",fill=True)
kde.set_xlabel("Glucose")
kde.set_ylabel("Density")
kde.legend(["Positive","Negative"])
#Replace 0 values with the mean/median of the respective features
#Glucose
dataset["Glucose"]=dataset["Glucose"].replace(0, dataset["Glucose"].median())
#BloodPressure
dataset["BloodPressure"]=dataset["BloodPressure"].replace(0, dataset["BloodPressure"].median())
#SkinThickness
dataset["SkinThickness"]=dataset["SkinThickness"].replace(0, dataset["SkinThickness"].mean())
#Insulin
dataset["Insulin"]=dataset["Insulin"].replace(0, dataset["Insulin"].median())
#BMI
dataset["BMI"]=dataset["BMI"].replace(0, dataset["BMI"].mean())
#Splitting the dependent and independent variable
x=dataset.drop(["Outcome"],axis=1)
y=dataset["Outcome"]
x
y
#Splitting the dataset into training and testing dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(x, y, test_size=0.33,random_state=42)
x_train

from sklearn.neighbors import KNeighborsClassifier
training_accuracy=[]
test_accuracy=[]
for n_neighbors in range (1,11):
    knn=KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(x_train,y_train)
    #Check accuracy score
    training_accuracy.append(knn.score(x_train,y_train))
    test_accuracy.append(knn.score(x_test,y_test))
plt.plot(range(1,11),training_accuracy,label="training_accuracy")
plt.plot(range(1,11),test_accuracy,label="test_accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
knn=KNeighborsClassifier(n_neighbors=9)
knn.fit(x_train,y_train)
print(knn.score(x_train,y_train),"Training accuracy")
print(knn.score(x_test,y_test),"Test accuracy")

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(random_state=0)
dt.fit(x_train,y_train)
print(dt.score(x_train,y_train),": Training Accuracy")
print(dt.score(x_test,y_test),": Test Accuracy")
dt1=DecisionTreeClassifier(random_state=0,max_depth=3)
dt1.fit(x_train,y_train)
print(dt1.score(x_train,y_train),": Training Accuracy")
print(dt1.score(x_test,y_test),": Test Accuracy")

from sklearn.neural_network import MLPClassifier
mlp=MLPClassifier(random_state=0)
mlp.fit(x_train,y_train)
print(mlp.score(x_train,y_train),": Training Accuracy")
print(mlp.score(x_test,y_test),": Test Accuracy")

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train_scaled=sc.fit_transform(x_train)
x_test_scaled=sc.fit_transform(x_test)
mlp1=MLPClassifier(random_state=0)
mlp1.fit(x_train,y_train)
print(mlp1.score(x_train_scaled,y_train),": Training Accuracy")
print(mlp1.score(x_test_scaled,y_test),": Test Accuracy")