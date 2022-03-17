#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Base libraries
import pandas as pd
import numpy as np

#Additional libraries
from tabulate import tabulate


# In[3]:


#Preprocess Transform libraries
from sklearn import preprocessing as p
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
#ML libraries
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import AffinityPropagation
#ML Metrics
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import silhouette_score#ML Metrics
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import silhouette_score
#Pipeline library
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.base import ClusterMixin
from sklearn.pipeline import Pipeline
from sklearn import set_config
#Graphic libraries
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:


data = pd.read_csv('WineQT.csv', header = 0)


# In[11]:


print (data.shape)


# In[19]:


data.head()


# In[20]:


#check data type and columns
data.info()


# In[23]:


#checking null values
data.isnull().sum()


# In[24]:


#il dataset è completo, senza valori nulli in alcuna colonna


# In[35]:


#describe values of the dataset
data.describe().round(2)


# In[37]:


# Drop column ID

data.drop(columns="Id",inplace=True)


# In[169]:


#traslo la matrice per compattarla
data.describe().round(2).T


# In[255]:


#normalizzo il mio dataset

def minmax_norm(data_input):
    return (data - data.min()) / ( data.max() - data.min())

data_minmax_norm = minmax_norm(data)

display(data_minmax_norm.round(3))


# In[256]:


from sklearn.preprocessing import StandardScaler
data = data
scaler = StandardScaler()
print(scaler.fit(data))

standard_dataframe = pd.DataFrame(scaler.transform(data))   
standard_dataframe.columns=["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","quality"]
display(standard_dataframe.round(3))


# In[60]:


#graph all the data set
data.plot(figsize=(20,8))


# In[253]:


data.loc[data['total sulfur dioxide'] >150]


# In[179]:


#graph all the data set
data_minmax_norm.plot(figsize=(20,8))


# In[217]:


#graph all the data set
standard_dataframe.plot(figsize=(20,8))


# In[174]:


# making Group by quality

ave_qu = data.groupby("quality").mean()
ave_qu


# In[176]:


# making Group by quality

ave_qu_norm = data_minmax_norm.groupby("quality").mean()
ave_qu_norm


# In[218]:


# making Group by quality

ave_qu_st = standard_dataframe.groupby("quality").mean()
ave_qu_st


# In[175]:


# graph the group by quality

ave_qu.plot(kind="bar",figsize=(20,10))


# In[177]:


# graph the group by quality

ave_qu_norm.plot(kind="bar",figsize=(20,10))


# In[221]:


# graph the group by quality

ave_qu_st.plot(kind="bar",figsize=(20,10))


# In[95]:


plt.figure(figsize=(20,7))
sns.lineplot(data=data, x="quality",y="volatile acidity",label="Volatile Acidity")
sns.lineplot(data=data, x="quality",y="citric acid",label="Citric Acid")
sns.lineplot(data=data, x="quality",y="chlorides",label="chlorides")
sns.lineplot(data=data, x="quality",y="pH",label="PH")
sns.lineplot(data=data, x="quality",y="sulphates",label="Sulphates")
plt.ylabel("quantity")
plt.title("Impact on quality")
plt.legend()
plt.show()


# In[178]:


plt.figure(figsize=(20,7))
sns.lineplot(data=data_minmax_norm, x="quality",y="volatile acidity",label="Volatile Acidity")
sns.lineplot(data=data_minmax_norm, x="quality",y="citric acid",label="Citric Acid")
sns.lineplot(data=data_minmax_norm, x="quality",y="chlorides",label="chlorides")
sns.lineplot(data=data_minmax_norm, x="quality",y="pH",label="PH")
sns.lineplot(data=data_minmax_norm, x="quality",y="sulphates",label="Sulphates")
plt.ylabel("quantity")
plt.title("Impact on quality")
plt.legend()
plt.show()


# In[222]:


plt.figure(figsize=(20,7))
sns.lineplot(data=standard_dataframe, x="quality",y="volatile acidity",label="Volatile Acidity")
sns.lineplot(data=standard_dataframe, x="quality",y="citric acid",label="Citric Acid")
sns.lineplot(data=standard_dataframe, x="quality",y="chlorides",label="chlorides")
sns.lineplot(data=standard_dataframe, x="quality",y="pH",label="PH")
sns.lineplot(data=standard_dataframe, x="quality",y="sulphates",label="Sulphates")
plt.ylabel("quantity")
plt.title("Impact on quality")
plt.legend()
plt.show()


# In[ ]:


import plotly.graph_objects as go
import plotly.express as px


# In[109]:


# using graph interactive the show the effect free and total - sulfur dioxide in the quality

t=px.scatter(data, x="free sulfur dioxide", y="total sulfur dioxide",animation_frame="quality")


# In[131]:


t.show()


# In[295]:


X = data.drop(columns="quality") 
Z = data_minmax_norm.drop(columns="quality")  
W = standard_dataframe.drop(columns="quality")  
y = data["quality"]    # y = quality


# In[298]:


# split the data train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
Z_train, Z_test, y_train, y_test = train_test_split(Z, y, test_size=0.25, random_state=42)
W_train, W_test, y_train, y_test = train_test_split(W, y, test_size=0.25, random_state=42)

print("X, Z, W Train : ", X_train.shape)
print("X, Z, W Test  : ", X_test.shape)
print("Y Train : ", y_train.shape)
print("Y Test  : ", y_test.shape)


# In[286]:


#Importing the basic librarires for building model


from sklearn.linear_model import LinearRegression ,LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error ,mean_squared_error, median_absolute_error,confusion_matrix,accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC ,SVR


# In[287]:


# using the model LinearRegression
LR_model=LinearRegression()

# fit model
LR_model.fit(X_train,y_train)

# Score X and Y - test and train

print("Score the X-train with Y-train is : ", LR_model.score(X_train,y_train))
print("Score the X-test  with Y-test  is : ", LR_model.score(X_test,y_test))

# Expected value Y using X test
y_pred_LR=LR_model.predict(X_test)

# Model Evaluation
print( " Model Evaluation Linear R : mean absolute error is ", mean_absolute_error(y_test,y_pred_LR))
print(" Model Evaluation Linear R : mean squared  error is " , mean_squared_error(y_test,y_pred_LR))
print(" Model Evaluation Linear R : median absolute error is " ,median_absolute_error(y_test,y_pred_LR)) 


# In[301]:


#applicando la regressione lineare ai dataset standardizzati o normalizzati i risultati sono perfettamente identici. 
# si è pertanto deciso di omettere questa parte perché superflua


# In[313]:


# using the model Logistic Regression

Lo_modelX=LogisticRegression(solver='liblinear')
Lo_modelZ=LogisticRegression(solver='liblinear')
Lo_modelW=LogisticRegression(solver='liblinear')

# fit model

Lo_modelX.fit(X_train,y_train)
Lo_modelZ.fit(Z_train,y_train)
Lo_modelW.fit(W_train,y_train)


# Score X and Y - test and train model Logistic Regression

print("Score the X-train with Y-train is : ", Lo_modelX.score(X_train,y_train))
print("Score the X-test  with Y-test  is : ", Lo_modelX.score(X_test,y_test))

print("Score the Z-train with Y-train is : ", Lo_modelZ.score(Z_train,y_train))
print("Score the Z-test  with Y-test  is : ", Lo_modelZ.score(Z_test,y_test))

print("Score the W-train with Y-train is : ", Lo_modelW.score(W_train,y_train))
print("Score the W-test  with Y-test  is : ", Lo_modelW.score(W_test,y_test))


# Expected value Y using X test
y_pred_Lo=Lo_modelX.predict(X_test)

# Model Evaluation
print( " Model Evaluation Logistic R : mean absolute error is ", mean_absolute_error(y_test,y_pred_Lo))
print(" Model Evaluation Logistic R : mean squared  error is " , mean_squared_error(y_test,y_pred_Lo))
print(" Model Evaluation Logistic R : median absolute error is " ,median_absolute_error(y_test,y_pred_Lo)) 

print(" Model Evaluation Logistic R : accuracy score " , accuracy_score(y_test,y_pred_Lo))


# In[320]:


# using the model Decision Tree Classifier
Tree_modelX=DecisionTreeClassifier(max_depth=10)
Tree_modelZ=DecisionTreeClassifier(max_depth=10)
Tree_modelW=DecisionTreeClassifier(max_depth=10)
# fit model
Tree_modelX.fit(X_train,y_train)
Tree_modelZ.fit(Z_train,y_train)
Tree_modelW.fit(W_train,y_train)

# Score X and Y - test and train

print("Score the X-train with Y-train is : ", Tree_modelX.score(X_train,y_train))
print("Score the X-test  with Y-test  is : ", Tree_modelX.score(X_test,y_test))
print("Score the Z-train with Y-train is : ", Tree_modelZ.score(Z_train,y_train))
print("Score the Z-test  with Y-test  is : ", Tree_modelZ.score(Z_test,y_test))
print("Score the W-train with Y-train is : ", Tree_modelW.score(W_train,y_train))
print("Score the W-test  with Y-test  is : ", Tree_modelW.score(W_test,y_test))

# Select  Important columns

print("The Important columns \n",Tree_modelX.feature_importances_)


# In[315]:


print("The classes ",Tree_modelX.classes_)

y_pred_T =Tree_modelX.predict(X_test)

print(" Model Evaluation Decision Tree : accuracy score " , accuracy_score(y_test,y_pred_T))


# In[327]:


# using the model SVC
svc_modelX=SVC(C=100,kernel="rbf")
svc_modelZ=SVC(C=100,kernel="rbf")
svc_modelW=SVC(C=100,kernel="rbf")

# fit model
svc_modelX.fit(X_train,y_train)
svc_modelZ.fit(Z_train,y_train)
svc_modelW.fit(W_train,y_train)

y_pred_svcX =svc_modelX.predict(X_test)
y_pred_svcZ =svc_modelZ.predict(Z_test)
y_pred_svcW =svc_modelW.predict(W_test)

print("Score the X-train with Y-train is : ", svc_modelX.score(X_train,y_train))
print("Score the X-test  with Y-test  is : ", svc_modelX.score(X_test,y_test))
print("Score the Z-train with Y-train is : ", svc_modelZ.score(Z_train,y_train))
print("Score the Z-test  with Y-test  is : ", svc_modelZ.score(Z_test,y_test))
print("Score the W-train with Y-train is : ", svc_modelW.score(W_train,y_train))
print("Score the W-test  with Y-test  is : ", svc_modelW.score(W_test,y_test))
print(" Model Evaluation Decision Tree : accuracy score " , accuracy_score(y_test,y_pred_svcZ))


# In[338]:


# using the model SVR

svr_modelX=SVR(degree=1,coef0=1, tol=0.001, C=1.5,epsilon=0.001)
svr_modelZ=SVR(degree=1,coef0=1, tol=0.001, C=1.5,epsilon=0.001)
svr_modelW=SVR(degree=1,coef0=1, tol=0.001, C=1.5,epsilon=0.001)

# fit model
svr_modelX.fit(X_train,y_train)
svr_modelZ.fit(Z_train,y_train)
svr_modelW.fit(W_train,y_train)

y_pred_svr =svc_modelW.predict(W_test)

print("Score the X-train with Y-train is : ", svr_modelX.score(X_train,y_train))
print("Score the X-test  with Y-test  is : ", svr_modelX.score(X_test,y_test))
print("Score the Z-train with Y-train is : ", svr_modelZ.score(Z_train,y_train))
print("Score the Z-test  with Y-test  is : ", svr_modelZ.score(Z_test,y_test))
print("Score the W-train with Y-train is : ", svr_modelW.score(W_train,y_train))
print("Score the W-test  with Y-test  is : ", svr_modelW.score(W_test,y_test))
print(" Model Evaluation Decision Tree : accuracy score " , accuracy_score(y_test,y_pred_svr))


# In[349]:


# using the model K Neighbors Classifier

K_modelX = KNeighborsClassifier(n_neighbors = 5)
K_modelX.fit(X_train, y_train)
K_modelZ = KNeighborsClassifier(n_neighbors = 13)
K_modelZ.fit(Z_train, y_train)
K_modelW = KNeighborsClassifier(n_neighbors = 7)
K_modelW.fit(W_train, y_train)

y_pred_k = K_model.predict(Z_test)

print("Score the X-train with Y-train is : ", K_modelX.score(X_train,y_train))
print("Score the X-test  with Y-test  is : ", K_modelX.score(X_test,y_test))
print("Score the Z-train with Y-train is : ", K_modelZ.score(Z_train,y_train))
print("Score the Z-test  with Y-test  is : ", K_modelZ.score(Z_test,y_test))
print("Score the W-train with Y-train is : ", K_modelW.score(W_train,y_train))
print("Score the W-test  with Y-test  is : ", K_modelW.score(W_test,y_test))
print(" Model Evaluation K Neighbors Classifier : accuracy score " , accuracy_score(y_test,y_pred_k))


# In[350]:


#i diversi n_neighbors sono stati scelti tra quelli che, in diverse prove casuali, massimizzano lo score nei test


# In[ ]:




