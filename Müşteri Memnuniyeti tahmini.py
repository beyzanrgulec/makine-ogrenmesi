#!/usr/bin/env python
# coding: utf-8
# import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
# In[3]:


data = pd.read_csv('C:/mydata/Customer_support_data.csv')


# In[5]:


data.head()


# In[6]:


data.dtypes


# In[7]:


data.columns


# In[8]:


data.isnull().sum()


# In[9]:


#Müşteri memnuniyetisınıfları belirleme

def classify_satisfaction(csat_score):
    if csat_score <=2:
        return 'Low Satisfaction'
    elif csat_score == 3:
        return 'Medium Satisfaction'
    else:
        return 'High Satisfaction'


# In[28]:


data.dropna(inplace=True)


# In[12]:


csat_score=data[['CSAT Score']]
print(csat_score)


# In[16]:


#hedef ve bağımsız değişkenleri belirleme
x=data.drop('CSAT Score', axis=1)
y=data['CSAT Score']


# In[17]:


x=pd.get_dummies(x)


# In[31]:


# Veriyi eğitim ve test setlerine böl
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
y_pred=m.predict(X_test)


# In[41]:


from sklearn.ensemble import RandomForestRegressor

# Modeli seçin ve eğitin
m = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42)
m.fit(X_train, y_train)


# In[49]:


from sklearn.metrics import mean_squared_error, r2_score

y_pred = m.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R2 Score: {r2}') 


# In[25]:


#Histogram
data.hist(bins=30, figsize=(15, 10))
plt.show()



# In[26]:


# Kutu Grafik
plt.figure(figsize=(15, 10))
sns.boxplot(data)
plt.show()


# In[36]:


# Scatter Plot
sns.pairplot(data)
plt.show()

