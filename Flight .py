#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns 


# In[6]:


data=pd.read_csv('C:/mydata/flight_dataset.csv')


# In[7]:


data.head()


# In[9]:


data.dtypes


# In[10]:


data.isnull().sum()


# In[12]:


#Label Encoder ile kategorik verileri sayısal verilere dönüştürme


# In[13]:


from sklearn.preprocessing import LabelEncoder


# In[19]:


data=pd.read_csv('C:/mydata/flight_dataset.csv')
ktg = ['Airline', 'Source', 'Destination']

le=LabelEncoder()

for sutun in ktg:
    data[sutun]=le.fit_transform(data[sutun])
    
print(data)    


# In[20]:


data.dtypes


# In[21]:


data.head


# In[25]:


data.columns


# In[22]:


#Hedef değişken belirleme 


# In[23]:


x=data.drop('Price', axis=1)
p= data['Price']


# In[33]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

x_train, x_test, p_train, p_test = train_test_split(x, p, test_size=0.2, random_state=42)

scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)


# In[34]:


#Haftaiçi ve haftasonu ve yoğun yaz ayları özelliği oluşturma


# In[37]:


data['Date']=pd.to_datetime(data['Date'])
data['Haftasonu'] = data['Date'].dt.weekday >=5
ysa=[6,7,8,9]
data['YoğunSezon'] = data['Date'].dt.month.isin(ysa)
print(data)


# In[40]:


#Regresyon analizi
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


# In[44]:


t=LinearRegression()
t.fit(x_train,p_train)
p_pred =t.predict(x_test)

r2 = r2_score(p_test, p_pred)
rmse = mean_squared_error(p_test, p_pred, squared=False)

print(f'R-kare (R²) değeri: {r2:.2f}')
print(f'RMSE değeri: {rmse:.2f}')


# In[49]:


#Fiyat ve Haftasonu için Görselleştirme
plt.figure(figsize=(10, 6))
sns.boxplot(x=data['Haftasonu'], y=data['Price'])
plt.title('Haftasonu ve Fiyatlar')
plt.xlabel('Haftasonu')
plt.ylabel('Price')
plt.xticks([0, 1], ['Hafta İçi', 'Hafta Sonu'])
plt.show()


# In[51]:


#Fiyat ve Yoğun sezon için Görselleştirme
plt.figure(figsize=(10, 6))
sns.boxplot(x=data['YoğunSezon'], y=data['Price'])
plt.title('YoğunSezon ve Fiyatlar')
plt.xlabel('Yoğun Sezon')
plt.ylabel('Price')
plt.xticks([0, 1], ['Normal Sezon', 'Yoğun Sezon'])
plt.show()


# In[52]:


#Random Forest algoritması


# In[53]:


from sklearn.ensemble import RandomForestRegressor


# In[56]:


rforest=RandomForestRegressor(n_estimators=100, random_state=42)
rforest.fit(x_train,p_train)
p_pred_rforest=rforest.predict(x_test)


# In[57]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Performans metriklerini hesaplama
mse_rforest = mean_squared_error(p_test, p_pred_rforest)
mae_rforest = mean_absolute_error(p_test, p_pred_rforest)
r2_rforest = r2_score(p_test, p_pred_rforest)

print("Random Forest Mean Squared Error:", mse_rforest)
print("Random Forest Mean Absolute Error:", mae_rforest)
print("Random Forest R-squared:", r2_rforest)


# In[58]:


#Görselleştirme


# In[59]:


# Gerçek ve tahmin edilen değerlerin scatter plot'u
plt.figure(figsize=(10, 6))
plt.scatter(p_test, p_pred, alpha=0.3)
plt.plot([p_test.min(), p_test.max()], [p_test.min(), p_test.max()], 'r--', lw=2)
plt.xlabel('Gerçek Değerler')
plt.ylabel('Tahmin Edilen Değerler')
plt.title('Gerçek vs Tahmin Edilen Değerler')
plt.show()


# In[ ]:




