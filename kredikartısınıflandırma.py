#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns 
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans 
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


# In[3]:


#Veri kümesini yükleme
data=pd.read_csv('C:/mydata/ccgeneral.csv')
data


# In[4]:


data.columns


# In[5]:


data.isnull().sum()


# In[6]:


data.dtypes


# In[42]:


#Veriyi ön işleme
data=data.select_dtypes(include=['float64', 'int64'])


# In[43]:


ktg=data.select_dtypes(include=['object']).columns
dt=data.drop(columns=ktg)


# In[44]:


#Eksik değerleri doldurma
imputer =SimpleImputer(strategy='mean')
imputed_data =imputer.fit_transform(dt)


# In[26]:


scaler=StandardScaler()
scaled_data=scaler.fit_transform(imputed_data)


# In[12]:


#KMeans


# In[50]:


#Elbow metodu ile küme sayısını belirleme
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

wcss = []  


for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(imputed_data)   
    wcss.append(kmeans.inertia_) 

# Plotting the results
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[56]:


kmeans = KMeans(n_clusters = 7, init = 'k-means++', random_state = 42)
kmeans.fit(scaled_data)
y_kmeans = kmeans.fit_predict(imputed_data)


# In[15]:


y_kmeans


# In[16]:


plt.scatter(imputed_data[:, 0], imputed_data[:, 1], c=y_kmeans, cmap='viridis', label='Clusters1')
plt.scatter(imputed_data[:, 1], imputed_data[:, 1], c=y_kmeans, cmap='viridis', label='Clusters2')
plt.scatter(imputed_data[:, 2], imputed_data[:, 1], c=y_kmeans, cmap='viridis', label='Clusters3')
plt.scatter(imputed_data[:, 3], imputed_data[:, 1], c=y_kmeans, cmap='viridis', label='Clusters4')
plt.scatter(imputed_data[:, 4], imputed_data[:, 1], c=y_kmeans, cmap='viridis', label='Clusters5')
plt.scatter(imputed_data[:, 5], imputed_data[:, 1], c=y_kmeans, cmap='viridis', label='Clusters6')
plt.scatter(imputed_data[:, 6], imputed_data[:, 1], c=y_kmeans, cmap='viridis', label='Clusters7')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300)
plt.title('The Relationship Between Purchases and Payments')
plt.xlabel('PAYMENTS')
plt.ylabel('PURCHASES')
plt.legend()
plt.show()


# In[52]:


#PCA 
pca = PCA(n_components=2)
X_PCA =pca.fit_transform(scaled_data)
pca_data = pd.DataFrame(data= X_PCA, columns=['PC1', 'PC2'])

print(pca_data.head())


# In[58]:


plt.figure(figsize=(10, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans, cmap='viridis')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('PCA ile K-Means Kümeleme Sonuçları')
plt.show()


# In[ ]:




