# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 15:23:05 2019

@author: SEVVAL
"""
#               *** CLUSTERİNG ***

import pandas as pd
data1 = pd.read_csv("data.csv")
data = data1.drop(['filename','label'],axis=1) #18
#%%  normalizasyon
import numpy as np
from sklearn.preprocessing import  StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(np.array(data, dtype = float))
#%%
dat = (X-np.min(X))/(np.max(X)-np.min(X))

#%%Choosing the Number of Components in a Principal Component Analysis

from sklearn.decomposition import PCA    
import matplotlib.pyplot as plt    
pca = PCA().fit(dat)
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title(' Dataset Explained Variance')
plt.show()

#%%  silhouette_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
result = []
#n_clusters=3
for n_clusters in list (range(3,25)):
   clusterer = KMeans (n_clusters=n_clusters, init = 'k-means++').fit(X)
   preds = clusterer.predict(X)
   centers = clusterer.cluster_centers_
   result.append(silhouette_score(X, preds, sample_size = 26))

import matplotlib.pyplot as plt 
plt.plot(range(3,25), result, 'bx-')
plt.xlabel('number of clusters')
plt.ylabel('result')

plt.show()      
#%% 3 boyutlu görselleştirmek için n_components değerini 3 seçtik
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(X)
#%%
from sklearn.cluster import KMeans
n_clusters=8

clusterer = KMeans (n_clusters=n_clusters, init = 'k-means++').fit(principalComponents)
preds = clusterer.predict(principalComponents)
centers = clusterer.cluster_centers_

  
#%% plt for kmeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(1, figsize=(8,6))
ax = Axes3D(fig, elev=-150, azim=110)
ax.scatter(principalComponents[:,0],principalComponents[:,1],principalComponents[:,2], c=preds,cmap=plt.cm.Set1, edgecolor='k')
#ax.scatter(data.iloc[:,0],data.iloc[:,1],data.iloc[:,2], c=aa,cmap=plt.cm.Set1, edgecolor='k')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
plt.show()
#%%  Hieararchial Clustering
from sklearn.cluster import AgglomerativeClustering
import numpy as np
clustering = AgglomerativeClustering(n_clusters=None,distance_threshold=2.4,compute_full_tree=True).fit(principalComponents)
#  fit(data) için distance_threshold=3000,   fit(X) için distance_threshold=22,  fit(dat) için distance_threshold=2
clusters_Sayi=clustering.n_clusters_
labels=clustering.labels_
#bb0=list(aa).count(0)
 
#%%   plt for Hieararchial Clustering
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(1, figsize=(8,6))
ax = Axes3D(fig, elev=-150, azim=110)
ax.scatter(principalComponents[:,0],principalComponents[:,1],principalComponents[:,2], c=labels,cmap=plt.cm.Set1, edgecolor='k')
#ax.scatter(data.iloc[:,0],data.iloc[:,1],data.iloc[:,2], c=aa,cmap=plt.cm.Set1, edgecolor='k')
ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
plt.show()