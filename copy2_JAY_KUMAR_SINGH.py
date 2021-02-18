#!/usr/bin/env python
# coding: utf-8

# In[26]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import plotly.graph_objs as go
import plotly .offline as offline
import plotly.figure_factory as ff
import seaborn as sns


# In[27]:



# Importing dataset and examining it
Estay = pd.read_csv("EireStay.csv")
print(Estay.head())
print(Estay.shape)
print(Estay.info())
print(Estay.describe())


# In[28]:



# Converting Categorical features into Numerical features
Estay['meal'] = Estay['meal'].map({'BB':0, 'FB':1, 'HB':2,'SC':3, 'Undefined':4 })
Estay['market_segment'] = Estay['market_segment'].map({'Direct':0, 'Online TA':1})
Estay['reserved_room_type'] = Estay['reserved_room_type'].map({'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7, 'I':8})
Estay['deposit_type'] = Estay['deposit_type'].map({'No Deposit':0, 'Non Refund':1, 'Refundable': 2})

print(Estay.info())


# In[4]:


# Plotting Correlation Heatmap
corrs = Estay.corr()
figure = ff.create_annotated_heatmap(
    z=corrs.values,
    x=list(corrs.columns),
    y=list(corrs.index),
    annotation_text=corrs.round(2).values,
    showscale=True)
offline.plot(figure,filename='corrheatmap.html')


# In[29]:


#Lead_time = Lead
#stays_in_weekend_nights : swn
#stays_in_week_nights : swd
#adults = adt
# children = cdn
# babies = bby
# meal = meal
# market_segment = mkt
# previous_stays = pst
# reserved_room_type = rrt
# booking_changes= bc
# deposit_type= dt
# days_in_waiting_list= diwl
# average_daily_rate = adr
# total_of_special_requests = tosr
# Dividing data into subsets 
#Subset 1
subset1 = Estay[['adults','children','babies','stays_in_week_nights', 'stays_in_weekend_nights','reserved_room_type']]

#Subset 2
subset2 = Estay[['adults','previous_stays','meal','deposit_type']]

#Subset 3
subset4 = Estay[['adults', 'market_segment', 'stays_in_weekend_nights', 'lead_time']]


# In[30]:


# Normalizing numerical features so that each feature has mean 0 and variance 1
feature_scaler = StandardScaler()
X1 = feature_scaler.fit_transform(subset1)
X2 = feature_scaler.fit_transform(subset2)
X3 = feature_scaler.fit_transform(subset3)


# In[31]:


# Analysis on subset1 
# Finding the number of clusters (K) - Elbow Plot Method
inertia = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, random_state = 100)
    kmeans.fit(X1)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia)
plt.title('The Elbow Plot')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()


# In[32]:



# Running KMeans to generate labels
kmeans = KMeans(n_clusters = 6)
kmeans.fit(X1)


# In[ ]:


#Subset 1

# Implementing t-SNE to visualize dataset
tsne = TSNE(n_components = 2, perplexity =50,n_iter=2000)
x_tsne = tsne.fit_transform(X1)

adt = list(Estay['adults'])
cdn = list(Estay['children'])
bby = list(Estay['babies'])
swd = list(Estay['stays_in_week_nights'])
swn = list(Estay['stays_in_weekend_nights'])
rrt = list(Estay['reserved_room_type'])

data = [go.Scatter(x=x_tsne[:,0], y=x_tsne[:,1], mode='markers',
                    marker = dict(color= kmeans.labels_, colorscale='Rainbow', opacity=0.5),
                                text=[f'adults: {a}; children: {b}; babies:{c}, stays_in_week_nights:{d}, stays_in_weekend_nights: {e}, reserved_room_type {f}' for a,b,c,d,e,f in list(zip(adt,cdn,bby,swd,swn,rrt))],
                                hoverinfo='text')]

layout = go.Layout(title = 't-SNE Dimensionality Reduction', width = 700, height = 700,
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)


offline.plot(fig,filename='t-SNE1.html')


# In[20]:


# Analysis on subset2 
# Finding the number of clusters (K) - Elbow Plot Method
inertia = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, random_state = 100)
    kmeans.fit(X2)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia)
plt.title('The Elbow Plot')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()


# In[21]:


# Running KMeans to generate labels
kmeans = KMeans(n_clusters = 6)
kmeans.fit(X2)


# In[22]:


#Subset 2


# Implementing t-SNE to visualize dataset
tsne = TSNE(n_components = 2, perplexity =5,n_iter=2000)
x_tsne = tsne.fit_transform(X2)

adt = list(Estay['adults'])
pst = list(Estay['previous_stays'])
meal = list(Estay['meal'])
dt = list(Estay['deposit_type'])
data = [go.Scatter(x=x_tsne[:,0], y=x_tsne[:,1], mode='markers',
                    marker = dict(color=kmeans.labels_, colorscale='Rainbow', opacity=0.5),
                                text=[f':adults {a}; previous_stays:{b},meal :{c}, deposit_type:{d}' for a,b,c,d in list(zip(adt,pst,meal,dt))],
                                hoverinfo='text')]

layout = go.Layout(title = 't-SNE Dimensionality Reduction', width = 1000, height = 700,
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='t-SNE2.html')


# In[23]:


#analysing subset 3
# Finding the number of clusters (K) - Elbow Plot Method
inertia = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, random_state = 100)
    kmeans.fit(X3)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia)
plt.title('The Elbow Plot')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()


# In[24]:


# Running KMeans to generate labels
kmeans = KMeans(n_clusters = 5)
kmeans.fit(X3)


# In[25]:


#Subset 3



# Implementing t-SNE to visualize dataset
tsne = TSNE(n_components = 2, perplexity =5,n_iter=2000)
x_tsne = tsne.fit_transform(X4)

adt = list(Estay['adults'])
mkt = list(Estay['market_segment'])
swn = list(Estay['stays_in_weekend_nights'])
Lead = list(Estay['lead_time'])
data = [go.Scatter(x=x_tsne[:,0], y=x_tsne[:,1], mode='markers',
                    marker = dict(color=kmeans.labels_, colorscale='Rainbow', opacity=0.5),
                                text=[f':adults {a}; market_segment: {b}; stays_in_weekend_nights:{c},lead_time :{d},' for a,b,c,d in list(zip(adt,mkt,swn,Lead,))],
                                hoverinfo='text')]

layout = go.Layout(title = 't-SNE Dimensionality Reduction', width = 1000, height = 700,
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='t-SNE3.html')


# In[ ]:




