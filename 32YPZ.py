import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
df = pd.read_csv("29-country_data.csv")
df2 = df.drop("country", axis=1)
scaler = MinMaxScaler()
df2 = scaler.fit_transform(df2)
df2 = pd.DataFrame(df2, columns=['child_mort', 'exports', 'health', 'imports', 'income',
       'inflation', 'life_expec', 'total_fer', 'gdpp'])
pca = PCA()
pca_df2 = pd.DataFrame(pca.fit_transform(df2))
pca.explained_variance_ratio_
pca_df2 = pca_df2.drop(columns = [3,4,5,6,7,8])
wcss = []
for k in range(1,11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pca_df2)
    wcss.append(kmeans.inertia_)
model = KMeans(n_clusters=3)
model.fit(pca_df2)
labels = model.labels_
silhouette_score(pca_df2,labels)
df['Class'] = labels
pca_df2.insert(0, column= "Country", value = df['country'])
pca_df2['Class'] = labels
pca_df2.loc[pca_df2['Class'] == 1, 'Class'] = "Budget Needed"
pca_df2.loc[pca_df2['Class'] == 2, 'Class'] = "In Between"
pca_df2.loc[pca_df2['Class'] == 0, 'Class'] = "No Budget Needed"
fig = px.choropleth(
    pca_df2[['Country', 'Class']],
    locationmode = "country names",
    locations = "Country",
    title = "Needed Budget by Country",
    color = pca_df2['Class'],
    color_discrete_map= {
                        "Budget Needed" : "Red",
                        "In Between" : "Yellow",
                        "No Budget Needed": "Green"
    })
fig.update_geos(fitbounds = "locations", visible = True)
fig.show()







