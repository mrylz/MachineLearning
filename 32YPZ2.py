import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Veriyi yükleme
df = pd.read_csv("29-country_data.csv")

# Ülke isimlerini ayırma ve ölçekleme
countries = df['country']
df_numeric = df.drop("country", axis=1)

# Ölçekleme
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df_numeric)

# PCA uygulama ve optimal bileşen sayısını belirleme
pca = PCA()
pca_result = pca.fit_transform(df_scaled)

# Varyans açıklama oranlarına bakalım
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# Grafikle varyansı görselleştirme
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(explained_variance)+1), explained_variance, alpha=0.5, 
        align='center', label='Bireysel açıklanan varyans')
plt.step(range(1, len(cumulative_variance)+1), cumulative_variance, where='mid',
         label='Kümülatif açıklanan varyans')
plt.ylabel('Açıklanan Varyans Oranı')
plt.xlabel('PCA Bileşenleri')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# Toplam varyansın %85-95'ini açıklayan bileşenleri seç
n_components = np.where(cumulative_variance >= 0.85)[0][0] + 1
print(f"Seçilen bileşen sayısı: {n_components} (%{cumulative_variance[n_components-1]*100:.2f} varyans)")

pca_df = pd.DataFrame(pca_result[:, :n_components])

# Optimal küme sayısını belirleme
wcss = []
silhouette_scores = []
k_range = range(2, 11)  # 2'den 10'a kadar küme sayıları

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(pca_df)
    wcss.append(kmeans.inertia_)
    
    # Silhouette skoru sadece 2 veya daha fazla küme için hesaplanabilir
    silhouette_scores.append(silhouette_score(pca_df, kmeans.labels_))

# Dirsek yöntemi ve silhouette skoruna göre optimal küme sayısı
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(k_range, wcss, 'bx-')
ax1.set_xlabel('Küme sayısı')
ax1.set_ylabel('WCSS')
ax1.set_title('Dirsek Yöntemi')

ax2.plot(k_range, silhouette_scores, 'bx-')
ax2.set_xlabel('Küme sayısı')
ax2.set_ylabel('Silhouette Skoru')
ax2.set_title('Silhouette Yöntemi')

plt.show()

# Optimal küme sayısına karar ver (örneğin 3 seçelim)
optimal_clusters = 3
model = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
model.fit(pca_df)
labels = model.labels_

# Kümeleri analiz ederek anlamlı isimler verme
df_result = df_numeric.copy()
df_result['Cluster'] = labels

# Kümelerin özelliklerini analiz etme
cluster_means = df_result.groupby('Cluster').mean()
print("Küme Ortalamaları:")
print(cluster_means)

# Küme özelliklerine göre anlamlı isimler belirleme
# Örnek: Düşük gelir, yüksek çocuk ölümü -> "Yardım Gerekli"
cluster_names = {}
for cluster in range(optimal_clusters):
    if cluster_means.loc[cluster, 'gdpp'] < cluster_means['gdpp'].mean() and \
       cluster_means.loc[cluster, 'child_mort'] > cluster_means['child_mort'].mean():
        cluster_names[cluster] = "Budget Needed"
    elif cluster_means.loc[cluster, 'gdpp'] > cluster_means['gdpp'].mean() and \
         cluster_means.loc[cluster, 'child_mort'] < cluster_means['child_mort'].mean():
        cluster_names[cluster] = "No Budget Needed"
    else:
        cluster_names[cluster] = "In Between"

df_result['Class'] = df_result['Cluster'].map(cluster_names)

# Görselleştirme için hazırlık
viz_df = pd.DataFrame({
    'Country': countries,
    'Class': df_result['Class']
})

# Harita görselleştirme
fig = px.choropleth(
    viz_df,
    locationmode="country names",
    locations="Country",
    title="Needed Budget by Country",
    color="Class",
    color_discrete_map={
        "Budget Needed": "Red",
        "In Between": "Yellow",
        "No Budget Needed": "Green"
    })
fig.update_geos(fitbounds="locations", visible=True)
fig.show()

# PCA sonuçlarını 2D/3D görselleştirme (ilk 2-3 bileşenle)
if n_components >= 2:
    pca_viz_df = pd.DataFrame(pca_result[:, :2], columns=['PC1', 'PC2'])
    pca_viz_df['Country'] = countries
    pca_viz_df['Class'] = df_result['Class']
    
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=pca_viz_df, x='PC1', y='PC2', hue='Class', 
                    palette={"Budget Needed": "red", "In Between": "yellow", "No Budget Needed": "green"})
    plt.title('Ülkelerin PCA ve Küme Dağılımı')
    plt.show()