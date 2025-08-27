import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

df = pd.read_csv("29-country_data.csv")
countries = df['country']
df = df.drop("country", axis=1)

# Önce veriyi ölçekle
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df)

# Tüm analizleri ölçeklenmiş veri ile yap
wcss = []
silhouette_scores = []
db_scores = []
ch_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
    
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    db_scores.append(davies_bouldin_score(X_scaled, kmeans.labels_))
    ch_scores.append(calinski_harabasz_score(X_scaled, kmeans.labels_))

# Görselleştirme
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

ax1.plot(k_range, wcss, 'bx-')
ax1.set_xlabel('Küme sayısı')
ax1.set_ylabel('WCSS')
ax1.set_title('Dirsek Yöntemi')

ax2.plot(k_range, silhouette_scores, 'bx-')
ax2.set_xlabel('Küme sayısı')
ax2.set_ylabel('Silhouette Skoru')
ax2.set_title('Silhouette Yöntemi')

ax3.plot(k_range, db_scores, 'bx-')
ax3.set_xlabel('Küme sayısı')
ax3.set_ylabel('Davies-Bouldin Skoru')
ax3.set_title('Davies-Bouldin Yöntemi')

ax4.plot(k_range, ch_scores, 'bx-')
ax4.set_xlabel('Küme sayısı')
ax4.set_ylabel('Calinski-Harabasz Skoru')
ax4.set_title('Calinski-Harabasz Yöntemi')

plt.tight_layout()
plt.show()

# Optimal küme sayısını otomatik belirleme (örneğin silhouette skoru maksimize eden)
optimal_clusters = k_range[np.argmax(silhouette_scores)]
print(f"Optimal küme sayısı: {optimal_clusters}")

# Final model
model = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
y_cls = model.fit_predict(X_scaled)

# Metrikleri yazdır
print(f"Silhouette Score: {silhouette_score(X_scaled, y_cls):.4f}")
print(f"Davies-Bouldin Score: {davies_bouldin_score(X_scaled, y_cls):.4f}")
print(f"Calinski-Harabasz Score: {calinski_harabasz_score(X_scaled, y_cls):.4f}")

# Sonuçları dataframe'e ekle
results_df = pd.DataFrame({
    'country': countries,
    'cluster': y_cls
})
print(results_df.head())