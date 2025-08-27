import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer, InterclusterDistance
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

df = pd.read_csv("29-country_data.csv")
countries = df['country']
df = df.drop("country", axis=1)

# Veriyi ölçekle
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df)

# 1. Dirsek Yöntemi ile Optimal Küme Sayısını Bulma
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
visualizer = KElbowVisualizer(KMeans(random_state=42), k=(2,11), metric='distortion', timings=False)
visualizer.fit(X_scaled)
visualizer.finalize()
plt.title('KMeans - Dirsek Yöntemi')

plt.subplot(1, 3, 2)
visualizer = KElbowVisualizer(KMeans(random_state=42), k=(2,11), metric='silhouette', timings=False)
visualizer.fit(X_scaled)
visualizer.finalize()
plt.title('KMeans - Silhouette Yöntemi')

plt.subplot(1, 3, 3)
visualizer = KElbowVisualizer(KMeans(random_state=42), k=(2,11), metric='calinski_harabasz', timings=False)
visualizer.fit(X_scaled)
visualizer.finalize()
plt.title('KMeans - Calinski-Harabasz')

plt.tight_layout()
plt.show()

# 2. Farklı Algoritmaları Karşılaştırma
algorithms = {
    'KMeans': KMeans(n_clusters=3, random_state=42, n_init=10),
    'DBSCAN': DBSCAN(eps=0.2, min_samples=4),
    'Agglomerative': AgglomerativeClustering(n_clusters=3),
    'GaussianMixture': GaussianMixture(n_components=3, random_state=42)
}

results = []

for name, model in algorithms.items():
    try:
        if name == 'GaussianMixture':
            # GMM için predict kullanıyoruz
            labels = model.fit_predict(X_scaled)
        else:
            labels = model.fit_predict(X_scaled)
        
        # Gürültü noktalarını hariç tut (DBSCAN için)
        valid_mask = labels != -1
        X_valid = X_scaled[valid_mask]
        labels_valid = labels[valid_mask]
        
        if len(np.unique(labels_valid)) > 1:  # En az 2 küme olmalı
            silhouette = silhouette_score(X_valid, labels_valid)
            db_score = davies_bouldin_score(X_valid, labels_valid)
            ch_score = calinski_harabasz_score(X_valid, labels_valid)
        else:
            silhouette = db_score = ch_score = np.nan
        
        n_clusters = len(np.unique(labels_valid))
        n_noise = np.sum(labels == -1)
        
        results.append({
            'Algorithm': name,
            'Silhouette': silhouette,
            'Davies-Bouldin': db_score,
            'Calinski-Harabasz': ch_score,
            'Clusters': n_clusters,
            'Noise_Points': n_noise
        })
        
    except Exception as e:
        print(f"{name} hatası: {e}")

# Sonuçları DataFrame'e dönüştür
results_df = pd.DataFrame(results)
print("\nAlgoritma Karşılaştırması:")
print(results_df)

# 3. Silhouette Skorlarını Görselleştirme
plt.figure(figsize=(12, 8))

for i, (name, model) in enumerate(algorithms.items(), 1):
    try:
        plt.subplot(2, 2, i)
        
        if name == 'GaussianMixture':
            labels = model.fit_predict(X_scaled)
        else:
            labels = model.fit_predict(X_scaled)
        
        # Gürültü noktalarını hariç tut
        valid_mask = labels != -1
        X_valid = X_scaled[valid_mask]
        labels_valid = labels[valid_mask]
        
        if len(np.unique(labels_valid)) > 1:
            visualizer = SilhouetteVisualizer(model, colors='yellowbrick')
            visualizer.fit(X_valid)
            visualizer.finalize()
            plt.title(f'{name} - Silhouette Plot')
        
    except Exception as e:
        print(f"{name} silhouette hatası: {e}")

plt.tight_layout()
plt.show()

# 4. Intercluster Distance Maps
plt.figure(figsize=(15, 10))

for i, (name, model) in enumerate(algorithms.items(), 1):
    try:
        plt.subplot(2, 2, i)
        
        if name == 'GaussianMixture':
            labels = model.fit_predict(X_scaled)
        else:
            labels = model.fit_predict(X_scaled)
        
        # Gürültü noktalarını hariç tut
        valid_mask = labels != -1
        X_valid = X_scaled[valid_mask]
        labels_valid = labels[valid_mask]
        
        if len(np.unique(labels_valid)) > 1:
            visualizer = InterclusterDistance(model)
            visualizer.fit(X_valid)
            visualizer.finalize()
            plt.title(f'{name} - Intercluster Distance')
        
    except Exception as e:
        print(f"{name} intercluster distance hatası: {e}")

plt.tight_layout()
plt.show()

# 5. Metrik Karşılaştırma Grafiği
metrics = ['Silhouette', 'Davies-Bouldin', 'Calinski-Harabasz']
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, metric in enumerate(metrics):
    axes[i].bar(results_df['Algorithm'], results_df[metric])
    axes[i].set_title(f'{metric} Score Comparison')
    axes[i].tick_params(axis='x', rotation=45)
    
    # Davies-Bouldin için düşük değerler daha iyidir
    if metric == 'Davies-Bouldin':
        axes[i].invert_yaxis()

plt.tight_layout()
plt.show()

# En iyi algoritmayı belirle
best_silhouette = results_df.loc[results_df['Silhouette'].idxmax(), 'Algorithm']
best_db = results_df.loc[results_df['Davies-Bouldin'].idxmin(), 'Algorithm']
best_ch = results_df.loc[results_df['Calinski-Harabasz'].idxmax(), 'Algorithm']

print(f"\nEn İyi Sonuçlar:")
print(f"Silhouette: {best_silhouette}")
print(f"Davies-Bouldin: {best_db}")
print(f"Calinski-Harabasz: {best_ch}")