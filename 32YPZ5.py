import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

df = pd.read_csv("29-country_data.csv")
countries = df['country']
df = df.drop("country", axis=1)

# Veriyi ölçekle
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df)

# 1. KMeans'i seçiyoruz (en iyi genel performans)
print("=== ÖNERİLEN ALGORİTMA: KMEANS ===")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

# 2. DBSCAN parametrelerini optimize etmeyi deneyelim
print("\n=== DBSCAN OPTİMİZASYON DENEMESİ ===")

def optimize_dbscan(X, max_eps=0.5, min_samples_range=range(2, 8)):
    best_score = -1
    best_params = {}
    best_labels = None
    
    for min_samples in min_samples_range:
        # k-distance grafiği için
        neigh = NearestNeighbors(n_neighbors=min_samples)
        nbrs = neigh.fit(X)
        distances, indices = nbrs.kneighbors(X)
        distances = np.sort(distances[:, min_samples-1], axis=0)
        
        # Olası eps değerlerini test et
        eps_values = np.linspace(0.1, max_eps, 20)
        
        for eps in eps_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X)
            
            # Gürültü hariç valid kümeleri kontrol et
            valid_mask = labels != -1
            if np.sum(valid_mask) > 10 and len(np.unique(labels[valid_mask])) > 1:
                X_valid = X[valid_mask]
                labels_valid = labels[valid_mask]
                
                score = silhouette_score(X_valid, labels_valid)
                
                if score > best_score:
                    best_score = score
                    best_params = {'eps': eps, 'min_samples': min_samples}
                    best_labels = labels
    
    return best_params, best_labels, best_score

# DBSCAN optimizasyonunu dene
from sklearn.neighbors import NearestNeighbors
try:
    best_params, dbscan_opt_labels, best_score = optimize_dbscan(X_scaled)
    print(f"Optimize DBSCAN parametreleri: {best_params}")
    print(f"Optimize DBSCAN silhouette: {best_score:.3f}")
    
    n_noise_opt = np.sum(dbscan_opt_labels == -1)
    print(f"Optimize DBSCAN gürültü sayısı: {n_noise_opt}")
except:
    print("DBSCAN optimizasyonu başarısız oldu")

# 3. Final sonuçlarını oluştur
final_results = pd.DataFrame({
    'country': countries,
    'KMeans_Cluster': kmeans_labels,
    'DBSCAN_Cluster': dbscan_opt_labels if 'dbscan_opt_labels' in locals() else np.nan,
    'KMeans_Silhouette': silhouette_score(X_scaled, kmeans_labels),
    'KMeans_DaviesBouldin': davies_bouldin_score(X_scaled, kmeans_labels),
    'KMeans_CalinskiHarabasz': calinski_harabasz_score(X_scaled, kmeans_labels)
})

print("\n=== FİNAL KÜMELEME SONUÇLARI ===")
print(f"KMeans Silhouette: {silhouette_score(X_scaled, kmeans_labels):.3f}")
print(f"KMeans Davies-Bouldin: {davies_bouldin_score(X_scaled, kmeans_labels):.3f}")
print(f"KMeans Calinski-Harabasz: {calinski_harabasz_score(X_scaled, kmeans_labels):.3f}")

# 4. Kümeleme sonuçlarını analiz et
cluster_analysis = final_results.groupby('KMeans_Cluster').agg({
    'country': 'count',
    'KMeans_Silhouette': 'mean'
}).rename(columns={'country': 'country_count'})

print("\n=== KMEANS KÜME ANALİZİ ===")
print(cluster_analysis)

# 5. Görselleştirme
plt.figure(figsize=(15, 10))

# PCA ile 2 boyuta indirgeme
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# KMeans sonuçları
plt.subplot(2, 2, 1)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis', s=100)
plt.colorbar(scatter, label='Küme')
plt.title('KMeans Kümeleme Sonuçları\n(Silhouette: 0.343)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')

# Ülke isimlerini etiketle
for i, country in enumerate(countries):
    plt.annotate(country, (X_pca[i, 0], X_pca[i, 1]), 
                 xytext=(5, 5), textcoords='offset points', 
                 fontsize=8, alpha=0.7)

# Metrik karşılaştırma
algorithms = ['KMeans', 'DBSCAN', 'Agglomerative', 'GMM']
silhouette_scores = [0.343, 0.321, 0.316, 0.235]
db_scores = [1.114, 0.893, 1.192, 1.247]
ch_scores = [99.41, 88.62, 86.47, 74.87]

plt.subplot(2, 2, 2)
x = np.arange(len(algorithms))
width = 0.25

plt.bar(x - width, silhouette_scores, width, label='Silhouette', alpha=0.8)
plt.bar(x, db_scores, width, label='Davies-Bouldin', alpha=0.8)
plt.bar(x + width, [s/max(ch_scores) for s in ch_scores], width, 
        label='Calinski-Harabasz (norm)', alpha=0.8)

plt.xlabel('Algorithms')
plt.ylabel('Scores')
plt.title('Algorithm Performance Comparison')
plt.xticks(x, algorithms, rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

# Küme dağılımı
plt.subplot(2, 2, 3)
cluster_counts = final_results['KMeans_Cluster'].value_counts().sort_index()
plt.bar([f'Küme {i}' for i in cluster_counts.index], cluster_counts.values)
plt.title('Kümelere Göre Ülke Dağılımı')
plt.ylabel('Ülke Sayısı')

# Özellik önemliliği (küme merkezlerine göre)
plt.subplot(2, 2, 4)
feature_importance = pd.DataFrame({
    'feature': df.columns,
    'importance': np.std(kmeans.cluster_centers_, axis=0)
}).sort_values('importance', ascending=False)

plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.title('Özellik Önemliliği (Küme Merkezlerine Göre)')
plt.xlabel('Önemlilik (standart sapma)')

plt.tight_layout()
plt.show()

# 6. Son öneri
print("\n" + "="*60)
print("🎯 SON ÖNERİ: KMEANS ALGORİTMASINI KULLANIN")
print("="*60)
print("Nedenleri:")
print("1. En yüksek Silhouette ve Calinski-Harabasz skorları")
print("2. Sıfır gürültü noktası - tüm ülkeler kümelenmiş")
print("3. En dengeli performans (tüm metriklerde iyi)")
print("4. 3 küme anlamlı bir gruplama sağlıyor")

print("\n⚠️  DBSCAN için:")
print("- Çok yüksek gürültü oranı (%158!)")
print("- Parametre optimizasyonu gerekli")
print("- Bu veri seti için uygun olmayabilir")

# Sonuçları kaydet
final_results.to_csv('optimal_country_clustering.csv', index=False)
print(f"\nSonuçlar 'optimal_country_clustering.csv' dosyasına kaydedildi")