import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Outlier tespiti için fonksiyon
def detect_outliers(df, method='iqr', threshold=1.5):
    """
    DataFrame'deki outlier'ları tespit eder
    
    Parameters:
    df: DataFrame
    method: 'iqr' (IQR method) veya 'zscore' (Z-score method)
    threshold: Outlier eşik değeri
    
    Returns:
    DataFrame with outliers and their details
    """
    
    outliers_list = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        data = df[col].dropna()
        
        if method == 'iqr':
            # IQR method
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            
        elif method == 'zscore':
            # Z-score method
            z_scores = np.abs((data - data.mean()) / data.std())
            outliers = df[z_scores > threshold]
        
        # Outlier'ları listeye ekle
        for idx in outliers.index:
            outliers_list.append({
                'Row_Index': idx,
                'Column': col,
                'Outlier_Value': df.loc[idx, col],
                'Method': method,
                'Threshold': threshold
            })
    
    return pd.DataFrame(outliers_list)

# Outlier'ları görselleştirme fonksiyonu
def plot_outliers(df):
    """Outlier'ları boxplot ile görselleştirir"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    fig, axes = plt.subplots(nrows=len(numeric_cols), ncols=1, 
                           figsize=(12, 4*len(numeric_cols)))
    
    if len(numeric_cols) == 1:
        axes = [axes]
    
    for i, col in enumerate(numeric_cols):
        axes[i].boxplot(df[col].dropna())
        axes[i].set_title(f'Boxplot of {col}')
        axes[i].set_ylabel('Values')
    
    plt.tight_layout()
    plt.show()

# Tüm outlier'ları detaylı raporla
def comprehensive_outlier_report(df):
    """Kapsamlı outlier raporu oluşturur"""
    
    print("=" * 60)
    print("KAPSAMLI OUTLIER RAPORU")
    print("=" * 60)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        data = df[col].dropna()
        
        # İstatistikler
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Outlier sayıları
        lower_outliers = len(df[df[col] < lower_bound])
        upper_outliers = len(df[df[col] > upper_bound])
        total_outliers = lower_outliers + upper_outliers
        
        print(f"\n📊 {col}:")
        print(f"   Toplam veri: {len(data)}")
        print(f"   Outlier sayısı: {total_outliers} ({total_outliers/len(data)*100:.1f}%)")
        print(f"   Alt sınır: {lower_bound:.2f}, Üst sınır: {upper_bound:.2f}")
        print(f"   Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
        
        if total_outliers > 0:
            print(f"   Outlier değerleri: {df[df[col] < lower_bound][col].tolist() + df[df[col] > upper_bound][col].tolist()}")

