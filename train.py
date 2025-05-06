# Gerekli kütüphanelerin yüklenmesi
import numpy as np  # Sayısal işlemler için
import pandas as pd  # Veri işleme için
from sklearn.cluster import KMeans  # Kümeleme algoritması
from sklearn.linear_model import Perceptron  # Sınıflandırma modeli
from sklearn.preprocessing import StandardScaler  # Veri ölçeklendirme
from sklearn.model_selection import train_test_split  # Veri bölme
from sklearn.metrics import accuracy_score  # Doğruluk hesaplama
import os  # Dosya işlemleri için

def load_data(directory):
    """CSV dosyalarından veri yükleme fonksiyonu"""
    all_data = []
    # Dizindeki tüm CSV dosyalarını bul
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            try:
                file_path = os.path.join(directory, filename)
                print(f"Loading {filename}...")
                # CSV dosyasını oku
                df = pd.read_csv(file_path)
                # Dosya adını tanımlayıcı olarak ekle
                df['identifier'] = filename.replace('.csv', '')
                all_data.append(df)
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")
    
    if not all_data:
        raise ValueError("No valid CSV files found in the directory")
    
    # Tüm verileri birleştir
    return pd.concat(all_data, ignore_index=True)

def preprocess_data(data):
    """Veriyi ön işleme fonksiyonu"""
    # Sensör sütunlarını seç (ivmeölçer ve jiroskop verileri)
    feature_columns = [col for col in data.columns if ('acc' in col.lower() or 'gyro' in col.lower()) and col not in ['identifier', 'activity']]
    
    if not feature_columns:
        raise ValueError("No relevant sensor columns found in the data")
    
    print(f"Selected features: {feature_columns}")
    
    # Sensör verilerini al
    X = data[feature_columns].values
    
    # Hareket yoğunluğunu hesapla (RMS yöntemi)
    movement_intensity = np.sqrt(np.mean(X**2, axis=1))
    
    # Hareket değişkenliğini hesapla
    movement_variability = np.std(X, axis=1)
    
    # Yoğunluk ve değişkenliği özellik olarak ekle
    X = np.column_stack((X, movement_intensity, movement_variability))
    
    # Verileri ölçeklendir
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, data['identifier'].values, feature_columns

def determine_active_cluster(X, clusters):
    """Aktif davranışı temsil eden kümeyi belirleme"""
    # Son iki sütunu al (hareket yoğunluğu ve değişkenliği)
    movement_features = X[:, -2:]
    
    # Her küme için ortalama değerleri hesapla
    cluster_0_stats = np.mean(movement_features[clusters == 0], axis=0)
    cluster_1_stats = np.mean(movement_features[clusters == 1], axis=0)
    
    # Birleşik skor hesapla (yoğunluk * değişkenlik)
    cluster_0_score = cluster_0_stats[0] * cluster_0_stats[1]
    cluster_1_score = cluster_1_stats[0] * cluster_1_stats[1]
    
    # Daha yüksek skora sahip kümeyi aktif olarak belirle
    active_cluster = 1 if cluster_1_score > cluster_0_score else 0
    
    return active_cluster

def train_model(X, identifiers):
    """K-Means ve Perceptron modellerini eğitme"""
    # K-Means kümeleme uygula
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10, max_iter=300)
    clusters = kmeans.fit_predict(X)
    
    # Aktif davranışı temsil eden kümeyi belirle
    active_cluster = determine_active_cluster(X, clusters)
    
    # Perceptron için etiketleri oluştur (1: hedef kişi, 0: diğerleri)
    target_person = identifiers[0]
    y = (identifiers == target_person).astype(int)
    
    # Veriyi eğitim ve test olarak böl
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Perceptron modelini eğit
    perceptron = Perceptron(random_state=42, max_iter=2000, tol=1e-3)
    perceptron.fit(X_train, y_train)
    
    # Tahminler yap
    y_pred = perceptron.predict(X_test)
    
    # Doğruluk oranını hesapla
    accuracy = accuracy_score(y_test, y_pred)
    
    # Her kişi için durumu hesapla
    status_results = {}
    unique_identifiers = set(identifiers)
    for identifier in unique_identifiers:
        mask = identifiers == identifier
        intensity = np.mean(X[mask, -2])
        variability = np.mean(X[mask, -1])
        activity_score = intensity * variability
        # Aktif/pasif durumunu belirle
        status = "Aktif" if (activity_score > 0.5 or intensity > 1.5 or variability > 1.5) else "Pasif"
        status_results[identifier] = status
    
    return {
        'accuracy': accuracy,
        'status_results': status_results
    }

def run_training():
    """Tüm eğitim sürecini çalıştırma fonksiyonu"""
    try:
        # Veriyi yükle
        print("Veri yükleniyor...")
        data = load_data('archive')
        
        # Veriyi ön işle
        print("Veri ön işleme yapılıyor...")
        X, identifiers, feature_columns = preprocess_data(data)
        
        # Modelleri eğit ve sonuçları al
        results = train_model(X, identifiers)
        
        return results
        
    except Exception as e:
        print(f"Eğitim sırasında hata oluştu: {str(e)}")
        return None 