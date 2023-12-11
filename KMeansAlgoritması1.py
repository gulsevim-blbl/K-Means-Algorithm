import pandas as pd
from sklearn.cluster import KMeans
import  matplotlib.pyplot as plt

# Veriyi yükle
data = pd.read_csv("data_file.csv")

# Kullanılacak değişkenleri seç
X = data[['x1', 'x2', 'x3', 'x4', 'x5']]

# K-means modelini oluştur
kmeans = KMeans(n_clusters=2)  # Kümelerin sayısını isteğinize göre ayarlayabilirsiniz

# Modeli eğit
kmeans.fit(X)

# Kümeleri tahmin et
labels = kmeans.predict(X)

# Elde edilen küme etiketlerini veri çerçevesine ekle
data['Cluster'] = labels

# Küme merkezlerini al
centers = kmeans.cluster_centers_

# Sonuçları yazdır
print("Küme Merkezleri:")
print(centers)
print("\nKümeleme Sonuçları:")
print(data)

# Kümeleme sonuçlarını görselleştirme (2D olarak varsayılmıştır)
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], marker='X', s=200, linewidths=3, color='r')
plt.title('K-means Kümeleme')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()


#pandas numpy sklearn kütüphanesini ekledim  (pip install pandas numpy sklearn)
