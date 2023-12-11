import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Veri seti
gözlemler = np.array([4, 6, 5, 10, 11])
değişken1 = np.array([4, 6, 5, 10, 11])
değişken2 = np.array([2, 4, 1, 6, 8])

# Veriyi düzenleme
X = np.array(list(zip(değişken1, değişken2)))

# K-means modelini oluşturma
kmeans = KMeans(n_clusters=2)  # Küme sayısını isteğinize göre ayarlayabilirsiniz

# Modeli eğitme
kmeans.fit(X)

# Merkezleri ve küme etiketlerini alıyoruz
merkezler = kmeans.cluster_centers_
küme_etiketleri = kmeans.labels_

# Sonuçları yazdırma
print("Küme Merkezleri:")
print(merkezler)
print("\nGözlemler ve Küme Etiketleri:")
for i in range(len(gözlemler)):
    print(f"Gözlem {gözlemler[i]} => Küme {küme_etiketleri[i]}")

# Sonuçları görselleştirme
plt.scatter(değişken1, değişken2, c=küme_etiketleri, cmap='viridis')
plt.scatter(merkezler[:, 0], merkezler[:, 1], marker='X', s=200, c='red')
plt.title('K-means Kümeleme')
plt.xlabel('Değişken 1')
plt.ylabel('Değişken 2')
plt.show()
 

#n_clusters parametresi, küme sayısını belirtir ve bu değeri ihtiyaca göre değiştirebiliriz.
#  Sonuçlar, küme merkezleri ve her bir gözlem için atanmış küme etiketleri olarak ekrana yazdırılır. 
# Ayrıca, kümeleme sonuçlarını görselleştirmek için matplotlib kullanılmıştır.

