import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

# Dosyaları yükle
df_lemma = pd.read_csv("lemmatized_output.csv")
df_stem = pd.read_csv("stemmed_output.csv")

# Kelime listelerini birleştir
all_lemma_words = " ".join(df_lemma['Lemmatized_Text']).split()
all_stem_words = " ".join(df_stem['Stemmed_Text']).split()

# Frekans sayımları
lemma_counts = Counter(all_lemma_words)
stem_counts = Counter(all_stem_words)

# Sıralı frekans listeleri (yüksekten düşüğe)
lemma_freq = sorted(lemma_counts.values(), reverse=True)
stem_freq = sorted(stem_counts.values(), reverse=True)

# Log-log grafiği çiz
plt.figure(figsize=(12, 6))

# Lemma grafiği
plt.subplot(1, 2, 1)
plt.loglog(range(1, len(lemma_freq)+1), lemma_freq, marker='o', color='blue')
plt.title("Zipf Grafiği - Lemmatized")
plt.xlabel("Kelime Sırası (log)")
plt.ylabel("Frekans (log)")

# Stem grafiği
plt.subplot(1, 2, 2)
plt.loglog(range(1, len(stem_freq)+1), stem_freq, marker='o', color='green')
plt.title("Zipf Grafiği - Stemmed")
plt.xlabel("Kelime Sırası (log)")
plt.ylabel("Frekans (log)")

plt.tight_layout()
plt.savefig("zipf_graphs_lemma_stem.png", dpi=300)
plt.show()
