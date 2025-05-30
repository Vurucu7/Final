import json
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

# JSON dosyasının yolu
json_path = 'drug-event-0016-of-0034.json'  # Dosyan aynı klasördeyse böyle bırakabilirsin

# JSON dosyasını oku
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# reactionmeddrapt (yan etki açıklamaları) metinlerini topla
all_texts = []
for entry in data.get("results", []):
    reactions = entry.get("patient", {}).get("reaction", [])
    for reaction in reactions:
        text = reaction.get("reactionmeddrapt", "")
        all_texts.append(text.lower())

# Basit tokenizasyon (kelimeye ayırma)
all_tokens = []
for phrase in all_texts:
    tokens = phrase.split()  # boşluk bazlı ayır
    all_tokens.extend(tokens)

# Kelime frekanslarını say
word_freq = Counter(all_tokens)

# En sık geçen ilk 1000 kelimeyi al
most_common = word_freq.most_common(1000)
ranks = np.arange(1, len(most_common) + 1)
frequencies = np.array([freq for word, freq in most_common])

# Zipf yasası log-log grafiği çiz
plt.figure(figsize=(10, 6))
plt.plot(np.log(ranks), np.log(frequencies), marker='o', linestyle='-', color='darkblue')
plt.title("Zipf Yasası - Yan Etki Açıklamaları", fontsize=14)
plt.xlabel("log(Rank)", fontsize=12)
plt.ylabel("log(Frequency)", fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()
