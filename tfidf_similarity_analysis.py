import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os

# Dosya yolunu kendine göre ayarla
base_path = r"C:\Users\ahmet\OneDrive\Desktop\Dogal_Dil_İşleme"

tfidf_models = {
    "tfidf_lemmatized": os.path.join(base_path, "tfidf_lemmatized.csv"),
    "tfidf_stemmed": os.path.join(base_path, "tfidf_stemmed.csv")
}

text_files = {
    "tfidf_lemmatized": os.path.join(base_path, "lemmatized_output.csv"),
    "tfidf_stemmed": os.path.join(base_path, "stemmed_output.csv")
}

# Giriş metni index'i (örnek olarak 0'dan başlayabilir)
input_index = 0  # İstersen başka satır seçebilirsin

results = []

for model_name, tfidf_path in tfidf_models.items():
    # TF-IDF matrisini oku
    tfidf_df = pd.read_csv(tfidf_path, index_col=0)
    # Metinleri oku
    text_df = pd.read_csv(text_files[model_name])

    # Giriş vektörü
    query_vector = tfidf_df.iloc[input_index].values.reshape(1, -1)

    # Cosine similarity hesapla
    similarities = cosine_similarity(query_vector, tfidf_df.values).flatten()

    # En yüksek 5 skor (kendisi hariç)
    top_indices = similarities.argsort()[::-1][1:6]
    top_scores = similarities[top_indices]
    top_texts = text_df.iloc[top_indices].iloc[:, 0].values  # ilk sütun metin

    for rank, (text, score) in enumerate(zip(top_texts, top_scores), start=1):
        results.append({
            "Model": model_name,
            "Rank": rank,
            "Benzer Metin": text,
            "Cosine Skoru": round(score, 4)
        })

# Sonuçları DataFrame yap
results_df = pd.DataFrame(results)
print(results_df)

# CSV olarak kaydet (opsiyonel)
results_df.to_csv(os.path.join(base_path, "tfidf_similarity_results.csv"), index=False)
