
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import os

# Veri Yolu (önce csv'yi yükle)
lemmatized_csv_path = "lemmatized_output.csv"
df_lemmatized = pd.read_csv(lemmatized_csv_path).iloc[:500]  # örneklem

# Giriş metni (zenginleştirilmiş versiyon)
query_text = "acute bacterial prostatitis causing severe pelvic pain and fever"

# Ortalama vektör çıkarma fonksiyonu
def get_avg_vector(model, text):
    tokens = text.lower().split()
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

# Benzerlik hesaplama fonksiyonu
def word2vec_similarity(model, text_series, query_text):
    query_vec = get_avg_vector(model, query_text)
    all_vecs = np.vstack([get_avg_vector(model, t) for t in text_series])
    similarities = cosine_similarity([query_vec], all_vecs).flatten()
    top_indices = similarities.argsort()[::-1][1:6]
    return text_series.iloc[top_indices].values, similarities[top_indices]

# Modellerin bulunduğu klasör
model_dir = "trained_models"
model_files = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith(".model")]

# Sonuçları topla
results = []
for model_path in model_files:
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    model = Word2Vec.load(model_path)
    top_texts, top_scores = word2vec_similarity(model, df_lemmatized["Lemmatized_Text"], query_text)
    for rank, (text, score) in enumerate(zip(top_texts, top_scores), start=1):
        results.append({
            "Model": model_name,
            "Rank": rank,
            "Benzer Metin": text,
            "Cosine Skoru": round(score, 4)
        })

# Sonuçları kaydet
results_df = pd.DataFrame(results)
results_df.to_csv("word2vec_similarity_results.csv", index=False)
print("✔️ Benzerlik hesaplamaları tamamlandı ve CSV'ye kaydedildi.")
