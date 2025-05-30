import pandas as pd
import os
from gensim.models import Word2Vec

# Dosya yolları
lemm_path = "lemmatized_output.csv"
stem_path = "stemmed_output.csv"

# Kaydedilecek klasör
output_dir = "trained_models"
os.makedirs(output_dir, exist_ok=True)

# Tokenize et
def tokenize_column(file_path, column_name):
    df = pd.read_csv(file_path)
    return [text.lower().split() for text in df[column_name].dropna()]

lemmatized_sentences = tokenize_column(lemm_path, "Lemmatized_Text")
stemmed_sentences = tokenize_column(stem_path, "Stemmed_Text")

# Parametre varyasyonları
configs = [
    ("cbow", 0), ("skipgram", 1)
]
windows = [2, 4]
dims = [100, 300]

# Eğitim fonksiyonu
def train_models(sentences, tag):
    for sg_name, sg_val in configs:
        for win in windows:
            for dim in dims:
                model = Word2Vec(sentences=sentences, vector_size=dim, window=win, sg=sg_val, min_count=1, workers=4)
                model_name = f"word2vec_{tag}_{sg_name}_win{win}_dim{dim}.model"
                model.save(os.path.join(output_dir, model_name))
                print(f"✔️ Model kaydedildi: {model_name}")

# Modelleri eğit
train_models(lemmatized_sentences, "lemmatized")
train_models(stemmed_sentences, "stemmed")
