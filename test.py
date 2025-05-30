import os
from gensim.models import Word2Vec

# Hedef klasör (model dosyalarının bulunduğu klasör)
model_dir = "models"

# Test etmek istediğin kelime
test_word = "pain"

# Klasördeki tüm .model dosyalarını bul
model_files = [f for f in os.listdir(model_dir) if f.endswith(".model")]

for model_file in model_files:
    model_path = os.path.join(model_dir, model_file)
    try:
        model = Word2Vec.load(model_path)
        if test_word in model.wv:
            similar_words = model.wv.most_similar(test_word, topn=5)
            print(f"\n🔎 {model_file} — '{test_word}' kelimesine en yakın 5 kelime:")
            for word, score in similar_words:
                print(f"   {word} (benzerlik: {score:.4f})")
        else:
            print(f"\n⚠️ {model_file}: '{test_word}' kelimesi bu modelde bulunamadı.")
    except Exception as e:
        print(f"\n⛔ Hata ({model_file}): {e}")
