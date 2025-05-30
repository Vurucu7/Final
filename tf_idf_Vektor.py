import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Lemmatized ve stemmed dosya yolları
lemmatized_path = "lemmatized_output.csv"
stemmed_path = "stemmed_output.csv"

# CSV'leri oku
df_lemmatized = pd.read_csv(lemmatized_path)
df_stemmed = pd.read_csv(stemmed_path)

# TF-IDF vektörleştirici
vectorizer = TfidfVectorizer()

# Lemmatized veri için
tfidf_lemmatized = vectorizer.fit_transform(df_lemmatized["Lemmatized_Text"])
df_tfidf_lemmatized = pd.DataFrame(tfidf_lemmatized.toarray(), columns=vectorizer.get_feature_names_out())
df_tfidf_lemmatized.to_csv("tfidf_lemmatized.csv", index=False)

# Stemmed veri için
tfidf_stemmed = vectorizer.fit_transform(df_stemmed["Stemmed_Text"])
df_tfidf_stemmed = pd.DataFrame(tfidf_stemmed.toarray(), columns=vectorizer.get_feature_names_out())
df_tfidf_stemmed.to_csv("tfidf_stemmed.csv", index=False)

print("✅ TF-IDF dosyaları oluşturuldu:")
print("- tfidf_lemmatized.csv")
print("- tfidf_stemmed.csv")
