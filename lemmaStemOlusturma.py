import json
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Gerekli nltk verilerini indir (ilk çalıştırmada)
nltk.download('stopwords')
nltk.download('wordnet')

# Araçlar
tokenizer = TreebankWordTokenizer()
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# JSON dosyasını oku
with open("drug-event-0016-of-0034.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Yan etki metinlerini topla
texts = []
for entry in data["results"]:
    reactions = entry.get("patient", {}).get("reaction", [])
    for r in reactions:
        if "reactionmeddrapt" in r:
            texts.append(r["reactionmeddrapt"])

# Ön işleme fonksiyonu
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = tokenizer.tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    return tokens

# Lemmatize ve Stem işlemlerini uygula
lemmatized_list = []
stemmed_list = []

for text in texts:
    tokens = preprocess(text)
    lemmatized = [lemmatizer.lemmatize(t) for t in tokens]
    stemmed = [stemmer.stem(t) for t in tokens]

    lemmatized_list.append(" ".join(lemmatized))
    stemmed_list.append(" ".join(stemmed))

# DataFrame oluştur
df_lemmatized = pd.DataFrame({'Lemmatized_Text': lemmatized_list})
df_stemmed = pd.DataFrame({'Stemmed_Text': stemmed_list})

# CSV olarak kaydet
df_lemmatized.to_csv("lemmatized_output.csv", index=False)
df_stemmed.to_csv("stemmed_output.csv", index=False)

print("✅ İki dosya başarıyla oluşturuldu:")
print("- lemmatized_output.csv")
print("- stemmed_output.csv")
