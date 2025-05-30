import json
import re
import nltk
import pandas as pd
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Gerekli veri setlerini indir (ilk kullanımda gerekir)
nltk.download('stopwords')
nltk.download('wordnet')

# Araçlar ve ayarlar
stop_words = set(stopwords.words('english'))
tokenizer = TreebankWordTokenizer()
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# JSON verisini yükle
with open("drug-event-0016-of-0034.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# İlk 20 reaction açıklamasını al
texts = []
for entry in data["results"]:
    reactions = entry.get("patient", {}).get("reaction", [])
    for r in reactions:
        if "reactionmeddrapt" in r:
            texts.append(r["reactionmeddrapt"])
    if len(texts) >= 20:
        break

# Ön işleme adımlarını uygula
rows = []

for text in texts:
    original = text
    text = text.lower()  # lowercase
    text = re.sub(r'[^\w\s]', '', text)  # noktalama temizliği
    tokens = tokenizer.tokenize(text)  # tokenization
    no_stop = [t for t in tokens if t not in stop_words]  # stopword temizliği
    lemmatized = [lemmatizer.lemmatize(w) for w in no_stop]
    stemmed = [stemmer.stem(w) for w in no_stop]

    rows.append({
        "Original": original,
        "Tokens": tokens,
        "No_Stopwords": no_stop,
        "Lemmatized": lemmatized,
        "Stemmed": stemmed
    })

# DataFrame olarak göster
df = pd.DataFrame(rows)
print(df.head(10))
