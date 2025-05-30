import pandas as pd
import ast
import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

# NLTK veri setlerini indir (bir kereye mahsus)
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# 1. CSV dosyasını yükle
csv_path = "C:/Users/ahmet/OneDrive/Desktop/Dogal_Dil_İşleme/adverse_drug_35.csv"  # Dosya yolu

# Büyük dosya olduğundan parçalı okuyacağız
chunks = []
for chunk in pd.read_csv(csv_path, usecols=['patient.drug', 'patient.reaction'], chunksize=5000):
    chunks.append(chunk.dropna())

df_raw = pd.concat(chunks, ignore_index=True)

# 2. Güvenli JSON ayrıştırıcılar
def safe_parse_json_list(field):
    try:
        data = ast.literal_eval(field)
        if isinstance(data, list) and len(data) > 0:
            return data
    except Exception:
        return None
    return None

def extract_drugname(drug_field):
    drugs = safe_parse_json_list(drug_field)
    if drugs:
        return drugs[0].get('medicinalproduct', None)
    return None

def extract_reaction(reaction_field):
    reactions = safe_parse_json_list(reaction_field)
    if reactions:
        return reactions[0].get('reactionmeddrapt', None)
    return None

# 3. İlacı ve yan etkiyi çıkar
print("İlaç ve yan etki bilgileri çıkarılıyor...")
df_raw['drugname'] = df_raw['patient.drug'].apply(extract_drugname)
df_raw['reaction'] = df_raw['patient.reaction'].apply(extract_reaction)

# Geçerli kayıtları al
df_clean = df_raw[['drugname', 'reaction']].dropna()

# 4. Lemmatizasyon ve stem
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def normalize_text(text):
    text = text.lower()
    words = word_tokenize(text)
    lemmatized = [lemmatizer.lemmatize(word) for word in words]
    stemmed = [stemmer.stem(word) for word in lemmatized]
    return ' '.join(stemmed)

print("Normalize işlemi uygulanıyor...")
df_clean['drugname_normalized'] = df_clean['drugname'].astype(str).apply(normalize_text)
df_clean['reaction_normalized'] = df_clean['reaction'].astype(str).apply(normalize_text)

# 5. Final veri çerçevesi
final_df = df_clean[['drugname_normalized', 'reaction_normalized']].rename(columns={
    'drugname_normalized': 'drugname',
    'reaction_normalized': 'reaction'
})

# 6. CSV'ye kaydet
save_path = "C:/Users/ahmet/OneDrive/Desktop/Dogal_Dil_İşleme/normalized_adverse_events.csv"
final_df.to_csv(save_path, index=False)

print(f"\n✅ Temizleme tamamlandı! Kaydedilen dosya: {save_path}")
