import re
import unicodedata
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

def preprocessar(texto):
    texto = texto.lower()
    texto = unicodedata.normalize('NFD', texto).encode('ascii', 'ignore').decode('utf-8')
    texto = re.sub(r'[^\w\s]', '', texto)
    return texto

df = pd.read_csv('dataset.csv')

df['frase_processada'] = df['frase'].apply(preprocessar)

vetorizador = SentenceTransformer('all-mpnet-base-v2')
X = vetorizador.encode(df['frase_processada'].tolist())

le = LabelEncoder()
y = le.fit_transform(df['sentimento'])

modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X, y)

joblib.dump(modelo, 'modelo_sentimentos.pkl')
joblib.dump(vetorizador, 'vetorizador.pkl')
joblib.dump(le, 'label_encoder.pkl')

print("Modelo treinado e salvo com sucesso!")