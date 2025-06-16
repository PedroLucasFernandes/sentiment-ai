import re
import unicodedata
import joblib

def preprocessar(texto):
    texto = texto.lower()
    texto = unicodedata.normalize('NFD', texto).encode('ascii', 'ignore').decode('utf-8')
    texto = re.sub(r'[^\w\s]', '', texto)
    return texto

modelo = joblib.load('modelo_sentimentos.pkl')
vetorizador = joblib.load('vetorizador.pkl')
le = joblib.load('label_encoder.pkl')

entrada = input("Digite uma frase: ")
entrada_proc = preprocessar(entrada)
entrada_vetor = vetorizador.encode([entrada_proc])

resultado = modelo.predict(entrada_vetor)
sentimento = le.inverse_transform(resultado)[0]

if sentimento == 1:
    print("Sentimento: Positivo 😊")
elif sentimento == 0:
    print("Sentimento: Negativo 😠")
else:
    print("Sentimento: Neutro 😐")