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

entrada = input("Digite uma frase: ")
entrada_proc = preprocessar(entrada)
entrada_vetor = vetorizador.encode([entrada_proc])

resultado = modelo.predict(entrada_vetor)

if resultado[0] == 1:
    print("Sentimento: Positivo 😊")
elif resultado[0] == 0:
    print("Sentimento: Negativo 😠")
else:
    print("Sentimento: Neutro 😐")