import re
import unicodedata
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

def preprocessar(texto):
    texto = texto.lower()
    texto = unicodedata.normalize('NFD', texto).encode('ascii', 'ignore').decode('utf-8')
    texto = re.sub(r'[^\w\s]', '', texto)
    return texto

frases = [
    # Positivas
    "Eu amei esse filme", "O produto é ótimo", "Que dia maravilhoso",
    "Gostei muito da comida", "O atendimento foi excelente", "Serviço de primeira qualidade",
    "Funcionou perfeitamente bem", "Recomendo para todos", "Entrega rápida e eficiente",
    "Muito feliz com o resultado", "Experiência incrível", "Muito satisfeito com a compra",
    "Ambiente agradável e limpo", "Funcionários super atenciosos", "Maravilhoso, nota 10",
    "Sensacional, voltarei com certeza", "Produto excelente, superou as expectativas",
    "Nota 10, tudo perfeito", "Recomendo de olhos fechados", "Melhor escolha que fiz",

    # Negativas
    "Eu odiei isso", "O atendimento foi péssimo", "Não gostei do serviço",
    "Horrível, perdi meu tempo", "Aplicativo cheio de erros", "Nada funcionou direito",
    "Muito ruim, não recomendo", "A comida estava fria", "Demorou demais para chegar",
    "Me arrependi de ter comprado", "Não atendeu às expectativas", "Experiência decepcionante",
    "Serviço horrível", "O site é confuso e lento", "Fui mal tratado",
    "Produto veio com defeito", "Pior compra que já fiz", "Não volto nunca mais",
    "Lixo de atendimento", "Total desrespeito com o cliente",
    "Que filme péssimo", "Filme horrível", "Filme ruim demais"
]

sentimentos = [1]*20 + [0]*23
frases_processadas = [preprocessar(f) for f in frases]

modelo_vetorizador = SentenceTransformer('paraphrase-MiniLM-L6-v2')

x = modelo_vetorizador.encode(frases_processadas)

modelo = LogisticRegression()
modelo.fit(x, sentimentos)

entrada = input("Digite uma frase: ")
entrada_proc = preprocessar(entrada)
entrada_vetor = modelo_vetorizador.encode([entrada_proc])

resultado = modelo.predict(entrada_vetor)

if resultado[0] == 1:
    print("Sentimento: Positivo 😊")
else:
    print("Sentimento: Negativo 😠")