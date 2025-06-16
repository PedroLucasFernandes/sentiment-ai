from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

frases = [
    # Positivas
    "Eu amei esse filme", "O produto é ótimo", "Que dia maravilhoso", "Gostei muito da comida",
    "O atendimento foi excelente", "Serviço de primeira qualidade", "O aplicativo é muito bom", "Funcionou perfeitamente bem",
    "Recomendo para todos", "Entrega rápida e eficiente", "Tudo funcionou como esperado", "Muito feliz com o resultado",
    "Experiência incrível", "Muito satisfeito com a compra", "Foi melhor do que imaginei", "Ambiente agradável e limpo",
    "Funcionários super atenciosos", "Estou encantado com o atendimento", "Maravilhoso, nota 10", "Sensacional, voltarei com certeza",

    # Negativas
    "Eu odiei isso", "O atendimento foi péssimo", "Não gostei do serviço", "Horrível, perdi meu tempo",
    "Aplicativo cheio de erros", "Nada funcionou direito", "Muito ruim, não recomendo", "A comida estava fria",
    "Demorou demais para chegar", "Me arrependi de ter comprado", "Não atendeu às expectativas", "Experiência decepcionante",
    "Serviço horrível", "O site é confuso e lento", "Fui mal tratado", "Produto veio com defeito",
    "Pior compra que já fiz", "Não volto nunca mais", "Lixo de atendimento", "Total desrespeito com o cliente"
]

sentimentos = [1]*20 + [0]*20

# Transformar texto em vetores
vetor = CountVectorizer()
x = vetor.fit_transform(frases)

# Criar o modelo
modelo = MultinomialNB()
modelo.fit(x, sentimentos)

# Testar
entrada = input("Digite uma frase: ")
entrada_vetor = vetor.transform([entrada])
resultado = modelo.predict(entrada_vetor)

if resultado[0] == 1:
    print("Sentimento: Positivo 😊")
else:
    print("Sentimento: Negativo 😠")