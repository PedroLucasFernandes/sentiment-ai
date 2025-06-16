import re
import unicodedata
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

def preprocessar(texto):
    texto = texto.lower()
    texto = unicodedata.normalize('NFD', texto).encode('ascii', 'ignore').decode('utf-8')
    texto = re.sub(r'[^\w\s]', '', texto)
    return texto

frases = [
    "Eu amei esse filme", "O produto é ótimo", "Que dia maravilhoso",
    "Gostei muito da comida", "O atendimento foi excelente", "Serviço de primeira qualidade",
    "Funcionou perfeitamente bem", "Recomendo para todos", "Entrega rápida e eficiente",
    "Muito feliz com o resultado", "Experiência incrível", "Muito satisfeito com a compra",
    "Ambiente agradável e limpo", "Funcionários super atenciosos", "Maravilhoso, nota 10",
    "Sensacional, voltarei com certeza", "Produto excelente, superou as expectativas",
    "Nota 10, tudo perfeito", "Recomendo de olhos fechados", "Melhor escolha que fiz",
    "Achei legal", "Foi bom", "Funcionou como deveria", "Fiquei contente", "Está ótimo",
    
    "Eu odiei isso", "O atendimento foi péssimo", "Não gostei do serviço",
    "Horrível, perdi meu tempo", "Aplicativo cheio de erros", "Nada funcionou direito",
    "Muito ruim, não recomendo", "A comida estava fria", "Demorou demais para chegar",
    "Me arrependi de ter comprado", "Não atendeu às expectativas", "Experiência decepcionante",
    "Serviço horrível", "O site é confuso e lento", "Fui mal tratado",
    "Produto veio com defeito", "Pior compra que já fiz", "Não volto nunca mais",
    "Lixo de atendimento", "Total desrespeito com o cliente", "Filme péssimo", "Filme horrível",
    "Filme ruim demais", "Decepcionante", "Não gostei mesmo", "Perdi dinheiro",

    "Recebi o produto", "Está funcionando", "É o que eu pedi", "Foi entregue ontem",
    "O produto chegou", "Está aqui comigo", "Testei ontem", "É isso", "Comprei semana passada"
]

# 0 = negativo, 1 = positivo, 2 = neutro
sentimentos = [1]*25 + [0]*26 + [2]*9
frases_proc = [preprocessar(f) for f in frases]

vetorizador = SentenceTransformer('paraphrase-MiniLM-L6-v2')
x = vetorizador.encode(frases_proc)

modelo = LogisticRegression(max_iter=1000)
modelo.fit(x, sentimentos)

joblib.dump(modelo, 'modelo_sentimentos.pkl')
joblib.dump(vetorizador, 'vetorizador.pkl')

print("Modelo e vetorizador salvos com sucesso!")