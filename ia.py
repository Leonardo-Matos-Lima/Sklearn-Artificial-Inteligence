import pandas as pd

# Importar a base de dados
dados = pd.read_csv('clientes.csv')

# Preparar a base de dados para a inteligencia artificial
from sklearn.preprocessing import LabelEncoder

codificador = LabelEncoder()
dados['profissao'] = codificador.fit_transform(dados["profissao"])
dados['mix_credito'] = codificador.fit_transform(dados["mix_credito"])
dados['comportamento_pagamento'] = codificador.fit_transform(dados["comportamento_pagamento"])

y = dados['score_credito']
x = dados.drop(columns=['id_cliente', 'score_credito'])


from sklearn.model_selection import train_test_split

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3)

# Criar o modelo de IA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import kneighbors_graph

modelo1 = RandomForestClassifier
modelo2 = kneighbors_graph

modelo1.fit(x_treino, y_treino)
modelo2.fit(x_treino, y_treino)

# Escolher o melhor modelo IA
previsaoM1 = modelo1.predict(x_teste)
previsaoM2 = modelo2.predict(x_teste)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_teste, previsaoM1))
print(accuracy_score(y_teste, previsaoM2))

# Usar a IA para definir o Score de credito dos clientes