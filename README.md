# 0- entender o desafio e a empresa
# 1- importar a base de dados

 import pandas as pd
 from sklearn.preprocessing import LabelEncoder

 tabela = pd.read_csv("clientes.csv")

 display(tabela)

# pacotes de codigos = bibliotecas
#!pip install pandas scikit-learn

# 2- preparar a base de dados para a inteligencia artificial

codificador = LabelEncoder()

tabela["profissao"] = codificador.fit_transform(tabela["profissao"])
tabela["mix_credito"] = codificador.fit_transform(tabela["mix_credito"])
tabela["comportamento_pagamento"] = codificador.fit_transform(tabela["comportamento_pagamento"])


display(tabela.info())

y = tabela["score_credito"]
x = tabela.drop(columns=["score_credito", "id_cliente"])

from sklearn.model_selection import train_test_split

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y)

# 3- criar um modelo de IA -> nota/score de credito: bom, ok, ruim
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

modelo_arvoredecisao = RandomForestClassifier()
modelo_knn = KNeighborsClassifier()

modelo_arvoredecisao.fit (x_treino, y_treino)
modelo_knn.fit (x_treino, y_treino)

# 4- avaliar qual o melhor modelo de IA
previsao_arvoredecisao = modelo_arvoredecisao.predict(x_teste)
previsao_knn = modelo_knn.predict(x_teste)

from sklearn.metrics import accuracy_score

display(accuracy_score(y_teste, previsao_arvoredecisao))
display(accuracy_score(y_teste, previsao_knn))

# 5- fazer novas previsoes

tabela_novos_clientes = pd.read_csv("novos_clientes.csv")
display(tabela_novos_clientes)

tabela_novos_clientes["profissao"] = codificador.fit_transform(tabela_novos_clientes["profissao"])
tabela_novos_clientes["mix_credito"] = codificador.fit_transform(tabela_novos_clientes["mix_credito"])
tabela_novos_clientes["comportamento_pagamento"] = codificador.fit_transform(tabela_novos_clientes["comportamento_pagamento"])

previsao = modelo_arvoredecisao.predict(tabela_novos_clientes)
display(previsao)

